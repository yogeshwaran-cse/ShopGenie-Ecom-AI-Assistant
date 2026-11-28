# 🤖 AGENTS.md — Multi-Agent Architecture & Conversation Log

This document provides a comprehensive record of the AI Agent architecture powering the **ShopGenie E-Commerce Chatbot**, followed by a transcript and operational log of the enhancement session conducted to modernize the user interface, strengthen routing, and document the project.

---

## 1. Multi-Agent System Architecture

The chatbot is structured as a decoupled multi-agent decision and retrieval system using a **Semantic-First Tiered Router** with **Multi-Intent support** and **Conversational Filter Memory**, orchestrated by a LangGraph `StateGraph`.

```
                    ┌─────────────────────────┐
                    │       User Query        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  TIER 1: Semantic Gate  │
                    │  (all-MiniLM-L6-v2)     │
                    │  cosine similarity      │
                    └──────┬──────────┬───────┘
                           │          │
              score≥0.78   │          │  ambiguous /
              gap≥0.12     │          │  low confidence
                           │          │
              ⚡ Fast Path  │          │  🧠 Escalate
                           │          ▼
                           │  ┌──────────────────┐
                           │  │ TIER 2: LLM      │
                           │  │ Multi-Intent      │
                           │  │ Router (Groq)     │
                           │  │ + Filter Extract  │
                           │  └────────┬─────────┘
                           │           │
                           ▼           ▼
                    ┌─────────────────────────┐
                    │   Dispatch Node         │
                    │  (fans out to 1+ agents)│
                    └──┬──────┬──────┬────────┘
                       │      │      │
          ┌────────────┘      │      └────────────┐
          ▼                   ▼                   ▼
 ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
 │ SQL Catalog     │ │ FAQ Vector RAG  │ │ Small-Talk      │
 │ Agent           │ │ Agent           │ │ Agent           │
 │ (NL-to-SQL +    │ │ (ChromaDB +     │ │ (Persona +      │
 │ Comprehension)  │ │ Groq LLM)       │ │ Fallback)       │
 └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
          │                   │                   │
          └───────────┬───────┴───────────────────┘
                      ▼
           ┌──────────────────────┐
           │ Synthesizer Node     │  (multi-route only)
           │ Merges sql + faq     │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │ Streamlit UI         │
           │ (Tier badges,        │
           │  route pills,        │
           │  SQL inspector,      │
           │  filter sidebar)     │
           └──────────────────────┘
```

---

## 2. Specialized Agent Specifications

### Agent 1: Semantic-First Tiered Router (`app/graph.py`)

#### Tier 1 — Semantic Gate Node (`semantic_gate_node`)
- **Role**: Zero-API-call intent classification via embedding cosine similarity.
- **Underlying Model**: `sentence-transformers/all-MiniLM-L6-v2` (local CPU inference).
- **Latency Profile**: Sub-50ms.
- **Fast-Path Criteria**: `top_score >= 0.78` AND `(top_score - second_score) >= 0.12`.
  - Sets `routing_source = "semantic_fast_path"`, bypasses LLM entirely.
- **Escalation**: When scores are ambiguous or below threshold, forwards to Tier 2.

#### Tier 2 — LLM Multi-Intent Router Node (`llm_router_node`)
- **Role**: Complex/ambiguous/multi-intent classification with filter extraction.
- **Underlying Model**: Groq (`GROQ_MODEL`, default `openai/gpt-oss-20b`).
- **Capabilities**:
  - Classifies queries into **one or more** routes: `sql`, `faq`, `small-talk`.
  - Extracts and merges `active_filters` (brand, max_price, min_rating, category) across conversation turns.
  - Sets `routing_source = "llm_escalated"`.
- **Fallback Policy**: If Groq API fails, falls back to highest cosine label from Tier 1 with `routing_source = "semantic_fallback"`.

#### Dispatch & Synthesis
- **Dispatch Node**: Fans out execution to each active route's specialist agent.
- **Synthesizer Node**: When `len(routes) > 1`, merges `sql_result` and `faq_result` into a single cohesive `final_response` via LLM, avoiding duplicate greetings.
- **Finalize Node**: For single-route results, promotes the relevant result field to `final_response`.

#### GraphState Schema
```python
class GraphState(TypedDict):
    query: str
    chat_history: list[dict]
    routes: list[Literal["sql", "faq", "small-talk"]]
    routing_source: Literal["semantic_fast_path", "llm_escalated", "semantic_fallback"]
    confidence_score: float
    active_filters: dict   # brand, max_price, min_rating, category
    sql_query: str
    sql_result: str
    faq_result: str
    final_response: str
```

---

### Agent 2: SQL Catalog Database Agent (`app/sql.py`)
- **Role**: Dynamic Natural Language to SQL generation, execution, and data comprehension.
- **Underlying Model**: Groq (`openai/gpt-oss-20b`, configurable via `GROQ_MODEL`).
- **Database**: SQLite3 (`app/db.sqlite`), table `product` (903 footwear products from Flipkart).
- **Schema Context**:
  - `product_link` (string): Hyperlink to the product listing.
  - `title` (string): Title of the product.
  - `brand` (string): Brand name (e.g., Nike, Campus, Puma, Sparx, Aadi).
  - `price` (integer): Price in Indian Rupees (INR).
  - `discount` (float): Fractional discount (e.g., 0.3 for 30%).
  - `avg_rating` (float): Rating from 0.0 to 5.0.
  - `total_ratings` (integer): Volume of customer reviews.
- **Execution Pipeline**:
  1. Formulates `<SQL> SELECT ... </SQL>` query respecting schema constraints (uses `%LIKE%` for case insensitivity).
  2. Executes read-only queries safely via SQLite.
  3. Formulates a natural language recommendation list with prices, discounts, ratings, and product links.

---

### Agent 3: FAQ Vector RAG Agent (`app/faq.py`)
- **Role**: Semantic knowledge retrieval for customer support policies.
- **Vector Store**: ChromaDB (`faqs` collection, in-memory client).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Knowledge Base**: `app/resources/faq_data.csv` (10 core store policies).
- **Pipeline**:
  1. Retrieves top-2 most similar question-answer pairs.
  2. Context is injected into a grounded Groq prompt with strict guardrails (`If the answer is not found in the context, kindly state "I don't know"`).

---

### Agent 4: Conversational Small-Talk Agent (`app/smalltalk.py`)
- **Role**: General conversational AI and personality provider.
- **Prompt Guardrails**: Friendly, polite, and helpful e-commerce assistant persona.

---

## 3. Session Conversation & Task Transcript

### User Request
> *"want this in clean ui and document it in README.md and save this convo in AGENTS.md"*

### Operational Goals Identified:
1. **Clean UI Overhaul**:
   - Modernize the bare Streamlit interface in `app/main.py`.
   - Add modern typography, hero header, responsive status indicators, and clean CSS styling.
   - Introduce visual route badges (`Product Catalog`, `Store Knowledge Base`, `Conversational Assistant`).
   - Add a collapsible SQL query inspector so users and developers can view the generated database queries.
   - Convert long raw product URLs into clean, clickable `[🔗 View on Flipkart]` markdown buttons.
   - Add an interactive sidebar with live dataset metrics, clickable quick-starter prompts, system toggles, and chat reset functionality.
   - Cache ChromaDB knowledge base ingestion so it does not reload on every Streamlit rerun.
2. **Comprehensive Project Documentation (`README.md`)**:
   - Document project overview, architecture diagram, tech stack, installation, environment setup, and example queries.
3. **Session & Architecture Record (`AGENTS.md`)**:
   - Record multi-agent contracts, prompt structures, session history, and guidance for future AI agents.
4. **Environment & Dependency Standardization**:
   - Create `requirements.txt` and `.env.example` at repository root.

---

### Actions Taken & Code Modifications

#### Step 1: Codebase Inspection & Verification
- Examined project tree: `app/main.py`, `app/router.py`, `app/faq.py`, `app/sql.py`, `app/smalltalk.py`, and `app/db.sqlite`.
- Inspected database contents: Confirmed 903 products loaded in SQLite.
- Verified Groq API and sentence-transformer embeddings execution in the workspace environment.

#### Step 2: Router Utterance Expansion (`app/router.py`)
- **Issue**: Standard greetings like `"Hello"`, `"Hey there"`, or specific catalog questions were previously unrepresented in the seed utterances, occasionally leading to `None` route matches.
- **Enhancement**: Enriched the semantic training utterances across `faq`, `sql`, and `small-talk` routes to dramatically improve routing recall.

#### Step 3: SQL Chain Enhancement (`app/sql.py`)
- **Enhancement**:
  - Added `return_sql=False` parameter to `sql_chain` to optionally return the generated raw SQL query along with the synthesized response without breaking backward compatibility.
  - Implemented graceful handling for empty search results (`response.empty`), advising users to adjust filters rather than hallucinating.

#### Step 4: Streamlit UI Redesign (`app/main.py`)
- **Enhancement**:
  - Implemented custom CSS with modern gradients, hero headers, and pill tags.
  - Added sidebar controls: Quick starter prompt buttons, live catalog statistics, debug toggle, and conversation clear button.
  - Created automatic link cleaner regex `clean_and_format_text` to convert raw Flipkart URLs into formatted markdown links.
  - Added expandable SQL query inspection block for product searches.
  - Added `@st.cache_resource` on `initialize_knowledge_base` to prevent duplicate ChromaDB ingestion.

#### Step 5: Dependency and Environment Files
- Created `requirements.txt` listing `streamlit`, `groq`, `chromadb`, `sentence-transformers`, `semantic-router`, `pandas`, and `python-dotenv`.
- Created `.env.example` showing required keys `GROQ_API_KEY` and `GROQ_MODEL`.

#### Step 6: Full Testing & Verification
- Ran test queries across all 3 routes:
  - Route `sql`: `"Show top 2 puma shoes"` -> Routed to `sql`, generated SQL query, executed against SQLite, and returned formatted recommendations with Flipkart links.
  - Route `faq`: Tested order tracking and return policy queries -> ChromaDB returned relevant context, Groq synthesized exact answer.
  - Route `small-talk`: Tested greetings -> Groq responded warmly and politely.
  - Validated Python compilation across all scripts without errors.

---

## 4. Instructions for Future Agents & Contributors

When extending or maintaining this project:
1. **Adding Database Fields**: If new columns are added to `product` table in `db.sqlite`, remember to update `sql_prompt` in `app/sql.py` so the LLM knows the column definitions and data types.
2. **Adding FAQ Articles**: Add new rows to `app/resources/faq_data.csv`. The ChromaDB collection will automatically index them on initial application startup.
3. **Adding New Routes**: When creating a new route (e.g., `order-tracking` or `support-ticket`):
   - Define the route and utterances in `app/router.py`.
   - Create the corresponding handler chain.
   - Add the routing condition and badge style in `app/main.py`.
4. **Environment Variables**: Always verify `GROQ_API_KEY` is set in `app/.env` before launching Streamlit.
5. **Token Limits & Query Bounds**: Never allow unbounded queries like `SELECT * FROM product` to feed directly into LLM comprehension. Always enforce `LIMIT 5` in SQL and enforce code-level `head(5)` slicing to respect Groq TPM thresholds (8,000 TPM limit on free tier).

---

## 5. Incident Log: Token Limit Overrun (413 Rate Limit Error)

### Symptom
```
groq.APIStatusError: Error code: 413 - {'error': {'message': 'Request too large for model openai/gpt-oss-20b ... on tokens per minute (TPM): Limit 8000, Requested 86908, please reduce your message size and try again.'}}
```

### Root Cause
When executing broad product queries (e.g., *"Find Nike shoes that have discount"* or *"Show me shoes"*), the generated SQL lacked a `LIMIT` clause or matched hundreds of rows from `db.sqlite`. Passing hundreds of products with long Flipkart URLs directly into the `data_comprehension` prompt resulted in ~86,908 tokens, instantly exceeding Groq's 8,000 TPM rate limit for `openai/gpt-oss-20b`.

### Resolution Applied
1. **SQL Prompt Guardrail**: Added strict `LIMIT` rule to `sql_prompt` in `app/sql.py`:
   > *"ALWAYS include a LIMIT clause in your SQL query. If the user does not specify a count, default to LIMIT 5 (maximum LIMIT 8). Never return an unbounded query without a LIMIT clause."*
2. **Code-Level Safety Cap**: Added `response.head(5)` in `sql_chain` to enforce a hard cap before converting rows to dictionary context.
3. **Fault-Tolerant Synthesis Fallback**: Wrapped `data_comprehension` in `try...except`. If a rate limit or API error occurs, the system automatically formats the products directly from the SQL records without requiring an LLM call.
4. **Global Safe Error Handling**: Added `try...except` to `generate_sql_query()`, `faq.py`, `smalltalk.py`, and `main.py::ask()` to guarantee the Streamlit UI never crashes with unhandled tracebacks.

---

## 6. Incident Log: Streamlit Community Cloud Deployment ImportError

### Symptom
```
ImportError: This app has encountered an error. The original error message is redacted to prevent data leaks.
File "/mount/src/shopgenie-ecom-ai-assistant/app/main.py", line 8, in <module>
    from router import router
File "/mount/src/shopgenie-ecom-ai-assistant/app/router.py", line 1, in <module>
    from semantic_router import Route, RouteLayer
```

### Root Cause
1. **API Refactor in `semantic-router`**: In `semantic-router` 0.3.x, `RouteLayer` moved from root to `semantic_router.layer`. In 1.x+, `RouteLayer` was deprecated and replaced by `SemanticRouter`.
2. **Subfolder Execution Path**: Streamlit Cloud runs apps from the repo root (`/mount/src/...`), requiring explicit `sys.path.insert(0, ...)` to reliably resolve module imports inside `app/`.
3. **Requirement Discovery**: Streamlit Cloud looks for `requirements.txt` in the root or in the subfolder containing `main.py`. Without `app/requirements.txt`, package installations can be skipped depending on deployment configuration.

### Resolution Applied
1. **Multi-Version Router Imports**: Added tiered fallback imports in `app/router.py` supporting `0.0.x`, `0.3.x`, and `1.x` syntax.
2. **Autonomous Fallback Router**: Implemented a standalone `FallbackRouter` in `app/router.py` using `SentenceTransformer` and cosine similarity directly. If `semantic_router` is missing or fails on Linux, the router switches seamlessly with identical `.name` outputs.
3. **Path Resolution & Secrets Sync**: Added `sys.path.insert(0, ...)` and `st.secrets` synchronization into `os.environ` in `app/main.py`.
4. **Mirrored Requirements**: Created `app/requirements.txt` mirroring the root `requirements.txt`.
5. **Safe Environment Defaults**: Used `os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b')` across `sql.py`, `faq.py`, and `smalltalk.py`.
