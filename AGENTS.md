# 🤖 AGENTS.md — Multi-Agent Architecture & Conversation Log

This document provides a comprehensive record of the AI Agent architecture powering the **ShopGenie E-Commerce Chatbot**, followed by a transcript and operational log of the enhancement session conducted to modernize the user interface, strengthen routing, and document the project.

---

## 1. Multi-Agent System Architecture

The chatbot is structured as a decoupled multi-agent decision and retrieval system. Rather than funneling every user query through an expensive monolithic LLM prompt, the system employs specialized agents coordinated by a zero-shot semantic router.

```
                    ┌─────────────────────────┐
                    │       User Query        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Semantic Router Agent  │
                    │   (all-MiniLM-L6-v2)    │
                    └──────┬─────┬─────┬──────┘
                           │     │     │
            ┌──────────────┘     │     └──────────────┐
            ▼                    ▼                    ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ SQL Catalog     │  │ FAQ Vector RAG  │  │ Small-Talk      │
   │ Agent           │  │ Agent           │  │ Agent           │
   │ (NL-to-SQL +    │  │ (ChromaDB +     │  │ (Persona +      │
   │ Comprehension)  │  │ Groq LLM)       │  │ Fallback)       │
   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
            │                    │                    │
            └──────────────┬─────┴────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │ Streamlit UI Controller │
              │ (Badges, Links, SQL exp)│
              └─────────────────────────┘
```

---

## 2. Specialized Agent Specifications

### Agent 1: Semantic Router Agent (`app/router.py`)
- **Role**: Intent classification and query dispatcher.
- **Underlying Model**: `sentence-transformers/all-MiniLM-L6-v2` via Aurelio AI's `semantic-router`.
- **Latency Profile**: Sub-50ms (local CPU embedding inference).
- **Target Routes**:
  1. `sql`: Triggers catalog search queries (e.g., shoe price, brand, rating, discount filters).
  2. `faq`: Triggers store policy queries (e.g., returns, refunds, payment methods, delivery).
  3. `small-talk`: Triggers conversational banter, greetings, and system identity queries.
- **Fallback Policy**: In cases where cosine similarity falls below the router threshold, the query automatically routes to the Conversational Agent to guarantee a graceful, friendly response.

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
