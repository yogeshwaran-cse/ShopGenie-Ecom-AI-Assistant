# 🛍️ ShopGenie — Intelligent E-Commerce AI Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Groq API](https://img.shields.io/badge/Groq-Powered_LLM-orange.svg)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green.svg)](https://www.trychroma.com)
[![Semantic Router](https://img.shields.io/badge/Semantic_Router-Aurelio_AI-purple.svg)](https://github.com/aurelio-labs/semantic-router)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, multi-route conversational AI shopping assistant designed for e-commerce platforms. **ShopGenie** seamlessly navigates between natural language product searches, customer support FAQs, and friendly small-talk by combining **Semantic Routing**, **Text-to-SQL database querying**, and **ChromaDB vector Retrieval-Augmented Generation (RAG)** into a sleek, modern UI.

---

## 📸 User Interface Preview

![Product Search & Chat UI](app/resources/product-ss.png)

---

## 🏗️ Architecture & How It Works

The assistant operates on a decoupled multi-agent routing architecture:

```mermaid
flowchart TD
    User([🧑‍💻 User Query]) --> Router["🧠 Semantic Router<br/>(all-MiniLM-L6-v2)"]
    
    Router -->|Intent: sql| SQLAgent["🔍 SQL Catalog Agent"]
    Router -->|Intent: faq| FAQAgent["📚 FAQ RAG Agent"]
    Router -->|Intent: small-talk / fallback| TalkAgent["💬 Small-Talk Agent"]
    
    subgraph SQL Pipeline
        SQLAgent --> GenSQL["Prompt -> Groq LLM<br/>(Generate SQL)"]
        GenSQL --> ExecSQL["Execute on SQLite DB<br/>(product table: 903 items)"]
        ExecSQL --> CompSQL["LLM Comprehension<br/>(Format Results & Links)"]
    end
    
    subgraph FAQ RAG Pipeline
        FAQAgent --> ChromaQuery["ChromaDB Vector Query<br/>(Cosine Similarity)"]
        ChromaQuery --> GenFAQ["Context + Question -> Groq LLM"]
    end
    
    subgraph Conversational Pipeline
        TalkAgent --> GenTalk["Groq LLM Persona Prompt"]
    end

    CompSQL --> UI["🎨 Modern Streamlit UI<br/>(Badges, Clean Links, SQL Expander)"]
    GenFAQ --> UI
    GenTalk --> UI
```

### High-Level Architecture Diagram
![System Architecture](app/resources/architecture-diagram.png)

---

## ✨ Key Features

- **⚡ Zero-Shot Semantic Routing**: Fast intent classification using embedding-based sentence encoders (`sentence-transformers/all-MiniLM-L6-v2`) without incurring LLM latency or token overhead.
- **🔍 Natural Language to SQL (NL-to-SQL)**: Dynamically translates complex search queries (brands, price caps, discounts, ratings, review volume) into safe SQLite `SELECT` queries on a catalog of 900+ products.
- **📚 Semantic FAQ RAG**: Embeds customer care policies into ChromaDB to answer order tracking, returns, refunds, payment options, and international shipping questions with grounded truth.
- **💬 Engaging Small Talk**: Friendly conversational agent providing natural interaction and graceful fallback when queries fall outside predefined catalog intents.
- **🎨 Modern Clean UI**:
  - Hero onboarding card with interactive feature highlights.
  - Route classification badges (`Product Catalog`, `Store Knowledge Base`, `Conversational`).
  - Transparent SQL inspector drawer for debugging and auditing generated database queries.
  - Click-to-ask quick starter prompt chips.
  - Automatic link transformation converting raw URLs into clean `[🔗 View on Store]` buttons.
  - Catalog and knowledge base real-time status counters in the sidebar.

---

## 📂 Project Structure

```plaintext
Ecommerce_chatbot/
├── app/
│   ├── resources/
│   │   ├── architecture-diagram.png     # System architecture visual
│   │   ├── ecommerce_data_final.csv    # Raw product catalog dataset (Flipkart)
│   │   ├── faq_data.csv                # E-commerce store FAQs dataset
│   │   └── product-ss.png              # UI preview screenshot
│   ├── .env                            # API keys & model configuration
│   ├── .env.example                    # Sample environment variables
│   ├── db.sqlite                       # SQLite database (product table)
│   ├── faq.py                          # ChromaDB RAG vector ingestion & QA chain
│   ├── main.py                         # Streamlit clean UI application
│   ├── router.py                       # Semantic Router definitions & encoder
│   ├── smalltalk.py                    # Conversational chatbot chain
│   └── sql.py                          # NL-to-SQL generator & data comprehension
├── AGENTS.md                           # Multi-agent system documentation & session log
├── README.md                           # Main project documentation
└── requirements.txt                    # Python project dependencies
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Interactive web UI with custom modern CSS |
| **LLM Provider** | [Groq Cloud](https://groq.com/) | High-speed inference (`openai/gpt-oss-20b`) |
| **Semantic Router** | [semantic-router](https://github.com/aurelio-labs/semantic-router) | Aurelio AI semantic intent router |
| **Embeddings** | [Sentence Transformers](https://sbert.net/) | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | Embedding store for semantic FAQ search |
| **Relational DB** | [SQLite3](https://sqlite.org/) | 903 Flipkart footwear products |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) | Data manipulation and query result formatting |

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- A **Groq API Key** (get one free at [console.groq.com](https://console.groq.com)).

### 2. Clone the Repository
```bash
git clone <repository-url>
cd Ecommerce_chatbot
```

### 3. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create an `.env` file in the `app/` folder (or copy from `.env.example`):
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

> **Tip:** You can also use other Groq-supported models such as `llama-3.3-70b-versatile` or `mixtral-8x7b-32768`.

---

## 🖥️ Running the Application

Launch the Streamlit app from the project directory:

```bash
streamlit run app/main.py
```

The application will launch in your default browser at:
`http://localhost:8501`

---

## 💡 Example Queries to Try

### 🔍 Product Searches (SQL Agent)
- *"Show me running shoes under Rs. 3000"*
- *"Find top 3 Nike shoes with at least 30% discount"*
- *"What are the highest rated shoes for women?"*
- *"Show me Puma sneakers with rating above 4.2"*

### 📚 Store Policies & FAQs (RAG Agent)
- *"What is your return policy?"*
- *"Do I get a discount with my HDFC credit card?"*
- *"How can I track my order?"*
- *"What payment methods do you accept?"*
- *"How long does it take to process a refund?"*

### 💬 Casual Conversation (Small Talk Agent)
- *"Hello! Who are you?"*
- *"What can you help me with?"*
- *"Tell me a fun shoe fact"*

---

## ⚙️ How Each Module Works

### 1. Semantic Router (`app/router.py`)
Uses the local `all-MiniLM-L6-v2` dense embedding model to represent predefined route utterances for `faq`, `sql`, and `small-talk`. Incoming user queries are embedded into the same vector space, and cosine similarity matches the query to the correct route in milliseconds.

### 2. Product SQL Chain (`app/sql.py`)
1. **Query Generation**: Formulates an executable SQLite `SELECT` query given the table schema.
2. **Execution**: Safely executes `SELECT` statements on `db.sqlite`.
3. **Data Comprehension**: Translates raw rows into conversational product recommendations with title, price, discount %, rating, and direct product links.

### 3. FAQ Vector RAG (`app/faq.py`)
1. **Ingestion**: Ingests store FAQ questions and answers into a ChromaDB collection (`faqs`).
2. **Retrieval**: Retrieves top-2 relevant Q&A pairs for the query.
3. **Generation**: Groq LLM synthesizes an accurate, concise answer strictly grounded in the retrieved context.

### 4. Small Talk (`app/smalltalk.py`)
Handles conversational greetings, questions about the chatbot's identity, and general pleasantries.

---

## 🛡️ Rate Limits & Token Guardrails

To prevent Groq TPM (Tokens Per Minute) limit overruns (e.g. HTTP 413 on free tiers):
- **SQL Generation Guardrail**: System prompt mandates an explicit `LIMIT 5` (max 8) on all generated queries.
- **Data Capping**: The Python pipeline enforces `response.head(5)` before feeding rows into data comprehension, preventing large multi-thousand-token prompts.
- **Fail-Safe Fallback**: If an API rate limit or outage occurs, the system automatically builds recommendation lists directly from the database rows without crashing the application.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
