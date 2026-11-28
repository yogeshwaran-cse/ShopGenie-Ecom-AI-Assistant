import os
import sys
from pathlib import Path

# Ensure local modules in the app directory resolve on Streamlit Community Cloud
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st

# Sync Streamlit Cloud secrets into os.environ if present
try:
    for secret_key in ["GROQ_API_KEY", "GROQ_MODEL"]:
        if secret_key in st.secrets and secret_key not in os.environ:
            os.environ[secret_key] = st.secrets[secret_key]
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv(current_dir / ".env")
load_dotenv()

import re
from faq import ingest_faq_data
from graph import run_graph

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ShopGenie | E-Commerce AI Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container and font adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.4rem;
        color: #e0e7ff;
    }

    /* Route tags */
    .route-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        margin-right: 0.3rem;
    }
    .badge-sql {
        background-color: #ede9fe;
        color: #5b21b6;
        border: 1px solid #c4b5fd;
    }
    .badge-faq {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-smalltalk {
        background-color: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
    }
    .badge-default {
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
    }

    /* Tier indicator banner */
    .tier-banner {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.5rem;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .tier-label {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.2rem 0.65rem;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .tier-fast {
        background: linear-gradient(135deg, #fef9c3, #fef08a);
        color: #854d0e;
        border: 1px solid #fde047;
    }
    .tier-llm {
        background: linear-gradient(135deg, #ede9fe, #ddd6fe);
        color: #5b21b6;
        border: 1px solid #c4b5fd;
    }
    .tier-fallback {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        border: 1px solid #fca5a5;
    }

    /* Welcome Card */
    .welcome-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.6rem;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .welcome-card h3 {
        margin-top: 0;
        color: #1e293b;
        font-weight: 700;
    }

    /* Feature pills in welcome screen */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .feature-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
    }
    .feature-box b {
        color: #1e293b;
    }
    .feature-box p {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0.3rem 0 0 0;
    }

    /* Sidebar styling */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e1b4b;
        margin-bottom: 0.3rem;
    }
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        color: #059669;
        font-weight: 600;
        background: #ecfdf5;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid #a7f3d0;
        margin-bottom: 1rem;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FAQ Ingestion Cache (Run Once)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_knowledge_base():
    faqs_path = Path(__file__).parent / "resources" / "faq_data.csv"
    ingest_faq_data(faqs_path)
    return True

initialize_knowledge_base()

# ---------------------------------------------------------
# Helper: Format Links & Clean Text
# ---------------------------------------------------------
def clean_and_format_text(text: str) -> str:
    """
    Transforms raw URLs or <url> into clean, clickable Markdown links.
    """
    if not isinstance(text, str):
        return text

    # Transform <http...> links
    text = re.sub(
        r'<(https?://[^\s>]+)>',
        r'[🔗 **View on Flipkart**](\1)',
        text
    )
    # Transform bare http... links that are not already in markdown [label](url)
    text = re.sub(
        r'(?<!\]\()(?<!\()(https?://[^\s\)]+)(?!\))',
        r'[🔗 **View on Flipkart**](\1)',
        text
    )
    return text


# ---------------------------------------------------------
# Routing display helpers
# ---------------------------------------------------------
ROUTE_BADGE_CLASSES = {
    'sql': 'badge-sql',
    'faq': 'badge-faq',
    'small-talk': 'badge-smalltalk',
}

ROUTE_LABELS = {
    'sql': '🔍 Product Catalog',
    'faq': '📚 Store Knowledge Base',
    'small-talk': '💬 Conversational',
}

TIER_CONFIG = {
    'semantic_fast_path': ('⚡ Semantic Fast-Path', 'tier-fast'),
    'llm_escalated':     ('🧠 LLM Escalated',      'tier-llm'),
    'semantic_fallback': ('🔄 Semantic Fallback',   'tier-fallback'),
}


def build_tier_badge_html(
    routing_source: str,
    confidence: float,
    routes: list[str],
) -> str:
    """Build the HTML for the tier indicator + route badges."""
    tier_label, tier_class = TIER_CONFIG.get(
        routing_source, ('❓ Unknown', 'tier-fallback')
    )

    # Confidence suffix for fast-path
    conf_str = f" ({confidence:.2f})" if routing_source == "semantic_fast_path" else ""

    parts = [
        f'<span class="tier-label {tier_class}">{tier_label}{conf_str}</span>',
        '<span style="color:#94a3b8;margin:0 0.15rem">•</span>',
    ]

    for route in routes:
        badge_cls = ROUTE_BADGE_CLASSES.get(route, 'badge-default')
        label = ROUTE_LABELS.get(route, route)
        parts.append(f'<span class="route-badge {badge_cls}">{label}</span>')

    return f'<div class="tier-banner">{"".join(parts)}</div>'


# ---------------------------------------------------------
# Core Query Handler with LangGraph Routing
# ---------------------------------------------------------
def ask(query: str, chat_history: list[dict] | None = None, active_filters: dict | None = None):
    """
    Intelligently routes query using the tiered LangGraph router.
    Returns dict with final_response, routes, routing_source, confidence_score,
    active_filters, sql_query.
    """
    try:
        result = run_graph(
            query,
            chat_history=chat_history,
            active_filters=active_filters,
        )
        result["final_response"] = clean_and_format_text(result.get("final_response", ""))
        return result
    except Exception as e:
        print(f"Error in ask(): {e}")
        return {
            "final_response": "I apologize, but I encountered an unexpected issue while processing your request. Please try again with a different query!",
            "routes": [],
            "routing_source": "semantic_fallback",
            "confidence_score": 0.0,
            "active_filters": active_filters or {},
            "sql_query": "",
        }

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if 'message' not in st.session_state:
    st.session_state.message = []

if 'show_debug' not in st.session_state:
    st.session_state.show_debug = True

if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">🛍️ ShopGenie AI</div>
        <div class="status-indicator">
            <span class="status-dot"></span> System Ready &amp; Active
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Starters")
    st.caption("Click any sample query to test:")

    sample_queries = [
        ("👟 Running shoes under ₹3000", "Show me running shoes under Rs. 3000"),
        ("⭐ Top 3 rated shoes", "Show top 3 shoes with highest rating"),
        ("🔥 Nike shoes with discounts", "Find Nike shoes that have discount"),
        ("🔄 Return & Refund policy", "What is the return policy of the products?"),
        ("💳 Payment methods", "What payment methods are accepted?"),
        ("👋 Who are you?", "What can you help me with?"),
        ("🔀 Multi-intent test", "Nike shoes under 3000 and what is your return policy?"),
    ]

    for label, query_text in sample_queries:
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            st.session_state.pending_query = query_text
            st.rerun()

    st.divider()

    st.markdown("### ⚙️ System Settings")
    st.session_state.show_debug = st.checkbox(
        "Show Router Details & SQL Query",
        value=st.session_state.show_debug,
        help="Displays the tiered routing path, confidence scores, and generated SQL queries."
    )

    st.markdown("### 📊 Knowledge & Catalog Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Catalog Products", value="903")
    with col2:
        st.metric(label="FAQ Articles", value="10")

    st.caption("• **Orchestrator**: LangGraph StateGraph\n• **Tier-1**: Cosine Similarity (MiniLM)\n• **Tier-2**: Groq LLM Multi-Intent\n• **Vector Store**: ChromaDB\n• **Database**: SQLite3")

    # Active filters display
    if st.session_state.active_filters:
        st.divider()
        st.markdown("### 🔍 Active Filters")
        for k, v in st.session_state.active_filters.items():
            if v is not None:
                st.caption(f"**{k}**: {v}")
        if st.button("🧹 Clear Filters", use_container_width=True):
            st.session_state.active_filters = {}
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.message = []
        st.session_state.active_filters = {}
        st.rerun()

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🛍️</span> ShopGenie E-Commerce Assistant
    </div>
    <div class="hero-subtitle">
        Intelligent product search, store policies, and shopping guidance powered by Semantic-First Tiered Routing &amp; LLMs.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Empty State / Welcome Screen
# ---------------------------------------------------------
if len(st.session_state.message) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Welcome to ShopGenie!</h3>
        <p style="color: #475569; font-size: 0.95rem;">
            I am your automated AI shopping assistant. Ask me anything about our footwear catalog, store policies, or recommendations!
        </p>
        <div class="feature-grid">
            <div class="feature-box">
                <b>🔍 Product Catalog Search</b>
                <p>Natural language queries converted to SQL for real-time filtering by brand, price, rating, and discounts.</p>
            </div>
            <div class="feature-box">
                <b>📚 Store Policies & FAQs</b>
                <p>Vector RAG system retrieving exact answers regarding returns, shipping, discounts, and payment methods.</p>
            </div>
            <div class="feature-box">
                <b>💬 Conversational Support</b>
                <p>Friendly small-talk and conversational assistance for a delightful shopping journey.</p>
            </div>
            <div class="feature-box">
                <b>🔀 Multi-Intent Routing</b>
                <p>Ask about products AND policies in one message — our tiered router handles both simultaneously.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")


def render_assistant_message(
    content: str,
    routes: list[str] | None = None,
    routing_source: str | None = None,
    confidence: float = 0.0,
    sql: str | None = None,
) -> None:
    """Render an assistant message with optional tier badge and SQL inspector."""
    if routes and routing_source and st.session_state.show_debug:
        badge_html = build_tier_badge_html(routing_source, confidence, routes)
        st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown(content)

    if sql and st.session_state.show_debug:
        with st.expander("🔍 View Generated SQL Query", expanded=False):
            st.code(sql, language="sql")


# ---------------------------------------------------------
# Chat History Display
# ---------------------------------------------------------
for msg in st.session_state.message:
    role = msg.get('role', 'user')
    content = msg.get('content', '')

    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🛍️"):
        if role == 'assistant':
            render_assistant_message(
                content,
                routes=msg.get('routes'),
                routing_source=msg.get('routing_source'),
                confidence=msg.get('confidence_score', 0.0),
                sql=msg.get('sql'),
            )
        else:
            st.markdown(content)

# ---------------------------------------------------------
# Chat Input & Query Execution
# ---------------------------------------------------------
user_input = st.chat_input(placeholder="Ask about shoes, discounts, return policies, or order help...")
pending = st.session_state.pop('pending_query', None)
active_query = pending if pending else user_input

if active_query:
    # 1. Build prior chat history (before appending current query)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.message
    ]

    # 2. Render user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(active_query)
    st.session_state.message.append({'role': 'user', 'content': active_query})

    # 3. Compute response via tiered router
    with st.chat_message("assistant", avatar="🛍️"):
        with st.spinner("Finding best answer..."):
            result = ask(
                active_query,
                chat_history=chat_history,
                active_filters=st.session_state.active_filters,
            )

        render_assistant_message(
            result["final_response"],
            routes=result.get("routes"),
            routing_source=result.get("routing_source"),
            confidence=result.get("confidence_score", 0.0),
            sql=result.get("sql_query"),
        )

    # 4. Persist active_filters for next turn
    st.session_state.active_filters = result.get("active_filters", {})

    # 5. Save assistant message to state
    st.session_state.message.append({
        'role': 'assistant',
        'content': result["final_response"],
        'routes': result.get("routes", []),
        'routing_source': result.get("routing_source", ""),
        'confidence_score': result.get("confidence_score", 0.0),
        'sql': result.get("sql_query"),
    })
