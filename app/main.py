import os
import sys
from pathlib import Path

# Ensure local modules in the app directory resolve on Streamlit Community Cloud
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st

# Sync Streamlit Cloud secrets into os.environ if present
for secret_key in ["GROQ_API_KEY", "GROQ_MODEL"]:
    if hasattr(st, "secrets") and secret_key in st.secrets:
        if secret_key not in os.environ:
            os.environ[secret_key] = st.secrets[secret_key]

import re
import time
from faq import ingest_faq_data, faq_chain
from sql import sql_chain
from smalltalk import talk
from router import router

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

    /* Clean link formatting inside messages */
    a.product-link-btn {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: #4f46e5;
        color: white !important;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.82rem;
        margin-left: 0.4rem;
        transition: background-color 0.2s ease;
    }
    a.product-link-btn:hover {
        background-color: #4338ca;
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
# Core Query Handler with Semantic Routing
# ---------------------------------------------------------
def ask(query: str):
    """
    Intelligently routes query using semantic routing to FAQ, SQL, or Small-talk.
    Returns (answer_text, route_name, generated_sql).
    """
    try:
        route_obj = router(query)
        route_name = route_obj.name if route_obj else None
        generated_sql = None

        if route_name == 'faq':
            answer = faq_chain(query)
        elif route_name == 'sql':
            answer, generated_sql = sql_chain(query, return_sql=True)
        elif route_name == 'small-talk':
            answer = talk(query)
        else:
            # Graceful fallback: If router similarity was below cutoff, try general assistant
            route_name = 'small-talk'
            answer = talk(query)

        clean_answer = clean_and_format_text(answer)
        return clean_answer, route_name, generated_sql
    except Exception as e:
        print(f"Error in ask(): {e}")
        return "I apologize, but I encountered an unexpected issue while processing your request. Please try again with a different query!", "error", None

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if 'message' not in st.session_state:
    st.session_state.message = []

if 'show_debug' not in st.session_state:
    st.session_state.show_debug = True

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">🛍️ ShopGenie AI</div>
        <div class="status-indicator">
            <span class="status-dot"></span> System Ready & Active
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
        help="Displays the identified semantic route and generated SQL queries."
    )

    st.markdown("### 📊 Knowledge & Catalog Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Catalog Products", value="903")
    with col2:
        st.metric(label="FAQ Articles", value="10")

    st.caption("• **Vector Store**: ChromaDB (all-MiniLM-L6-v2)\n• **Database**: SQLite3\n• **LLM Engine**: Groq API")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.message = []
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
        Intelligent product search, store policies, and shopping guidance powered by Semantic Routing & LLMs.
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
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

# ---------------------------------------------------------
# Chat History Display
# ---------------------------------------------------------
for msg in st.session_state.message:
    role = msg.get('role', 'user')
    content = msg.get('content', '')
    route = msg.get('route')
    sql = msg.get('sql')

    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🛍️"):
        if role == 'assistant' and route and st.session_state.show_debug:
            badge_class = {
                'sql': 'badge-sql',
                'faq': 'badge-faq',
                'small-talk': 'badge-smalltalk'
            }.get(route, 'badge-default')

            route_labels = {
                'sql': '🔍 Product Catalog (SQL Query)',
                'faq': '📚 Store Knowledge Base (FAQ RAG)',
                'small-talk': '💬 Conversational Assistant'
            }
            display_route = route_labels.get(route, f"Route: {route}")

            st.markdown(
                f'<span class="route-badge {badge_class}">{display_route}</span>',
                unsafe_allow_html=True
            )

        st.markdown(content)

        if role == 'assistant' and sql and st.session_state.show_debug:
            with st.expander("🔍 View Generated SQL Query", expanded=False):
                st.code(sql, language="sql")

# ---------------------------------------------------------
# Chat Input & Query Execution
# ---------------------------------------------------------
user_input = st.chat_input(placeholder="Ask about shoes, discounts, return policies, or order help...")
pending = st.session_state.pop('pending_query', None)
active_query = pending if pending else user_input

if active_query:
    # 1. Render user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(active_query)
    st.session_state.message.append({'role': 'user', 'content': active_query})

    # 2. Compute response
    with st.chat_message("assistant", avatar="🛍️"):
        with st.spinner("Finding best answer..."):
            start_time = time.time()
            response_text, route_name, sql_query = ask(active_query)
            elapsed = time.time() - start_time

        # Render route badge
        if route_name and st.session_state.show_debug:
            badge_class = {
                'sql': 'badge-sql',
                'faq': 'badge-faq',
                'small-talk': 'badge-smalltalk'
            }.get(route_name, 'badge-default')

            route_labels = {
                'sql': '🔍 Product Catalog (SQL Query)',
                'faq': '📚 Store Knowledge Base (FAQ RAG)',
                'small-talk': '💬 Conversational Assistant'
            }
            display_route = route_labels.get(route_name, f"Route: {route_name}")

            st.markdown(
                f'<span class="route-badge {badge_class}">{display_route}</span>',
                unsafe_allow_html=True
            )

        st.markdown(response_text)

        if sql_query and st.session_state.show_debug:
            with st.expander("🔍 View Generated SQL Query", expanded=False):
                st.code(sql_query, language="sql")

    # 3. Save assistant message to state
    st.session_state.message.append({
        'role': 'assistant',
        'content': response_text,
        'route': route_name,
        'sql': sql_query
    })
