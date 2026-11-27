import os
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

# Ensure local modules in the app directory resolve regardless of cwd
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from dotenv import load_dotenv

load_dotenv(current_dir / ".env")
load_dotenv()

from groq import Groq
from langgraph.graph import END, START, StateGraph

from faq import faq_chain
from smalltalk import talk
from sql import sql_chain

RouteName = Literal["sql", "faq", "small-talk"]

VALID_ROUTES: tuple[RouteName, ...] = ("sql", "faq", "small-talk")

ROUTER_SYSTEM_PROMPT = """You are an intent classification router for ShopGenie, an e-commerce footwear assistant.
Classify the user's message into EXACTLY ONE of these three routes:
- sql: Searching, browsing, filtering, comparing, or asking for recommendations/prices/ratings/discounts on products or shoes in the catalog (e.g., "Nike shoes under 3000", "top rated running shoes", "Campus sneakers on sale").
- faq: Store policies, customer support, returns, refunds, exchanges, order tracking, shipping, delivery, payment methods, credit card discounts, promo codes, or damaged/defective items.
- small-talk: Greetings, farewells, thank-yous, casual conversation, jokes, or questions about who you are and what you can do.

Respond with ONLY the route label: sql, faq, or small-talk."""

# Fallback utterances used only if the Groq router call fails
FALLBACK_UTTERANCES: dict[RouteName, list[str]] = {
    "faq": [
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
        "Can I cancel or modify my order after placing it?",
        "Do you offer international shipping?",
        "What should I do if I receive a damaged product?",
        "How do I use a promo code during checkout?",
        "Can I exchange an item?",
    ],
    "sql": [
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Are there any Puma shoes on sale?",
        "Show me adidas shoes",
        "Top rated shoes for women",
        "Sneakers under 2000",
        "Find sports shoes with rating above 4",
        "Show me Campus shoes",
    ],
    "small-talk": [
        "How are you?",
        "What is your name?",
        "What can you help me with?",
        "Hello",
        "Hi",
        "Good morning",
        "Tell me a joke",
        "Thank you",
        "Goodbye",
    ],
}

_groq_client: Optional[Groq] = None
_fallback_cache: Optional[tuple[object, list[tuple[RouteName, object]]]] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq()
    return _groq_client


def _parse_route_label(raw_label: str) -> Optional[RouteName]:
    cleaned = raw_label.strip().lower().strip("`\"'.,:;!")
    if cleaned in VALID_ROUTES:
        return cleaned  # type: ignore[return-value]
    for route in VALID_ROUTES:
        if route in cleaned:
            return route
    return None


def _fallback_classify(query: str) -> RouteName:
    """Embedding-similarity fallback if LLM routing fails, with keyword safety net."""
    global _fallback_cache
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if _fallback_cache is None:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            encoded = [
                (route, model.encode(utterances, normalize_embeddings=True))
                for route, utterances in FALLBACK_UTTERANCES.items()
            ]
            _fallback_cache = (model, encoded)

        model, encoded = _fallback_cache
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        best_route: RouteName = "small-talk"
        best_score = -1.0
        for route, emb in encoded:
            max_score = float(np.max(np.dot(emb, query_emb)))
            if max_score > best_score:
                best_score = max_score
                best_route = route
        return best_route if best_score >= 0.3 else "small-talk"
    except Exception:
        lowered = query.lower()
        if any(k in lowered for k in ("return", "refund", "track", "payment", "shipping", "policy", "promo", "defective", "damaged", "exchange", "cancel")):
            return "faq"
        if any(k in lowered for k in ("shoe", "shoes", "sneaker", "nike", "puma", "adidas", "campus", "sparx", "price", "rating", "discount", "under", "cheap", "buy")):
            return "sql"
        return "small-talk"


def classify_route(query: str) -> RouteName:
    """Classify a user query into 'sql', 'faq', or 'small-talk'."""
    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    try:
        completion = _get_groq_client().chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=16,
        )
        raw_content = completion.choices[0].message.content or ""
        parsed = _parse_route_label(raw_content)
        if parsed is not None:
            return parsed
    except Exception as e:
        print(f"LangGraph Groq router failed ({e}), using fallback classifier.")

    return _fallback_classify(query)


class GraphState(TypedDict):
    query: str
    route: RouteName
    answer: str
    generated_sql: Optional[str]


def router_node(state: GraphState) -> dict:
    route = classify_route(state["query"])
    return {"route": route}


def route_decision(state: GraphState) -> RouteName:
    return state["route"]


def sql_node(state: GraphState) -> dict:
    answer, generated_sql = sql_chain(state["query"], return_sql=True)
    return {"answer": answer, "generated_sql": generated_sql}


def faq_node(state: GraphState) -> dict:
    answer = faq_chain(state["query"])
    return {"answer": answer, "generated_sql": None}


def smalltalk_node(state: GraphState) -> dict:
    answer = talk(state["query"])
    return {"answer": answer, "generated_sql": None}


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("router", router_node)
    builder.add_node("sql_agent", sql_node)
    builder.add_node("faq_agent", faq_node)
    builder.add_node("smalltalk_agent", smalltalk_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "sql": "sql_agent",
            "faq": "faq_agent",
            "small-talk": "smalltalk_agent",
        },
    )
    builder.add_edge("sql_agent", END)
    builder.add_edge("faq_agent", END)
    builder.add_edge("smalltalk_agent", END)

    return builder.compile()


graph = build_graph()


def run_graph(query: str) -> tuple[str, RouteName, Optional[str]]:
    """Invoke the LangGraph workflow and return (answer, route_name, generated_sql)."""
    result = graph.invoke(
        {
            "query": query,
            "route": "small-talk",
            "answer": "",
            "generated_sql": None,
        }
    )
    return result["answer"], result["route"], result.get("generated_sql")


if __name__ == "__main__":
    test_queries = [
        "What is your policy on defective product?",
        "Pink Puma shoes in price range 1000 to 5000",
        "How are you?",
    ]
    for q in test_queries:
        ans, route, sql = run_graph(q)
        print(f"Query: {q}\nRoute: {route}\nSQL: {sql}\nAnswer: {ans}\n{'-' * 50}")
