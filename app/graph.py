"""
ShopGenie LangGraph Routing — Semantic-First Tiered Router
==========================================================
Tier 1: Embedding cosine-similarity gate  (sub-50ms, zero API calls)
Tier 2: Groq LLM multi-intent escalation  (complex / ambiguous queries)
Fallback: Highest cosine label when Groq is unavailable

Supports multi-intent routing (e.g. sql + faq in one turn) and
conversational filter memory across chat turns.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

# ---------------------------------------------------------------------------
# Path & env bootstrap
# ---------------------------------------------------------------------------
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from dotenv import load_dotenv

load_dotenv(current_dir / ".env")
load_dotenv()

import numpy as np
from groq import Groq
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

from faq import faq_chain
from smalltalk import talk
from sql import sql_chain

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RouteName = Literal["sql", "faq", "small-talk"]
RoutingSource = Literal["semantic_fast_path", "llm_escalated", "semantic_fallback"]

VALID_ROUTES: tuple[RouteName, ...] = ("sql", "faq", "small-talk")

# ---------------------------------------------------------------------------
# Anchor utterances (shared by Tier-1 gate and Tier-2 fallback)
# ---------------------------------------------------------------------------
ANCHOR_UTTERANCES: dict[RouteName, list[str]] = {
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
        "cheap running shoes with good ratings",
        "compare Nike and Puma shoes",
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

# ---------------------------------------------------------------------------
# Tier-2 LLM router system prompt
# ---------------------------------------------------------------------------
LLM_ROUTER_SYSTEM_PROMPT = """\
You are an intent classifier for ShopGenie, an e-commerce footwear assistant.

Given the chat history, previous active_filters, and the current user message,
return a JSON object with exact keys:
{
  "routes": ["sql"],
  "active_filters": {
    "brand": null,
    "max_price": null,
    "min_rating": null,
    "category": null
  }
}

Rules:
- "routes": non-empty list using ONLY: "sql", "faq", "small-talk".
  - "sql": searching, filtering, comparing, or asking about shoes/products/prices/ratings/discounts.
  - "faq": store policies, returns, exchanges, refunds, shipping, tracking, payments, promo codes, damaged items.
  - "small-talk": greetings, farewells, thanks, jokes, assistant identity/capabilities.
  - Include MULTIPLE routes whenever the user asks about multiple domains in one message
    (e.g. searching for shoes AND asking about return policy -> ["sql", "faq"]).
- "active_filters":
  - For a follow-up refinement ("show cheaper ones", "only Puma", "above 4 rating", "under 2000"),
    carry forward Previous active_filters and update/add the changed filter.
    If the user asks for "cheaper ones" and max_price existed, lower max_price (e.g. by ~30%) or keep it.
  - For a brand-new product search that does not refer to prior results, extract only the constraints
    mentioned in the current message (set unmentioned keys to null).
  - If "sql" is not in "routes", copy Previous active_filters unchanged.
- Respond with raw JSON only."""

# ---------------------------------------------------------------------------
# Synthesizer system prompt (merges multi-intent results)
# ---------------------------------------------------------------------------
SYNTHESIZER_SYSTEM_PROMPT = """\
You are a helpful e-commerce assistant. You have been given partial answers
from different specialist agents. Merge them into a single cohesive reply.

Rules:
- Do NOT repeat greetings or introductions that already appear.
- Keep product lists exactly as provided (numbered, with links).
- Stitch FAQ answers naturally after the product section.
- Drop any "I don't know about <product query>" disclaimer from the FAQ answer when product results are already provided.
- Be concise and friendly."""

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_groq_client: Optional[Groq] = None
_embedding_model: Optional[SentenceTransformer] = None
_anchor_cache: Optional[dict[RouteName, np.ndarray]] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq()
    return _groq_client


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def _get_anchor_embeddings() -> dict[RouteName, np.ndarray]:
    """Pre-encode anchor utterances (cached for process lifetime)."""
    global _anchor_cache
    if _anchor_cache is None:
        model = _get_embedding_model()
        _anchor_cache = {
            route: model.encode(utterances, normalize_embeddings=True)
            for route, utterances in ANCHOR_UTTERANCES.items()
        }
    return _anchor_cache


# ---------------------------------------------------------------------------
# GraphState — the full state tracked across the workflow
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    chat_history: list[dict]
    routes: list[RouteName]
    routing_source: RoutingSource
    confidence_score: float
    active_filters: dict          # brand, max_price, min_rating, category
    sql_query: str                # generated SQL text (or empty)
    sql_result: str               # answer from SQL agent
    faq_result: str               # answer from FAQ agent
    final_response: str


# ---------------------------------------------------------------------------
# Tier-1: Semantic Gate Node
# ---------------------------------------------------------------------------
# Thresholds
FAST_PATH_MIN_SCORE = 0.78
FAST_PATH_MIN_GAP = 0.12


def _cosine_scores(query: str) -> dict[RouteName, float]:
    """Return max cosine similarity per route for *query*."""
    model = _get_embedding_model()
    anchors = _get_anchor_embeddings()
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores: dict[RouteName, float] = {}
    for route, emb in anchors.items():
        scores[route] = float(np.max(np.dot(emb, query_emb)))
    return scores


def semantic_gate_node(state: GraphState) -> dict:
    """Tier-1 fast-path classifier using cosine similarity."""
    scores = _cosine_scores(state["query"])
    sorted_routes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_route, top_score = sorted_routes[0]
    second_score = sorted_routes[1][1] if len(sorted_routes) > 1 else 0.0

    # Store raw scores on state for Tier-2 fallback use
    result: dict = {"_cosine_scores": scores}

    if top_score >= FAST_PATH_MIN_SCORE and (top_score - second_score) >= FAST_PATH_MIN_GAP:
        # High-confidence fast path — bypass LLM entirely
        result.update(
            {
                "routes": [top_route],
                "routing_source": "semantic_fast_path",
                "confidence_score": round(top_score, 4),
            }
        )
    else:
        # Mark for escalation (actual routing done in llm_router_node)
        result.update(
            {
                "routes": [],  # empty → signals "needs LLM"
                "routing_source": "semantic_fast_path",  # will be overwritten
                "confidence_score": round(top_score, 4),
            }
        )
    return result


def _needs_llm_escalation(state: GraphState) -> str:
    """Conditional edge: 'fast_path' if Tier-1 resolved, else 'escalate'."""
    if state.get("routes"):
        return "fast_path"
    return "escalate"


# ---------------------------------------------------------------------------
# Tier-2: LLM Multi-Intent Router Node
# ---------------------------------------------------------------------------

def _parse_llm_router_json(raw: str) -> Optional[dict]:
    """Robustly extract JSON from LLM output (handles markdown fences)."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find first { ... }
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _clean_filters(raw_filters: dict) -> dict:
    """Keep only known filter keys with non-null values."""
    cleaned: dict = {}
    for key in ("brand", "max_price", "min_rating", "category"):
        val = raw_filters.get(key)
        if val is not None and val != "":
            cleaned[key] = val
    return cleaned


def _compact_history_text(content: str, max_chars: int = 240) -> str:
    """Strip long URLs and cap turn length to avoid Groq TPM/reasoning bloat."""
    no_urls = re.sub(r"https?://\S+", "[link]", content)
    collapsed = re.sub(r"\s+", " ", no_urls).strip()
    if len(collapsed) > max_chars:
        return collapsed[: max_chars - 3] + "..."
    return collapsed


def _groq_chat_completion(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
):
    """Call Groq chat completions with reasoning-model awareness and safe param fallback."""
    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if any(tag in model_name.lower() for tag in ("gpt-oss", "qwen", "deepseek")):
        kwargs["reasoning_effort"] = "low"

    try:
        return _get_groq_client().chat.completions.create(**kwargs)
    except Exception as first_err:
        # Retry without optional model-specific params if the backend rejects them
        if "reasoning_effort" in kwargs or "response_format" in kwargs:
            err_msg = str(first_err).lower()
            if any(k in err_msg for k in ("reasoning_effort", "response_format", "json_object", "invalid_request")):
                kwargs.pop("reasoning_effort", None)
                kwargs.pop("response_format", None)
                return _get_groq_client().chat.completions.create(**kwargs)
        raise


def llm_router_node(state: GraphState) -> dict:
    """Tier-2 LLM-based multi-intent classifier with filter extraction."""
    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

    # Build condensed chat history context (last 4 prior turns, URLs stripped)
    history_lines: list[str] = []
    for turn in (state.get("chat_history") or [])[-4:]:
        role = turn.get("role", "user")
        snippet = _compact_history_text(str(turn.get("content", "")))
        if snippet:
            history_lines.append(f"{role}: {snippet}")
    history_ctx = "\n".join(history_lines) if history_lines else "(no prior turns)"

    prev_filters = _clean_filters(state.get("active_filters") or {})
    prev_filters_str = json.dumps(prev_filters)

    user_prompt = (
        f"Chat history:\n{history_ctx}\n\n"
        f"Previous active_filters: {prev_filters_str}\n\n"
        f"Current user message: {state['query']}"
    )

    try:
        completion = _groq_chat_completion(
            model_name,
            [
                {"role": "system", "content": LLM_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=1024,
            json_mode=True,
        )
        raw_content = (completion.choices[0].message.content or "").strip()
        parsed = _parse_llm_router_json(raw_content)

        if parsed and "routes" in parsed:
            routes_raw = parsed["routes"]
            routes = list(dict.fromkeys(r for r in routes_raw if r in VALID_ROUTES))
            if not routes:
                routes = ["small-talk"]

            new_filters_raw = parsed.get("active_filters")
            if "sql" not in routes:
                merged_filters = prev_filters
            elif isinstance(new_filters_raw, dict):
                merged_filters = _clean_filters(new_filters_raw)
            else:
                merged_filters = prev_filters

            return {
                "routes": routes,
                "routing_source": "llm_escalated",
                "active_filters": merged_filters,
            }

        print(
            f"LLM router returned unparseable content "
            f"(finish_reason={completion.choices[0].finish_reason}, content={raw_content!r}), "
            f"falling back to cosine label."
        )

    except Exception as e:
        print(f"LLM router failed ({e}), falling back to cosine label.")

    # ----- Semantic fallback -----
    scores: dict[RouteName, float] = state.get("_cosine_scores") or _cosine_scores(state["query"])
    best_route = max(scores, key=scores.get)  # type: ignore[arg-type]
    return {
        "routes": [best_route] if scores[best_route] >= 0.3 else ["small-talk"],
        "routing_source": "semantic_fallback",
        "confidence_score": round(scores[best_route], 4),
    }


# ---------------------------------------------------------------------------
# Dispatch Node — fans out to individual agents
# ---------------------------------------------------------------------------

def dispatch_node(state: GraphState) -> dict:
    """Execute each active route and collect results."""
    routes: list[RouteName] = state.get("routes") or ["small-talk"]
    query = state["query"]

    # Inject active filters into the SQL query when present
    effective_query = query
    filters = state.get("active_filters") or {}
    filter_parts: list[str] = []
    if filters.get("brand"):
        filter_parts.append(f"brand: {filters['brand']}")
    if filters.get("max_price"):
        filter_parts.append(f"max price: {filters['max_price']}")
    if filters.get("min_rating"):
        filter_parts.append(f"minimum rating: {filters['min_rating']}")
    if filters.get("category"):
        filter_parts.append(f"category: {filters['category']}")
    if filter_parts and "sql" in routes:
        effective_query = f"{query} (filters: {', '.join(filter_parts)})"

    results: dict = {}

    for route in routes:
        if route == "sql":
            answer, generated_sql = sql_chain(effective_query, return_sql=True)
            results["sql_result"] = answer or ""
            results["sql_query"] = generated_sql or ""
        elif route == "faq":
            results["faq_result"] = faq_chain(query) or ""
        elif route == "small-talk":
            # For small-talk in a multi-intent, still produce a result
            results["faq_result"] = results.get("faq_result", "")
            results["sql_result"] = results.get("sql_result", "")
            results["final_response"] = talk(query)

    return results


# ---------------------------------------------------------------------------
# Synthesizer Node — merges multi-route results
# ---------------------------------------------------------------------------

def _needs_synthesis(state: GraphState) -> str:
    """Conditional edge: synthesize when >1 route produced results."""
    routes = state.get("routes") or []
    if len(routes) > 1:
        return "synthesize"
    return "done"


def _compress_urls(text: str) -> tuple[str, dict[str, str]]:
    """Replace long URLs with short placeholders to save Groq TPM tokens."""
    url_map: dict[str, str] = {}
    rev_map: dict[str, str] = {}

    def _repl(match: re.Match) -> str:
        url = match.group(0)
        if url not in rev_map:
            token = f"URL_{len(url_map)}"
            rev_map[url] = token
            url_map[token] = url
        return rev_map[url]

    compressed = re.sub(r"https?://[^\s>)]+", _repl, text)
    return compressed, url_map


def synthesizer_node(state: GraphState) -> dict:
    """Merge multi-route results into a single coherent final_response."""
    sql_result = state.get("sql_result", "")
    faq_result = state.get("faq_result", "")
    talk_result = state.get("final_response", "")

    # Collect non-empty parts
    parts = [p for p in (talk_result, sql_result, faq_result) if p]
    if len(parts) <= 1:
        return {"final_response": parts[0] if parts else ""}

    # Compress long Flipkart URLs before sending to LLM to save TPM tokens
    compact_sql, url_map = _compress_urls(sql_result)

    sections: list[str] = []
    if talk_result:
        sections.append(f"Conversational reply:\n{talk_result}")
    if compact_sql:
        sections.append(f"Product search results:\n{compact_sql}")
    if faq_result:
        sections.append(f"FAQ / policy answer:\n{faq_result}")

    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    merge_prompt = "\n\n".join(sections)

    try:
        completion = _groq_chat_completion(
            model_name,
            [
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": merge_prompt},
            ],
            temperature=0.3,
            max_tokens=1536,
        )
        merged = (completion.choices[0].message.content or "").strip()
        if merged and all(tok in merged for tok in url_map):
            for tok, orig_url in url_map.items():
                merged = merged.replace(tok, orig_url)
            return {"final_response": merged}
    except Exception as e:
        print(f"Synthesizer LLM failed ({e}), concatenating results.")

    return {"final_response": "\n\n---\n\n".join(parts)}


def finalize_node(state: GraphState) -> dict:
    """Pick whichever result field is populated as the final_response."""
    if state.get("final_response"):
        return {}
    # Single route — promote the relevant result
    routes = state.get("routes") or []
    if "sql" in routes and state.get("sql_result"):
        return {"final_response": state["sql_result"]}
    if "faq" in routes and state.get("faq_result"):
        return {"final_response": state["faq_result"]}
    return {"final_response": state.get("sql_result") or state.get("faq_result") or ""}


# ---------------------------------------------------------------------------
# Build the StateGraph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    # --- Nodes ---
    builder.add_node("semantic_gate", semantic_gate_node)
    builder.add_node("llm_router", llm_router_node)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("finalize", finalize_node)

    # --- Edges ---
    builder.add_edge(START, "semantic_gate")

    # Tier-1 → fast_path (dispatch) or escalate (llm_router)
    builder.add_conditional_edges(
        "semantic_gate",
        _needs_llm_escalation,
        {
            "fast_path": "dispatch",
            "escalate": "llm_router",
        },
    )

    # Tier-2 → dispatch
    builder.add_edge("llm_router", "dispatch")

    # dispatch → synthesize (multi-route) or finalize (single-route)
    builder.add_conditional_edges(
        "dispatch",
        _needs_synthesis,
        {
            "synthesize": "synthesizer",
            "done": "finalize",
        },
    )

    builder.add_edge("synthesizer", END)
    builder.add_edge("finalize", END)

    return builder.compile()


graph = build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_graph(
    query: str,
    chat_history: Optional[list[dict]] = None,
    active_filters: Optional[dict] = None,
) -> dict:
    """Invoke the LangGraph tiered-router workflow.

    Returns a dict with keys:
        final_response, routes, routing_source, confidence_score,
        active_filters, sql_query
    """
    result = graph.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
            "routes": [],
            "routing_source": "semantic_fast_path",
            "confidence_score": 0.0,
            "active_filters": active_filters or {},
            "sql_query": "",
            "sql_result": "",
            "faq_result": "",
            "final_response": "",
        }
    )
    return {
        "final_response": result.get("final_response", ""),
        "routes": result.get("routes", []),
        "routing_source": result.get("routing_source", "semantic_fast_path"),
        "confidence_score": result.get("confidence_score", 0.0),
        "active_filters": result.get("active_filters", {}),
        "sql_query": result.get("sql_query", ""),
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What is your policy on defective product?",
        "Pink Puma shoes in price range 1000 to 5000",
        "How are you?",
        "Nike shoes under 3000 and what is your return policy?",
        "Show cheaper ones",  # filter memory test
    ]
    filters: dict = {}
    history: list[dict] = []
    for q in test_queries:
        result = run_graph(q, chat_history=history, active_filters=filters)
        print(f"Query  : {q}")
        print(f"Routes : {result['routes']}")
        print(f"Source : {result['routing_source']} (conf={result['confidence_score']:.2f})")
        print(f"Filters: {result['active_filters']}")
        print(f"SQL    : {result['sql_query'][:120]}...")
        print(f"Answer : {result['final_response'][:200]}...")
        print("-" * 60)
        # Accumulate history & filters for next turn
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["final_response"]})
        filters = result["active_filters"]
