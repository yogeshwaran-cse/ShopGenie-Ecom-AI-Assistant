import numpy as np

# Multi-version compatibility with semantic-router (0.0.x, 0.3.x, 1.x)
Route = None
RouteLayer = None
HuggingFaceEncoder = None

try:
    # 0.0.x syntax
    from semantic_router import Route, RouteLayer
    from semantic_router.encoders import HuggingFaceEncoder
except ImportError:
    try:
        # 0.3.x syntax
        from semantic_router import Route
        from semantic_router.layer import RouteLayer
        from semantic_router.encoders import HuggingFaceEncoder
    except ImportError:
        try:
            # 1.x syntax
            from semantic_router import Route, SemanticRouter as RouteLayer
            from semantic_router.encoders import HuggingFaceEncoder
        except ImportError:
            Route = None
            RouteLayer = None
            HuggingFaceEncoder = None


# -------------------------------------------------------------------
# Self-healing Fallback Router
# Ensures 100% uptime on cloud hosts (e.g., Streamlit Community Cloud)
# without crashing on environment or package import discrepancies.
# -------------------------------------------------------------------
class FallbackRouteMatch:
    def __init__(self, name):
        self.name = name

class FallbackRoute:
    def __init__(self, name, utterances):
        self.name = name
        self.utterances = utterances

class FallbackRouter:
    def __init__(self, routes, model_name='sentence-transformers/all-MiniLM-L6-v2', threshold=0.3):
        from sentence_transformers import SentenceTransformer
        self.routes = routes
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)
        self.route_data = []
        for r in routes:
            emb = self.model.encode(r.utterances, normalize_embeddings=True)
            self.route_data.append((r.name, emb))

    def __call__(self, text):
        query_emb = self.model.encode([text], normalize_embeddings=True)[0]
        best_route = None
        best_score = -1.0
        for name, emb in self.route_data:
            scores = np.dot(emb, query_emb)
            max_score = float(np.max(scores))
            if max_score > best_score:
                best_score = max_score
                best_route = name

        if best_score >= self.threshold:
            return FallbackRouteMatch(best_route)
        return FallbackRouteMatch(None)


RouteClass = Route if Route is not None else FallbackRoute

# -------------------------------------------------------------------
# Route Utterance Definitions
# -------------------------------------------------------------------
faq = RouteClass(
    name='faq',
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
        "Are there any ongoing sales or promotions?",
        "Can I cancel or modify my order after placing it?",
        "Do you offer international shipping?",
        "What should I do if I receive a damaged product?",
        "How do I use a promo code during checkout?",
        "Can I exchange an item?",
        "How to contact customer care?",
        "Where is my package?",
        "What is your refund policy?",
    ]
)

sql = RouteClass(
    name='sql',
    utterances=[
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?",
        "Show me adidas shoes",
        "Top rated shoes for women",
        "Shoes with highest rating",
        "Show me cheap walking shoes",
        "Sneakers under 2000",
        "Show me running shoes with discount",
        "Find sports shoes with rating above 4",
        "Show me Campus shoes",
    ]
)

small_talk = RouteClass(
    name='small-talk',
    utterances=[
        "How are you?",
        "What is your name?",
        "Are you a robot?",
        "What are you?",
        "What do you do?",
        "Hello",
        "Hi",
        "Hey",
        "Hey there",
        "Good morning",
        "Good evening",
        "Tell me a joke",
        "Who made you?",
        "Thank you",
        "Thanks",
        "Bye",
        "Goodbye",
    ]
)

# -------------------------------------------------------------------
# Router Initialization
# -------------------------------------------------------------------
if RouteLayer is not None and HuggingFaceEncoder is not None:
    try:
        encoder = HuggingFaceEncoder(name='sentence-transformers/all-MiniLM-L6-v2')
        router = RouteLayer(routes=[faq, sql, small_talk], encoder=encoder)
    except Exception as e:
        print(f"Native semantic-router init failed ({e}), using FallbackRouter.")
        router = FallbackRouter(routes=[faq, sql, small_talk])
else:
    router = FallbackRouter(routes=[faq, sql, small_talk])


if __name__ == "__main__":
    print("Defective product route:", router("What is your policy on defective product?").name)
    print("Puma shoes route:", router("Pink Puma shoes in price range 5000 to 1000").name)
    print("Greeting route:", router("How are you?").name)
