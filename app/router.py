from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    name = 'sentence-transformers/all-MiniLM-L6-v2'
)

faq = Route(
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

sql = Route(
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

small_talk = Route(
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

router = RouteLayer(routes=[faq, sql, small_talk], encoder=encoder)

if __name__ == "__main__":
    print(router("What is your policy on defective product?").name)
    print(router("Pink Puma shoes in price range 5000 to 1000").name)
    print(router("How are you?").name)