import os
from groq import Groq

groq_client = Groq()

def talk(query):
    prompt = f'''You are a helpful and friendly chatbot designed for small talk. You can answer questions about the weather, your name, your purpose, and more.
    
    QUESTION: {query}
    '''
    try:
        completion = groq_client.chat.completions.create(
            model=os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b'),
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in smalltalk: {e}")
        return "Hello! I am your ShopGenie assistant. How can I assist you with footwear, orders, or shopping inquiries today?"