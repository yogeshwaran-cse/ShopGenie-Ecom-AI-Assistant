import os

from chromadb import Client
from chromadb.utils import embedding_functions
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

faqs_path = str(Path(__file__).parent / "resources/faq_data.csv")

load_dotenv(Path(__file__).parent / ".env")
load_dotenv()
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name= embedding_model
)

chroma_client = Client()
groq_client = Groq()
collection_name_faq = 'faqs'

def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting FAQ's into ChromaDB...")
        collection = chroma_client.create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )
        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f'id_{i}' for i in range(len(docs))]
        collection.add(
            ids=ids,
            metadatas=metadata,
            documents=docs
        )
        print("FAQ Data successfully ingested into chromaDB collection: ", collection_name_faq)
    else:
        print("FAQ Collection already exists.")     

def get_relevant_qa(query):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        ingest_faq_data(faqs_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name_faq,
        embedding_function= ef
    )
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    return results

def generate_answer(query, context):
    prompt = f'''Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {query}
    '''
    try:
        model_name = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b')
        kwargs = {
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'model': model_name,
            'max_tokens': 512,
        }
        if any(tag in model_name.lower() for tag in ('gpt-oss', 'qwen', 'deepseek')):
            kwargs['reasoning_effort'] = 'low'
        try:
            completion = groq_client.chat.completions.create(**kwargs)
        except Exception as param_err:
            if 'reasoning_effort' in kwargs and 'reasoning_effort' in str(param_err).lower():
                kwargs.pop('reasoning_effort', None)
                completion = groq_client.chat.completions.create(**kwargs)
            else:
                raise
        return (completion.choices[0].message.content or '').strip() or context
    except Exception as e:
        print(f"Error in Groq FAQ completion: {e}")
        return context

def faq_chain(query):
    result = get_relevant_qa(query)
    context = " ".join([r.get('answer') for r in result['metadatas'][0]])
    print("Context: ", context)
    answer = generate_answer(query,context)
    return(answer)
    
if __name__ == '__main__':
    ingest_faq_data(faqs_path)
    query = "Do you take cash as a payment option?"
    print(faq_chain(query))