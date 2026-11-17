from groq import Groq
import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()

GROQ_MODEL = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b')

db_path = str(Path(__file__).parent / "db.sqlite")

client_sql = Groq()

sql_prompt = """You are an expert in understanding the database schema and generating SQL queries for a natural language question asked
pertaining to the data you have. The schema is provided in the schema tags. 
<schema> 
table: product 

fields: 
product_link - string (hyperlink to product)	
title - string (name of the product)	
brand - string (brand of the product)	
price - integer (price of the product in Indian Rupees)	
discount - float (discount on the product. 10 percent discount is represented as 0.1, 20 percent as 0.2, and such.)	
avg_rating - float (average rating of the product. Range 0-5, 5 is the highest.)	
total_ratings - integer (total number of ratings for the product)

</schema>
Make sure whenever you try to search for the brand name, the name can be in any case. 
So, make sure to use %LIKE% to find the brand in condition. Never use "ILIKE". 
Create a single SQL query for the question provided. 
The query should have all the fields in SELECT clause (i.e. SELECT *)

IMPORTANT: ALWAYS include a LIMIT clause in your SQL query. If the user does not specify a count, default to LIMIT 5 (maximum LIMIT 8). Never return an unbounded query without a LIMIT clause.

Just the SQL query is needed, nothing more. Always provide the SQL in between the <SQL></SQL> tags."""


comprehension_prompt = """You are an expert in understanding the context of the question and replying based on the data pertaining to the question provided. You will be provided with Question: and Data:. The data will be in the form of an array or a dataframe or dict. Reply based on only the data provided as Data for answering the question asked as Question. Do not write anything like 'Based on the data' or any other technical words. Just a plain simple natural language response.
The Data would always be in context to the question asked. For example is the question is “What is the average rating?” and data is “4.3”, then answer should be “The average rating for the product is 4.3”. So make sure the response is curated with the question and data. Make sure to note the column names to have some context, if needed, for your response.
There can also be cases where you are given an entire dataframe in the Data: field. Always remember that the data field contains the answer of the question asked. All you need to do is to always reply in the following format when asked about a product: 
Produt title, price in indian rupees, discount, and rating, and then product link. Take care that all the products are listed in list format, one line after the other. Not as a paragraph.
For example:
1. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
2. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
3. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>

"""


def generate_sql_query(question):
    try:
        chat_completion = client_sql.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": sql_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return ""



def run_query(query):
    if query.strip().upper().startswith('SELECT'):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query,conn)
            return df


def data_comprehension(question, context):
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": comprehension_prompt,
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}. DATA: {context}",
            }
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        # max_tokens=1024
    )

    return chat_completion.choices[0].message.content



def sql_chain(question, return_sql=False):
    sql_query = generate_sql_query(question)
    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, sql_query, re.DOTALL)

    if len(matches) == 0:
        msg = "Sorry, I was unable to generate a search query for your request. Please try rephrasing your product inquiry."
        return (msg, None) if return_sql else msg

    raw_sql = matches[0].strip()
    print("Generated SQL:", raw_sql)

    response = run_query(raw_sql)
    if response is None:
        msg = "Sorry, there was an issue querying our product database. Please try again."
        return (msg, raw_sql) if return_sql else msg

    if response.empty:
        msg = "I couldn't find any products in our catalog matching your search criteria. Try broadening your keywords, brand names, or price ranges!"
        return (msg, raw_sql) if return_sql else msg

    # Hard cap records sent to LLM to prevent Groq TPM / 413 token limit errors
    max_records = 5
    if len(response) > max_records:
        response = response.head(max_records)

    context = response.to_dict(orient='records')
    try:
        answer = data_comprehension(question, context)
    except Exception as e:
        print(f"Fallback due to data_comprehension error: {e}")
        # Build clean response directly from data records without requiring LLM
        lines = ["Here are the top products matching your query:"]
        for idx, row in enumerate(context, 1):
            title = row.get('title', 'Footwear')
            price = row.get('price', 'N/A')
            discount_val = row.get('discount', 0)
            try:
                discount_pct = int(round(float(discount_val) * 100))
            except (ValueError, TypeError):
                discount_pct = 0
            rating = row.get('avg_rating', 'N/A')
            link = row.get('product_link', '')
            lines.append(f"{idx}. {title}: Rs. {price} ({discount_pct} percent off), Rating: {rating} <{link}>")
        answer = "\n".join(lines)

    return (answer, raw_sql) if return_sql else answer


if __name__ == "__main__":
    # question = "All shoes with rating higher than 4.5 and total number of reviews greater than 500"
    # sql_query = generate_sql_query(question)
    # print(sql_query)
    question = "Show top 3 shoes in descending order of rating"
    # question = "Show me 3 running shoes for woman"
    # question = "sfsdfsddsfsf"
    answer = sql_chain(question)
    print(answer)
