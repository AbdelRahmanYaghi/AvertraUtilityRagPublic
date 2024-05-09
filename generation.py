from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import instructor # Doesn't include "anthropic" but has to be installed
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class check_relevance_output(BaseModel):
    relevant: int

class chat_response_output(BaseModel):
    text: str

client_instructor = instructor.from_openai(OpenAI())


SYS_PROMPTS = {
    'chat':'''
        You are an assistant, and a data extractor for an Utilities Company.
        You can read some relevant texts related to the question which will be given along with the question.
        Hence you are required to answer accordingly. Do not relate from something outside the knowledge base provided.   
    ''',
    'check_relevance': '''
        You will be given a certain text from the user.
        You should determine whether the text is related to a question or an inquery about Energy utilities.
        Examples of Energy utilities include, but are not limited to: electricity, energy, natural gas, water, waste, communication, smart cities, transport, etc...
        return 1 if the text is related to Energy Utilities, and return 0 otherwise. Return your output in the following format:

        {
            "relevant": 1 | 0
        }

        And only return the JSON. Do not return anything else.
        '''
}

def check_relevance(prompt):
    '''
    Determines whether the given text is relevant to Energy Utilities.

    Args:
        prompt (str): The text to check for relevance.

    Returns:
        int: 1 if the text is relevant, 0 otherwise.
    '''

    response = client_instructor.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYS_PROMPTS['check_relevance']},
        {"role": "user", "content": prompt}
    ],
    response_model = check_relevance_output
    )

    return response

def generate(query, reranked_texts):
    '''
    Generates a response for the given query.

    Args:
        query (str): The query to generate a response for.
        reranked_texts (list): The texts to generate a response from.
    
    Returns:
        str: The generated response.
    '''

    relevant = check_relevance(query)

    if not relevant.relevant:
        return "Sorry, as an AI for an Energy Utility company, I am unable to help with that. Please ask a question related to Energy Utilities."

    response = client_instructor.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYS_PROMPTS['chat']},
        {"role": "user", "content": f"user: {query}\nrelate texts: {reranked_texts}"},
        ],
    response_model = chat_response_output
    )

    return response.text