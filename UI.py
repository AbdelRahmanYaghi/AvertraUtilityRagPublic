import streamlit as st
from DataPipeline.embed import load_model, load_reranker
from retrieve import retrieve, rerank
from generation import generate

@st.cache_resource
def load_all_models():
    '''
    Loads all the models and tokenizers from the specified path.

    Returns:
        model (AutoModel): The model loaded from the path.
        tokenizer (AutoTokenizer): The tokenizer loaded from the path.
        reranker (AutoModel): The reranker model loaded from the path.
        reranker_tokenizer (AutoTokenizer): The reranker tokenizer loaded from the path.
    '''

    model, tokenizer = load_model('DataPipeline/model_weights')
    reranker, reranker_tokenizer = load_reranker('DataPipeline/model_weights/reranker')

    return model, tokenizer, reranker, reranker_tokenizer

model, tokenizer, reranker, reranker_tokenizer = load_all_models()

with st.sidebar:
    embed_locally = st.toggle('Embed Locally', value=False)
    rerank_locally = st.toggle('Rerank Locally', value=False)

def model_response(text):
    '''
    Generates a response for the given text.

    Args:
        text (str): The text to generate a response for.

    Returns:
        str: The generated response.
    '''

    texts = retrieve(text, 1 if embed_locally else 0 , model, tokenizer)

    reranked_texts = rerank(texts, text, 1 if rerank_locally else 0, model = reranker, tokenizer = reranker_tokenizer, device='cpu')

    generated_reponse = generate(text, reranked_texts)

    return generated_reponse


messages = st.container(height=300)
if prompt := st.chat_input("Say something"):
    messages.chat_message("user").write(prompt)
    messages.chat_message("assistant").write(model_response(prompt))


        




