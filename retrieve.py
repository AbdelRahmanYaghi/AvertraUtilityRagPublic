from dotenv import load_dotenv
from pinecone import Pinecone
import os
from DataPipeline.embed import embed_locally, embed_api

load_dotenv()

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(host=os.environ.get('PINECONE_HOST'))
pinecone_namespace = os.environ.get('PINECONE_NAMESPACE')
MIXEDBREAD_API = os.environ.get('MIXED_BREAD_API_KEY')

def retrieve(text, locally, model = None, tokenizer = None):
    '''
    Retrieves the most similar text to the given text.

    Args:
        text (str): The text to find similar text for.
        locally (int): Whether to embed the text locally or not.
        model (AutoModel): The model to use for embedding the text.
        tokenizer (AutoTokenizer): The tokenizer to use for embedding the text.

    Returns:
        list: The most similar text to the given text.
    '''
    locally = str(locally)
    assert locally == '1' or locally == '0', '"locally" should be either 1 or 0.'

    if locally == '1':

        assert model is not None and tokenizer is not None, 'When choosing locally, neither the model nor the tokenizer can be None.'

        embeddings = embed_locally(text, model, tokenizer)
        embeddings = embeddings.tolist()

    else:
        print("Embedding using MixedBread API.")

        embeddings = embed_api(text)

    returned = index.query(
        namespace=pinecone_namespace,
        vector=embeddings[0],
        top_k=10,
        include_metadata=True
    )

    possible_texts = [match.metadata['text'] for match in returned.matches]

    return possible_texts

def rerank(possible_texts, query, locally, n = 3, model = None, tokenizer = None, device = 'cpu'):
    '''
    Reranks the possible texts based on the query.

    Args:
        possible_texts (list): The possible texts to rerank.
        query (str): The query to rerank the texts with.
        locally (int): Whether to rerank the texts locally or not.
        n (int): The number of texts to return.
        model (AutoModel): The model to use for reranking the texts.
        tokenizer (AutoTokenizer): The tokenizer to use for reranking the texts.
        device (str): The device to use for reranking the texts.

    Returns:
        list: The reranked texts.
    '''
    locally = str(locally)
    assert locally == '1' or locally == '0', '"locally" should be either 1 or 0.'

    if locally == '1':
        import torch
    
        docs = [[query, doc] for doc in possible_texts]


        print('Reranking text locally...')
        inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    
        with torch.no_grad():
            outputs = model(**inputs)
            chosen_indicies = outputs.logits.flatten().argsort(descending=True)[:n]
            return [possible_texts[i] for i in chosen_indicies]
            
    else:
        from mixedbread_ai.client import MixedbreadAI

        mxbai = MixedbreadAI(api_key=MIXEDBREAD_API)

        print('Reranking text via API...')

        out = mxbai.reranking(
            model="mixedbread-ai/mxbai-rerank-large-v1",
            query=query,
            input=possible_texts,
            top_k=n,
            return_input=False
        )

        return ([possible_texts[i.index] for i in out.data])