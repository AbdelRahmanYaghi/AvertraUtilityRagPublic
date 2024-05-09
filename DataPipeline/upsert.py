from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(host=os.environ.get('PINECONE_HOST'))
pinecone_namespace = os.environ.get('PINECONE_NAMESPACE')

def upsert_data(embeddings, metadata):
    '''
    Upserts the data into the database.

    Args:
        embeddings (list): The embeddings of the data.
        metadata (list): The metadata of the data.
    '''

    data = []
    for indx, embd in enumerate(embeddings):
        data.append(
        {
            "id": str(indx),
            "values": embd,
            "metadata": {"text": metadata[indx]}
        }
        )

    index.upsert(
        data,
        namespace=pinecone_namespace
        )
    print("Data upserted successfully.")
