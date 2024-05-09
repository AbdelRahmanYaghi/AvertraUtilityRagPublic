'''
This script is the main script that runs the data pipeline.
It extracts data from Wikipedia, embeds the data using the 
MixedBread API or a local model, and upserts the data into the database (Pinecone).
'''

import sys
import ExtractDataFromWikipedia
import upsert
import numpy as np
import embed

assert len(sys.argv) >= 2, "Please provide an argument for the parameters required: \n \"local\": 0 | 1"
assert sys.argv[1] == '0' or sys.argv[1] == '1', "Please provide a valid argument for parameter \"local\": 0 or 1."

model_name = "mixedbread-ai/mxbai-embed-large-v1"
texts = ExtractDataFromWikipedia.get_data()

if sys.argv[1] == '1':

    embed.download_model()

    model, tokenizer = embed.load_model()

    print("Embedding using local model.")
    embeddings = embed.embed_locally(texts, model, tokenizer)

    upsert.upsert_data(embeddings, texts)

else:
    import embed_api

    print("Embedding using MixedBread API.")

    embeddings = embed.embed_api(texts)

    upsert.upsert_data(embeddings, texts)