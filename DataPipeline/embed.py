from mixedbread_ai.client import MixedbreadAI
from dotenv import load_dotenv
import os
import torch
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()
MIXED_BREAD_API_KEY = os.environ.get('MIXED_BREAD_API_KEY')

def embed_api(text):
    """
    Embeds text using the MixedBread API.

    Args:
        text (str): The text to embed.
        api_key (str): The API key to use for the request.

    Returns:
        dict: The embeddings of the text.
    """

    mxbai = MixedbreadAI(api_key=MIXED_BREAD_API_KEY)

    if isinstance(text, str):
        text = [text]

    embeddings = mxbai.embeddings(
        model="mixedbread-ai/mxbai-embed-large-v1",
        input=text,
        normalized=True
    )

    embeddings = [embeddings.data[i].embedding for i in range(len(text))]

    return embeddings


def load_model(path = './model_weights'):
    '''
    Loads the model and tokenizer from the specified path.

    Args:
        path (str): The path to the model and tokenizer.

    Returns:
        model (AutoModel): The model loaded from the path.
        tokenizer (AutoTokenizer): The tokenizer loaded from the path.
    '''
    print(f'Working on {device}')

    if not os.path.exists(path):
        download_model(path = path)


    model = AutoModel.from_pretrained(path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)

    print("Model loaded successfully.")

    return model, tokenizer

def load_reranker(path = './model_weights/reranker'):
    '''
    Loads the reranker model and tokenizer from the specified path.

    Args:
        path (str): The path to the model and tokenizer.

    Returns:
        model (AutoModel): The model loaded from the path.
        tokenizer (AutoTokenizer): The tokenizer loaded from the path.
    '''

    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)

    print("Tokenizer loaded successfully.")

    return model, tokenizer

def embed_locally(text, model, tokenizer):
    '''
    Embeds text locally using the specified model and tokenizer.

    Args:
        text (str): The text to embed.
        model (AutoModel): The model to use for embedding.
        tokenizer (AutoTokenizer): The tokenizer to use for embedding.

    Returns:
        torch.Tensor: The embeddings of the text.
    '''
    assert model is not None and tokenizer is not None, 'When choosing locally, neither the model nor the tokenizer can be None.'

    print('Embedding text...')
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings

def download_model(update_model = False, path = './model_weights/'):
    '''
    Downloads the model and tokenizer from the specified path.

    Args:
        update_model (bool): DEPRECATED
        path (str): The path to save the model and tokenizer.
    
    Returns:
        None
    '''
    if not os.path.exists(path):
        os.makedirs(path)

    os.environ["HF_HOME"] = path
    model_name = "mixedbread-ai/mxbai-embed-large-v1"

    if 'config.json' in os.listdir(path) and 'reranker' in os.listdir(path) and not update_model:
        print("A model already exists. Proceeding with the existing model.")
        return

    else:
        print("Downloading models...")
        model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1", force_download=True)
        tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1", force_download=True)
        tokenizer_rerenker = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-rerank-xsmall-v1")
        model_rerenker = AutoModelForSequenceClassification.from_pretrained("mixedbread-ai/mxbai-rerank-xsmall-v1")

        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        tokenizer_rerenker.save_pretrained(os.path.join(path, 'reranker'))
        model_rerenker.save_pretrained(os.path.join(path, 'reranker'))        