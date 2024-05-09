import requests
import json
from urllib.parse import quote
from bs4 import BeautifulSoup
import re

SEARCH_URL = "https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={term}&srlimit={limit}"
EXECLUDED_SECTIONS = ['Content', 'External links', 'References', 'Further reading', 'See also', 'Scientific journals']
EXECLUDED_CLASSES = ['figure', 'div', 'table']

with open('banned_sentences.txt', 'r') as f:
    EXECLUDED_TEXTS = [i.strip() for i in f.readlines()]

def get_topics():
    """
    Retrieves a list of topics from a file called 'topics.txt'.

    Returns:
        list: A list of topics read from the file.
    """
    with open('topics.txt', 'r') as f:
        topics = f.readlines()
        topics = [topic.strip() for topic in topics]

    return topics

def search_wikipedia(term, n = 5):
    """
    Searches Wikipedia for a given term.

    Args:
        term (str): The term to search for.
        n (int): The number of results to return.

    Returns:
        dict: The search results.
    """
    url = SEARCH_URL.format(term=quote(term), limit = n)
    response = requests.get(url)
    data = json.loads(response.text)
    return data['query']['search'][0]['pageid']

def extract_parse_html(wikipedia_pageid):
    texts = ['']
    url = f'https://en.wikipedia.org/?curid={wikipedia_pageid}'
    response = requests.get(url).content
    soup = BeautifulSoup(response, 'html.parser')

    title = soup.find('h1', string=True).text
    body = soup.find("div", class_="mw-content-ltr mw-parser-output")
    
    for class_name in EXECLUDED_CLASSES:
        for element in body.find_all(class_name):
            if class_name == 'table' and 'class' in element.attrs:
                if 'wikidata' != element.attrs['class']:
                    continue

            element.decompose()

    subtitle = 'Introduction'
    new_paragraph = False

    for item in body.find_all():
        # Your code here
        if item.name in ['h2', 'h3', 'h4', 'h5']:
            subtitle = item.text.strip().replace('[edit]', '')
            continue

        if item.name == 'p' and item.find('b') and item.find('big'):
            subtitle = item.text.strip()
            continue
        elif item.name == 'p' and item.find('b'):
            continue

        if subtitle in EXECLUDED_SECTIONS:
            continue

        found_bad_text = False
        for EXECLUDED_TEXT in EXECLUDED_TEXTS:
            if EXECLUDED_TEXT in item.text.strip():
                found_bad_text = True
        
        if found_bad_text:
            continue

        if item.text.strip() in ['', '\n']:
            continue

        if item.name == 'p':
            clean_text = re.sub(r'\[[\w\s]+\]', '', item.text.strip())
            if clean_text != '':
                texts.append(f'About {title}, {subtitle}: {clean_text}')
                
        if item.name in ['a', 'li', 'table']:
            clean_text = re.sub(r'\[[\w\s]+\]', '', item.text.strip())
            if clean_text != '':
                texts[-1] += f' {clean_text} '

    return texts

def get_data():
    """
    Retrieves data from Wikipedia for a list of topics.

    Returns:
        list: A list of texts extracted from the Wikipedia pages.
    """
    topics = get_topics()
    page_ids = [search_wikipedia(topic) for topic in topics]
    page_texts = [extract_parse_html(page_id) for page_id in page_ids]
    flatten_texts = [item for sublist in page_texts for item in sublist if item != '']

    sliding_window_texts = []
    size_limit = 256
    for i in range(2, len(flatten_texts)):
        chunk_text = (flatten_texts[i-2] + '\n' + flatten_texts[i-1] + '\n' + flatten_texts[i])
        if len(chunk_text.split(' ')) <= size_limit:
            sliding_window_texts.append(chunk_text)
        elif len((flatten_texts[i-1] + '\n' + flatten_texts[i]).split(' ')) <= size_limit:
            sliding_window_texts.append(flatten_texts[i-1] + '\n' + flatten_texts[i])
        else:
            sliding_window_texts.append(flatten_texts[i])

    return sliding_window_texts
    