<h1> Avertra Utility RAG (Retrieval Augmented Generation) Assignment </h1>
<h2> Table of Contents </h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#methodology">Methodology</a></li>
    <ol>
        <li><a href="#data">Data</a></li>
        <li><a href="#retrieval">Retrieval</a></li>
        <li><a href="#generation">Generation</a></li>
    </ol>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#examples">Examples</a></li>
</ul>

<h2 id="overview"> Overview </h2>
<p> This repository contains the code for the Avertra Utility RAG (Retrieval Augmented Generation) Assignment. The assignment is to implement a RAG model in the field of utility that can generate a response to a given input question. </p>

<h2 id="methodology"> Methodology </h2>
<h3 id="data"> Data </h3>
<p> The data for the assignment was extracted from Wikipedia, as required by the assignment. First, we start by adding terms of our interest in the file <code>/DataPipeline/topics.txt</code>. These terms could include topics such as: </p>
<ul>
    <li> Renewable Energy </li>
    <li> Electricity </li>
    <li> Power Generation </li>
    <li> etc.. </li>
</ul>
<p> Next, we run the script <code>/DataPipeline/data_pipeline.py</code>. This scripts consists of three parts: </p>

<ol>
  <li> <code>/DataPipeline/ExtractDataFromWikipedia.py</code>: Extracting data from Wikipedia. </li> 
  <li> <code>/DataPipeline/embed.py</code>: Embedding the extracted sentences. </li> 
  <li> <code>/DataPipeline/upsert.py</code>: Upserting the embeddings to Pinecone. </li> 
</ol>

<h4> Data Extration - <code>ExtractDataFromWikipedia.py</code> </h4>
This script extracts data from Wikipedia based on the terms wrote inside <code>/DataPipeline/topics.txt</code>. It works by first using the API to search for each one of the topics inside the topics text file. It then extracts the PageID for the first hit of each result, resulting in a list of PageIDs corresponding to each topic. After that, it uses the "requests" library from Python to extract the HTML source of each one of the webpages, with the corresponding PageID. The URLs for the webpages can be given in: <br>
<code>https://en.wikipedia.org/?curid={wikipedia_pageid}</code><br>
The reason behind manually scraping the wikipedia page, rather than using a pre-built library is to insure scalablity and flexibility, since we can customize the scraping process according to our needs. 
After scraping and parsing the HTML source, we get text paragraphs from each page, which are then chunked into smaller paragraphs.
Finally, to insure that the context of each of the paragraphs isn't lost across the chunking, sliding window is applied over the paragraphs, resulting in the final form of the data. The following image summarizes the Data Extration.

<figure>
  <img src="images/DataPipeline.png" alt="Data Pipeline">
</figure>

<h4> Embedding - <code>embed.py</code> </h4>
This script is responsible for embedding the extracted sentences using either a local embedding model or an API. The embedding process converts each sentence into a fixed-length vector representation. Mixed Bread was used because it offered an appropriate balance between performance and model size. In addition, it offered an API version, which gives you more options to use it in. Specifically, <a href="https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1">mixedbread-ai/mxbai-embed-large-v1</a> was used from HuggingFace.
Keep in mind that each one of the options has its own advantages and disadvantages. Choosing to embed locally will require alot of memory and processing power, and might take a long time based on the specifications of your PC, however, the only cost you will pay is the cost of the electricity used to embed. On the other hand, choosing to embed using an API will does not require alot of computational power, but will cost more depending on the size and number of sentences.


<h4> Upserting - <code>upsert.py</code> </h4>
This script is used to upsert the embeddings to Pinecone, which is a vector database service. The upsert process involves adding new embeddings or updating existing ones in the database. The script connects to the Pinecone service using the provided API key and performs the upsert operation. Pinecone was chosen because it offered a free tier which could be suitable when testing or in development. 

<h3 id="retrieval"> Retrieval </h3>
<p>This part will talk about the retrieval done for the assignment, which was pretty simple. It consisted of two parts, first the retrieval, and then the reranking. We will go in more depth regarding the two parts here:</p>

<h4> Retrieval </h4>
Because of the way the code was written, i.e., functional, it allows us to re-use previously written function, which is optimal, because we already wrote functions for embedddings when embedding the data. Meaning that to embed the user's query, we used the previously written <code>Datapipeline/embed.py</code>. The retrieval used Cosine Similarity as a metric.

<h4> Reranking </h4>
Reranking improves the performance of the retrieval by reranking the results of the embeddings distances/similarities. We also provided options for reranking locally or using the API, using <code>mixedbread-ai/mxbai-rerank-xsmall-v1</code> and <code>mixedbread-ai/mxbai-rerank-large-v1</code> respectively. 

The figure below shows the Retrieval Pipeline.

<figure>
  <img src="images/RetrievalPipeline.png" alt="Retrieval Pipeline" width="600">
</figure>


<h3 id="generation"> Generation </h3>
<p>Here, we will cover how the generation was done, and what measures were taken to insure stability in the model.</p>

<h4> Model Used </h4>
<p>OpenAI's ChatGPT 3.5 Turbo was usde for its cheap price and convenience. No local option is existant at the moment, simply because we had no processing power enough to consistantly run it, or even test it, reliably. The model is fed this system prompt:<p> 

<code>
You are an assistant, and a data extractor for an Utilities Company.
You can read some relevant texts related to the question which will be given along with the question.
Hence you are required to answer accordingly. Do not relate from something outside the knowledge base provided.</code><br><br>

<p>This system prompt offers the model with the necessary information needed to get it started. The prompt was pretty simple, and works as intended.</p>


<h4> Guard Rails </h4>
<p>Guard rails insure that the model does *NOT* go out of scope, and to reduce the effect of "hallucination" in the model visible to the user. "hallucinations" are incorrect or misleading results that AI models can generate. In addition to that, we insure, by using guard rails, that the model is not used for anything beside its intended purpose. The reason behind that is that since we are the ones who are hosting the model, we are paying for its usage, whether that be locally or through an API.</p>
<p>For this assignment, only one simple guard rail was implemented, which proved to be sufficient for our use case. The guard rail implemented is "Relevance", which insures that the user asked something whethen the scope of our model. This simple guard rail had the following prompt:</p>
<code>
You will be given a certain text from the user.
You should determine whether the text is related to a question or an inquery about Energy utilities.
Examples of Energy utilities include, but are not limited to: electricity, energy, natural gas, water, waste, communication, smart cities, transport, etc...
return 1 if the text is related to Energy Utilities, and return 0 otherwise. Return your output in the following format:<br>
{
    "relevant": 1 | 0
} <br>
And only return the JSON. Do not return anything else.</code><br><br>



<p>"Instructor" python library, in addition to PyDantic, were used to insure that OpenAI's output is structured and formated. This could reduce, or prevent the chance of jailbreaking the model. Jailbreak is a method used to get past the security measures implemented, which allows the model to responed inappropriately or illegally.</p>   

The figure below shows the Generation in conjuction with the Retrieval Pipeline.

<figure>
  <img src="images/RetrievalGenerationPipeline.png" alt="Retrieval And Generation Pipeline" width="800">
</figure>

<h2 id="usage"> Usage </h2>
First, make sure to add the appropriate the API keys in the <code>.env</code>.

To use the data pipeline, follow these steps:
1. Add the topics of interest in the file "topics.txt".
2. Run the script "data_pipeline.py" using the command: ```python data_pipeline.py 1|0```, where 1 refers to embedding locally, and 0 refers to embedding through an API. 
3. The extracted data will be embedded and uploaded to the Pinecone API inserted in <code>.env</code>.

To use the UI streamlit test. Run:
<code>streamlit run UI.py</code>

To switch between local embedding mode, and API embedding mode, for generation. You can simply switch it on/off in the top left of the streamlit. Such as below:

<figure>
  <img src="images/LocalOptionPreview.gif" alt="LocallyOption" >
</figure>

<h3 id="examples"> Examples </h3>

<figure>
  <img src="images/example1.png" alt="Random Question" >
</figure>


<figure>
  <img src="images/example2.png" alt="DAN Jailbreak" >
</figure>


<figure>
  <img src="images/example3.png" alt="Question1" >
</figure>


<figure>
  <img src="images/example4.png" alt="Question2" >
</figure>