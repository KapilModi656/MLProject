from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders.youtube import YoutubeLoader
import re
from validators import url as is_valid_url
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,JinaSearch
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import os
from dotenv import load_dotenv
import json
from pytube import YouTube
from yt_dlp import YoutubeDL
from langchain.schema import Document
from langchain_cerebras import ChatCerebras
import time
load_dotenv()
def has_url(prompt):
    url_pattern = r'https?://\S+'
    return bool(re.search(url_pattern, prompt))
def find_urls(prompt):
    url_pattern = r'https?://\S+'
    return re.findall(url_pattern, prompt)
def is_file(user_input):
    if isinstance(user_input, dict):
        return bool(user_input.get("files"))
    else:
        return hasattr(user_input, "files") and bool(user_input.files)
import tempfile
PROXIES = {
    "http": os.getenv("HTTP_PROXY", "http://your-proxy:port"),
    "https": os.getenv("HTTPS_PROXY", "http://your-proxy:port")
}

def save_uploaded_file(file):
    # Get a temporary file path in the system's temp directory
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.name)

    # Save the file content
    with open(file_path, "wb") as f:
        f.write(file.read())

    return file_path
def docreader(path: str):
    ext = os.path.splitext(path)[-1].lower()

    try:
        if ext == ".pdf":
            try:
                # Try pure PDF loader first
                loader = PyPDFLoader(path)
            except Exception as e:
                print(f"[Fallback] Using UnstructuredPDFLoader for PDF: {e}")
                loader = UnstructuredPDFLoader(path)

        elif ext in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(path)

        elif ext in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(path)

        elif ext in [".txt", ".md"]:
            loader = TextLoader(path)

        else:
            print("[Warning] Unknown file type, using UnstructuredFileLoader")
            loader = UnstructuredFileLoader(path)

        docs = loader.load()
        docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        return docs

    except Exception as e:
        print(f"[Error] Failed to load file {path}: {e}")
        return []

def doc_prompt_merger(prompt, doc):
    if not doc:
        return prompt

    # If single Document object
    if hasattr(doc, "page_content"):
        return f"{prompt}\n\nDocument:\n{doc.page_content}"

    # If list of Document objects
    if isinstance(doc, list):
        # If items are Document objects, extract page_content, else treat as strings
        if len(doc) > 0 and hasattr(doc[0], "page_content"):
            content = "\n".join(d.page_content for d in doc)
        else:
            content = "\n".join(str(d) for d in doc)
        return f"{prompt}\n\nDocuments:\n{content}"

    # If it's a raw string
    if isinstance(doc, str):
        return f"{prompt}\n\nDocument:\n{doc}"

    raise ValueError("Document must be a string or a list of Document objects or strings.")
def theory_summarizer(context: str):
    from langchain.chains.summarize import load_summarize_chain
    """
    Summarize the given context using LangChain's map_reduce summarization chain.
    
    Args:
        context (str): Large text to summarize.
        openai_api_key (str): OpenAI API key for LLM access.
    
    Returns:
        str: Summary text.
    """
    # Initialize LLM (you can customize temperature, model, etc.)
    llm = ChatCerebras(model="llama-4-scout-17b-16e-instruct", api_key=os.getenv("CEREBRAS_API_KEY"),temperature=0.2)

    # Split the context into chunks for map_reduce
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    texts = text_splitter.split_text(context)
    
    docs = [Document(page_content=t) for t in texts]
    sleep_seconds=3
    # Map step: summarize each chunk separately, with sleep in between
    chunk_summaries = []
    for i, doc in enumerate(docs):
        prompt = f"Summarize the following text concisely:\n\n{doc.page_content}"
        summary = llm.predict(prompt)
        chunk_summaries.append(summary)
        print(f"Summarized chunk {i+1}/{len(docs)}")
        
        # Sleep to avoid rate limits
        if i < len(docs) - 1:
            time.sleep(sleep_seconds)
    
    # Reduce step: summarize all chunk summaries into one final summary
    combined_text = "\n\n".join(chunk_summaries)
    final_prompt = f"Summarize the following summaries into one concise summary:\n\n{combined_text}"
    final_summary = llm.predict(final_prompt)
    
    return final_summary
    
    return summary
def type_url(urls):
    urlty={"youtube": [], "arxiv": [], "web": []}
    for url in urls:
        if not is_valid_url(url):
            continue
        elif("youtube" in url or "youtu.be" in url):
            urlty["youtube"].append(url)
        elif("arxiv" in url):
            urlty["arxiv"].append(url)
        else:
            urlty["web"].append(url)
    return urlty
def youtube_reader(url: str):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'no_warnings': True,
        'forceurl': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        metadata = f"""Title: {info.get('title')}
Uploader: {info.get('uploader')}
Upload Date: {info.get('upload_date')}
Duration: {info.get('duration')} seconds
View Count: {info.get('view_count')}
Description: {info.get('description')}
"""
        docs = [Document(page_content=metadata, metadata={"source": url})]
        return docs
    except Exception as e:
        error_doc = Document(page_content=f"Error fetching metadata: {str(e)}", metadata={"source": url})
        return [error_doc]
def web_reader(url):
    from langchain_community.document_loaders import UnstructuredURLLoader
    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return docs

def wikipedia_tool(prompt):
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    wiki_wrapper = WikipediaAPIWrapper()
    
    
    response= wiki_wrapper.run(prompt)
    return response
def text_retriever(file_path):  # it's a file, not directory
    # Import required modules
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # Load the document
    docs = TextLoader(file_path).load()  # ✅ Returns a list of Document

    if not docs:
        raise ValueError(f"No documents found in {file_path}")

    # Split the document
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents=split_docs, embedding=embeddings)

    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 7})
def make_retriever(directory_path):
    """
    Create a retriever from a document path.
    Supports PDF, PPTX, DOCX, TXT, and MD files.
    """
    docs = PyPDFDirectoryLoader(directory_path).load()
    print("Files in directory:", os.listdir(directory_path))
    docs= RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(docs)
    if not docs:
        raise ValueError(f"No documents found in {directory_path}")

    # Create embeddings and vector store
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    dense_retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Sparse retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # Hybrid retriever (Ensemble)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    #
    return dense_retriever
      # Return as retriever
def arxiv_tool(prompt):
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    arxiv_wrapper = ArxivAPIWrapper()
    
    arxiv_query_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    result = arxiv_query_tool.invoke(prompt)
    return [result] if isinstance(result, str) else result
def web_tool(prompt):
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    jina_search = JinaSearch(api_key=os.getenv("JINA_API_KEY"))
    
    response = jina_search.invoke({"query": prompt})
    response=json.loads(response)
    final_response=response[0].get("link","")
    return final_response
