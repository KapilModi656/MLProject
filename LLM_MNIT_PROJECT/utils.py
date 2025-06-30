from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.youtube import YoutubeLoader
import re
from validators import url as is_valid_url
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,JinaSearch
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import os
from dotenv import load_dotenv
import json
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
    try:
        # Extract video ID from URL
        video_id = url.split("v=")[-1].split("&")[0]

        # Override internal proxy config (hacky but works)
        from youtube_transcript_api._api import _TranscriptApi
        _TranscriptApi._TranscriptApi__proxy_config = PROXIES  # override private proxy config

        # Test API call (optional but safe)
        _ = YouTubeTranscriptApi.list_transcripts(video_id)

        # Now use LangChain loader
        loader = YoutubeLoader.from_youtube_url(url, language=["en", "en-IN", "hi"])
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    except (VideoUnavailable, TranscriptsDisabled, NoTranscriptFound) as e:
        return [f"Transcript error for {url}: {str(e)}"]
    except Exception as e:
        return [f"Proxy/Network error for {url}: {str(e)}"]
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
