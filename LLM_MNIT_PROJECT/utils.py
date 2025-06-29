from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
import re
from validators import url as is_valid_url
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,JinaSearch
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
import os
from dotenv import load_dotenv
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

def save_uploaded_file(file):
    # Get a temporary file path in the system's temp directory
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.name)

    # Save the file content
    with open(file_path, "wb") as f:
        f.write(file.read())

    return file_path
def docreader(file_path):
    if file_path.endswith(('.pdf', '.txt', '.docx', '.pptx')):
        loader = UnstructuredFileLoader(file_path)
        docs=loader.load()
        docs= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    else:
        raise ValueError("Unsupported file type. Supported types are: .pdf, .txt, .docx, .pptx")
    
    return docs

def doc_prompt_merger(prompt, doc):
    if not doc:
        return prompt
    if isinstance(doc, str):
        return f"{prompt}\n\nDocument:\n{doc}"
    elif isinstance(doc, list):
        return f"{prompt}\n\nDocuments:\n" + "\n".join(doc)
    else:
        raise ValueError("Document must be a string or a list of strings.")
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
def youtube_reader(url):
    
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return docs
def web_reader(url):
    loader = UnstructuredLoader(web_url=url)
    docs = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return docs

def wikipedia_tool(prompt):
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    wiki_wrapper = WikipediaAPIWrapper()
    
    wiki_tool = WikipediaQueryRun(wiki_wrapper)
    response= wiki_tool.run(prompt)
    return response

def arxiv_tool(prompt):
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    arxiv_wrapper = ArxivAPIWrapper()
    
    arxiv_tool = ArxivQueryRun(arxiv_wrapper)
    response= arxiv_tool.run(prompt)
    return response
def web_tool(prompt):
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    jina_search = JinaSearch(api_key=os.getenv("JINA_API_KEY"))
    
    response = jina_search.run(prompt)
    return response
