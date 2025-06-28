from langchain-unstructured import UnstructuredLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_tavily import TavilySearchAPIWrapper
from validators import url as is_valid_url
from langchain_core.utils
def has_url(prompt):
    url_pattern = r'https?://\S+'
    return bool(re.search(url_pattern, prompt))
def find_urls(prompt):
    url_pattern = r'https?://\S+'
    return re.findall(url_pattern, prompt)
def is_file(user_input):
    return bool(user_input.files) if user_input else False
def docreader(file_path):
    if file_path.endswith('.pdf'|| '.txt' || '.docx'||"pptx"):
        loader = UnstructuredLoader(file_path)
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
    search = TavilyWebSearchAPIWrapper()
    docs = search.run(url)
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return docs

