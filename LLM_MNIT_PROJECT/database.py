# database.py

from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
from pymilvus import connections, utility

load_dotenv()

def setup_vectordb():
    # Connect to Zilliz / Milvus server
    connections.connect(
        alias="default",
        uri="https://in03-16ead69f302718e.serverless.gcp-us-west1.cloud.zilliz.com",
        token=os.getenv("ZILLIZ_API_KEY")
    )

    # Load documents from PDF and PPTX folders
    loader1 = DirectoryLoader("1stSem/pdf", glob="**/*.pdf", silent_errors=True, loader_cls=UnstructuredFileLoader)
    loader2 = DirectoryLoader("1stSem/pptx", glob="**/*.pptx", silent_errors=True, loader_cls=UnstructuredFileLoader)

    docs1 = loader1.load()
    docs2 = loader2.load()
    docs = docs1 + docs2

    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    # Embedding model (MiniLM)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Milvus collection name
    collection_name = "llm_mnit_project_collection"

    # Drop collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Create Milvus vectorstore
    vectordb = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={
            "uri": "https://in03-16ead69f302718e.serverless.gcp-us-west1.cloud.zilliz.com",
            "token": os.getenv("ZILLIZ_API_KEY"),
            "user": os.getenv("ZILLIZ_USER"),
            "password": os.getenv("ZILLIZ_PASSWORD"),
            "collection_name": collection_name,
            
        }
    )
    return vectordb

# Lazy load vectordb singleton
_vectordb_instance = None
def get_vectordb():
    global _vectordb_instance
    if _vectordb_instance is None:
        _vectordb_instance = setup_vectordb()
    return _vectordb_instance

def get_retriever():
    vectordb = get_vectordb()
    return vectordb.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": None,
            "search_type": "hybrid",
            "score_threshold": 0.5,
        }
    )
retreiver= get_retriever()