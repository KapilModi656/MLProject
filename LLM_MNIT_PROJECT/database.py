# database.py

from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from pymilvus import connections, utility
import asyncio
load_dotenv()

COLLECTION_NAME = "llm_mnit_project_collection"

def connect_to_milvus():
    # Ensure event loop exists for Milvus async internals
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    connections.connect(
        alias="default",
        uri=os.getenv("ZILLIZ_URI"),
        token=os.getenv("ZILLIZ_API_KEY"),
        user=os.getenv("ZILLIZ_USER"),
        password=os.getenv("ZILLIZ_PASSWORD"),
    )

def ingest_documents():
    """
    Load documents from folders, split, embed and upload to Milvus.
    Drops the old collection if exists, then recreates.
    """
    connect_to_milvus()

    # Load PDFs and PPTX documents
    loader_pdf = DirectoryLoader("1stSem/pdf", glob="**/*.pdf", silent_errors=True, loader_cls=UnstructuredFileLoader)
    loader_pptx = DirectoryLoader("1stSem/pptx", glob="**/*.pptx", silent_errors=True, loader_cls=UnstructuredFileLoader)

    docs_pdf = loader_pdf.load()
    docs_pptx = loader_pptx.load()
    all_docs = docs_pdf + docs_pptx

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # Prepare embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Drop collection if it exists
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Create vector store and upload documents
    vectordb = Milvus.from_documents(
        documents=split_docs,
        embedding=embeddings,
        connection_args={
            "uri": os.getenv("ZILLIZ_URI"),
            "token": os.getenv("ZILLIZ_API_KEY"),
            "user": os.getenv("ZILLIZ_USER"),
            "password": os.getenv("ZILLIZ_PASSWORD"),
            "collection_name": COLLECTION_NAME,
        }
    )
    return vectordb

def get_retriever():
    """
    Connect to Milvus and return a retriever object.
    Does NOT perform ingestion.
    """
    connect_to_milvus()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "uri": os.getenv("ZILLIZ_URI"),
            "token": os.getenv("ZILLIZ_API_KEY"),
            "user": os.getenv("ZILLIZ_USER"),
            "password": os.getenv("ZILLIZ_PASSWORD"),
        }
    )

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 5,
            "search_type": "hybrid",
            "score_threshold": 0.0,
        }
    )
    return retriever
