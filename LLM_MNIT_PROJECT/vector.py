# retriever.py

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
import os
from dotenv import load_dotenv
from pymilvus import connections

load_dotenv()

def get_retriever() -> VectorStoreRetriever:
    # Connect to Zilliz / Milvus
    connections.connect(
        alias="default",
        uri=os.getenv("ZILLIZ_URI"),  # store URI in .env as ZILLIZ_URI
        token=os.getenv("ZILLIZ_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Milvus(
        embedding_function=embeddings,
        collection_name="llm_mnit_project_collection",
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
            "filter": None,
            "search_type": "hybrid",
            "score_threshold": 0.3,
        }
    )

    return retriever
