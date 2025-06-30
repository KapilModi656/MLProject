# retriever.py
# vector.py

import os
import asyncio
from dotenv import load_dotenv
from pymilvus import connections
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

load_dotenv()


def safe_connect():
    """
    Safely establish a connection to Milvus/Zilliz, ensuring
    that an event loop exists if needed by async internals.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No current event loop in this thread, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Connect to Milvus/Zilliz
    connections.connect(
        alias="default",
        uri=os.getenv("ZILLIZ_URI"),
        token=os.getenv("ZILLIZ_API_KEY"),
        # user=os.getenv("ZILLIZ_USER"),  # Optional if your cluster needs it
        # password=os.getenv("ZILLIZ_PASSWORD")
    )


def get_retriever() -> VectorStoreRetriever:
    safe_connect()

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize vector DB (Milvus or Zilliz)
    vectordb = Milvus(
        embedding_function=embeddings,
        collection_name="llm_mnit_project_collection",  # Must match your collection name in Zilliz
        connection_args={
            "uri": os.getenv("ZILLIZ_URI"),
            "token": os.getenv("ZILLIZ_API_KEY"),
            # Optional user/pass if your Zilliz cluster requires
            "user": os.getenv("ZILLIZ_USER", ""),
            "password": os.getenv("ZILLIZ_PASSWORD", ""),
        }
    )

    # Convert to retriever
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 2,
            "search_type": "similarity",  # use "similarity" or "hybrid"
            "score_threshold": 0.5
        }
    )

    return retriever
