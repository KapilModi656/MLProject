from langchain_community.document_loaders import UnstructuredDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformer
import os
from dotenv import load_dotenv
from pimilvus import utilities
load_dotenv()
loader1= UnstructuredDirectoryLoader("LLM_MNIT_PROJECT/1stSem/pdf", glob="**/*.pdf", mode="elements", silent_errors=True)
loader2= UnstructuredDirectoryLoader("LLM_MNIT_PROJECT/1stSem/pptx", glob="**/*.pptx", mode="elements", silent_errors=True)

docs1= loader1.load()
docs2= loader2.load()
docs= docs1 + docs2
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",dim=300)
collection_name= "llm_mnit_project_collection"
if not utilities.has_collection(collection_name):
    utilities.create_collection(collection_name, embedding_dim=embeddings.dim)
else:
    utilities.drop_collection(collection_name)
    utilities.create_collection(collection_name, embedding_dim=embeddings.dim)

vectordb= Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={
        "uri": "https://in03-16ead69f302718e.serverless.gcp-us-west1.cloud.zilliz.com",
        "token": os.getenv("ZILLIZ_API_KEY"),
        "user": os.getenv("ZILLIZ_USER"),
        "password": os.getenv("ZILLIZ_PASSWORD"),
        "collection_name": collection_name,
        "secure": True,
    }
)
def get_retriever():
    return vectordb.as_retriever(
        search_kwargs={
            "k": 5,  # Number of documents to retrieve
            "filter": None,  # Optional filter for the search
            "search_type": "hybrid",  # Type of search to perform
            "score_threshold": 0.5,

        }
    )
