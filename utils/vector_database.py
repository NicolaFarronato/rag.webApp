from llama_index.storage.index_store import MongoIndexStore
from llama_index.vector_stores import PineconeVectorStore

from pinecone import Pinecone
import os

pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_indexName = os.getenv("PINECONE_INDEX")


mongoUri = os.getenv("MONGODB_URI")
mongoDbName = os.getenv("MONGODB_DATABASE")


def build_pinecone_vector_store():
    pc = Pinecone(
        api_key=pinecone_key
    )

    pinecone_index = pc.Index(pinecone_indexName)

    return PineconeVectorStore(pinecone_index=pinecone_index)

def build_mongo_index():
    return MongoIndexStore.from_uri(uri=mongoUri, db_name=mongoDbName)