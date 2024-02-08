from llama_index import (
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    LLMPredictor,
)

from llama_index.indices.loading import load_index_from_storage
from utils.vector_database import build_pinecone_vector_store, build_mongo_index
from mongodb.index import getExistingLlamaIndexes
from llama_index import SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding,ClarifaiEmbedding
from llama_index.llms import Perplexity
import os

llm = Perplexity(
    api_key=os.getenv("PERPLEXITY_API_KEY"), model="mixtral-8x7b-instruct", temperature=0.5
)

# Create a clarifai embedding class just with model_url, assuming that CLARIFAI_PAT is set as an environment variable
# embed_model = ClarifaiEmbedding(
#     model_url="https://clarifai.com/clarifai/main/models/BAAI-bge-base-en"
# )

# # Alternatively you can initialize the class with model_name, user_id, app_id and pat as well.
embed_model = ClarifaiEmbedding(
    model_name="BAAI-bge-base-en",
    user_id="clarifai",
    app_id="main",
    pat=os.getenv("CLARIFAI_API_KEY"),
)
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index_store = build_mongo_index()
vector_store = build_pinecone_vector_store()

# llm_predictor = LLMPredictor(llm=llm)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

storage_context = StorageContext.from_defaults(
    index_store=index_store,
    vector_store=vector_store
)

mongoIndex = None


def initialize_index():
    existing_indexes = getExistingLlamaIndexes()

    global mongoIndex

    if len(existing_indexes) > 0:
        print("Loading existing index...")

        mongoIndex = load_index_from_storage(
            service_context=service_context,
            storage_context=storage_context,
            llm=llm,
            index_id='mongo-index',
        )

        return createQueryEngine(mongoIndex)
    else:
        print("Building index...")

        mongoIndex = buildVectorIndex()

        return createQueryEngine(mongoIndex)


def createQueryEngine(index):
    return index.as_query_engine(response_mode="tree_summarize")
    # return index.as_query_engine(response_mode="simple_summarize", top_k=3)


def get_service_context():
    return service_context

def update_index(doc):
    mongoIndex.insert(doc)


def delete_document_from_index(doc_id):
    mongoIndex.delete_ref_doc(doc_id)


def buildVectorIndex():
    reader = SimpleDirectoryReader(
        input_files=["./data/userguide.pdf"]
    )
    
    documents = reader.load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context
    )

    index.set_index_id("mongo-index")

    return index
