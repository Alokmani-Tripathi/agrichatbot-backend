import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agrichatbot")

def get_embeddings():
    # Uses HuggingFace Inference API — runs on HF servers, zero RAM on Render!
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY", ""),
        model_name="BAAI/bge-large-en-v1.5",
    )

def get_vectorstore(embeddings):
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

def build_retriever(llm):
    embeddings  = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.6},
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    return multi_query_retriever, vectorstore

def retrieve_with_parent(docs):
    expanded     = []
    seen_parents = set()
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_text      = doc.metadata.get("parent_text", doc.page_content)
            doc.page_content = parent_text
        expanded.append(doc)
    return expanded