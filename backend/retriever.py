import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "agrichatbot")

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.6},
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    reranker  = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker, top_n=3)

    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever,
    )

    return final_retriever, vectorstore

def retrieve_with_parent(docs):
    expanded    = []
    seen_parents = set()
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_text      = doc.metadata.get("parent_text", doc.page_content)
            doc.page_content = parent_text
        expanded.append(doc)
    return expanded