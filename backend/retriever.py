# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.retrievers.multi_query import MultiQueryRetriever

# load_dotenv()

# EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "agrichatbot")

# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name=EMBED_MODEL,
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#         cache_folder="/app/model_cache",  # uses model pre-downloaded at build time
#     )

# def get_vectorstore(embeddings):
#     return PineconeVectorStore(
#         index_name=INDEX_NAME,
#         embedding=embeddings,
#     )

# def build_retriever(llm):
#     embeddings  = get_embeddings()
#     vectorstore = get_vectorstore(embeddings)
#     base_retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.6},
#     )
#     multi_query_retriever = MultiQueryRetriever.from_llm(
#         retriever=base_retriever,
#         llm=llm,
#     )
#     return multi_query_retriever, vectorstore

# def retrieve_with_parent(docs):
#     if not docs:
#         return []
#     expanded     = []
#     seen_parents = set()
#     for doc in docs:
#         parent_id = doc.metadata.get("parent_id")
#         if parent_id and parent_id not in seen_parents:
#             seen_parents.add(parent_id)
#             parent_text      = doc.metadata.get("parent_text", doc.page_content)
#             doc.page_content = parent_text
#         expanded.append(doc)
#     return expanded




import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

load_dotenv()

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "agrichatbot")

# ── Embeddings ──
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="/app/model_cache",
    )

# ── Vectorstore ──
def get_vectorstore(embeddings):
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

# ── Fetch docs for BM25 ──
def fetch_docs_for_bm25(vectorstore):
    """
    Fetch representative docs from Pinecone
    to build BM25 keyword index
    """
    print("⏳ Building BM25 keyword index...")

    sample_queries = [
        "crop farming India",
        "fertilizer soil nutrients NPK DAP urea",
        "pest disease management insecticide",
        "irrigation water farming drip sprinkler",
        "kharif rabi zaid crops sowing",
        "wheat rice paddy maize cultivation",
        "organic farming biofertilizer compost",
        "government scheme farmer subsidy PM Kisan",
        "soil health preparation tillage",
        "seed variety selection hybrid",
        "harvest post harvest storage",
        "vegetable fruit horticulture farming",
        "weed control herbicide",
        "micronutrient zinc boron deficiency",
        "crop rotation intercropping",
    ]

    docs = []
    seen = set()

    for query in sample_queries:
        try:
            results = vectorstore.similarity_search(query, k=15)
            for doc in results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    docs.append(doc)
        except Exception as e:
            print(f"⚠️ BM25 fetch error for '{query}': {e}")
            continue

    print(f"✅ BM25 index built with {len(docs)} unique docs!")
    return docs

# ── Build Retriever ──
def build_retriever(llm):
    embeddings  = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    # ── Semantic Retriever (MMR) — tuned ──
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k"           : 5,
            "fetch_k"     : 20,  # increased for better candidate pool
            "lambda_mult" : 0.7, # slightly more relevance-focused
        },
    )

    # ── Keyword Retriever (BM25) ──
    bm25_docs = fetch_docs_for_bm25(vectorstore)
    bm25_retriever   = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 5

    # ── Hybrid = BM25 (40%) + Semantic MMR (60%) ──
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],
    )

    # ── MultiQuery on top of Hybrid ──
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=llm,
    )

    return multi_query_retriever, vectorstore

# ── Parent Doc Expander ──
def retrieve_with_parent(docs):
    if not docs:
        return []
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



