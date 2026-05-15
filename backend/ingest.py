import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PDF_DIR       = "data/pdfs"
EMBED_MODEL   = "BAAI/bge-large-en-v1.5"
INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME", "agrichatbot")
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
CHILD_SIZE    = 250

def load_pdfs():
    loader = DirectoryLoader(
        PDF_DIR,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages from PDFs")
    return docs

def chunk_documents(docs):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )

    parent_chunks = parent_splitter.split_documents(docs)
    print(f"✅ Parent chunks: {len(parent_chunks)}")

    child_chunks = []
    for i, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata.update({
                "parent_id"   : str(i),
                "parent_text" : parent.page_content,
                "source"      : parent.metadata.get("source", "unknown"),
                "page"        : parent.metadata.get("page", 0),
            })
            child_chunks.append(child)

    print(f"✅ Child chunks: {len(child_chunks)}")
    return child_chunks

def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✅ Created Pinecone index: {INDEX_NAME}")
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists")

def embed_and_store(chunks):
    print("⏳ Loading embedding model (first time may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("⏳ Uploading to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print(f"✅ Stored {len(chunks)} chunks in Pinecone!")
    return vectorstore

if __name__ == "__main__":
    print("── Starting ingestion ──")
    docs   = load_pdfs()
    chunks = chunk_documents(docs)
    init_pinecone()
    embed_and_store(chunks)
    print("── ✅ Ingestion complete! ──")