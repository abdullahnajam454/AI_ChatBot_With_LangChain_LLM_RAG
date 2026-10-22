import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DB_DIR=Path("index/chroma_db")
DATA_DIR = Path("data")
CHUNK_SIZE=1000
CHUNK_OVERLAP=100


def get_embedding():
    embedding = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    return embedding

def get_vectorstore() ->Chroma:
    return Chroma(
        collection_name= "docs",
        embedding_function=get_embedding(),
        persist_directory=DB_DIR,
    )
def load_document(path: Path = DATA_DIR):
    docs = []
    for f in sorted(path.rglob("*")):
        if f.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(f)).load())
        elif f.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(f), encoding="utf-8").load())
    return docs

def ingest(path : Path = DATA_DIR, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    docs = load_document(path)
    if not docs:
        raise FileNotFoundError(f"No .pdf/.txt/.md files found in '{path}'.")
    splitter = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap =  overlap)
    chunks = splitter.split_documents(docs)
    store = get_vectorstore()
    try:
        store.delete_collection()
    except Exception:
        pass
    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)
result = ingest(DATA_DIR, CHUNK_SIZE,CHUNK_OVERLAP)
print(result)