import os
from pathlib import Path
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_DIR = Path("index/chroma_db")


EMBEDDING_MODEL= os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

def search_document(question: str,k: int = 3):
    store = get_vectorstore()
    results = store.similarity_search_with_score(
        question,
        k=k
    )

    for doc,score in results:
        print("\n--- Retrieved Document ---")
        print("Content: ", doc.page_content)
        print("Metadata: ", doc.metadata)
        print("Distance: ", score)

if __name__ == "__main__":
    question = input("ASk a question: ")
    search_document(question)
