from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from retrieve import get_vectorstore

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions using the provided context.

Rules:
1. Answer the question using the context below.
2. If the context contains the answer, explain it clearly.
3. Do not say you don't know when the answer is present in the context.
4. Do not add facts that are not supported by the context.
5. If the context does not contain the answer, reply exactly:
"I don't know based on the available documents."

Context:
{context}

Question:
{question}

Answer:
""")
chain = prompt | llm



def ask_question(question: str, k: int = 5):
    store = get_vectorstore()

    results = store.similarity_search_with_score(
        question,
        k=k
    )

    if not results:
        return {
            "answer": "I don't know based on the available documents.",
            "sources": []
        }

    MAX_DISTANCE = 0.9

    relevant_results = [
        (doc, score)
        for doc, score in results
        if score <= MAX_DISTANCE
    ]

    if not relevant_results:
        return {
            "answer": "I don't know based on the available documents.",
            "sources": []
        }

    context = "\n\n".join(
        doc.page_content for doc, score in relevant_results
    )

    response = chain.invoke({
        "context": context,
        "question": question
    })

    sources = []

    for doc, score in relevant_results:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page_label", doc.metadata.get("page", "Unknown")),
            "distance": float(score)
        })

    return {
        "answer": response.content,
        "sources": sources
    }


if __name__ == "__main__":
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Chatbot closed.")
            break

        if question:
            ask_question(question)
