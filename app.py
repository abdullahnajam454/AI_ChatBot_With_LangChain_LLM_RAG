
import streamlit as st
from chat import ask_question

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📚"
)

st.title("📚 Document RAG Chatbot")
st.write("Ask questions about your PDF documents.")

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            for source in message.get("sources", []):
                st.caption(
                    f"Source: {source['source']} | "
                    f"Page: {source['page']}"
                )

# Get the user's question
question = st.chat_input("Ask a question about your documents...")

if question:
    # Display and save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask_question(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                for source in sources:
                    st.caption(
                        f"Source: {source['source']} | "
                        f"Page: {source['page']}"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                st.error(f"Something went wrong: {error}")
