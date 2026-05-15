import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AgriBot 🌾",
    page_icon="🌾",
    layout="centered",
)

st.title("🌾 AgriBot — Smart Farming Assistant")
st.caption("Ask me anything about crops, soil, pests, weather, or mandi prices!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your farming question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🌱"):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"message": prompt, "chat_history": history},
                    timeout=120,
                )
                answer = response.json().get("answer", "Sorry, something went wrong.")
            except Exception as e:
                answer = f"⚠️ Error connecting to backend: {str(e)}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("🌾 AgriBot")
    st.markdown("""
    **Your AI farming assistant**

    **Capabilities:**
    - 🌱 PDF knowledge base search
    - 🌤️ Live weather info
    - 💹 Today's mandi prices
    - 🔍 Web search fallback
    - 💬 Chat with memory

    **Powered by:**
    - Groq LLaMA 3.3 70b
    - Pinecone Vector DB
    - BGE Embeddings + Reranker
    - LangChain Agent
    """)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()