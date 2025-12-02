import streamlit as st
import requests
from api_client import call_rag_api

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

st.title("RAG Assistant (MLOps Version)")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Getting answer..."):
            result = call_rag_api(query)

        st.subheader("Answer")
        st.write(result.get("answer", "No answer"))

        st.subheader("Context preview")
        st.code(result.get("context_preview", "No context"))