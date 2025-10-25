import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.explainability.explain import Explainability
import streamlit as st
import json

st.set_page_config(page_title="RAG Explainability Dashboard", layout="wide")

@st.cache_resource
def load_components():
    retriever = Retriever(
        model_name="intfloat/e5-large",
        index_path="data/processed/hnsw_index.faiss",
        k=20
    )
    generator = Generator(model_name="google/flan-t5-base")
    explainer = Explainability(generator.model, generator.tokenizer)
    with open("data/processed/contexts.json", "r") as f:
        contexts = json.load(f)
    return retriever, generator, explainer, contexts


retriever, generator, explainer, contexts = load_components()

st.title("🧠 RAG Explainability Dashboard")
st.write("Ask a question and see how the model retrieves, generates, and explains answers.")

query = st.text_input("Enter your question:", "")

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Running retrieval and generation..."):
            indices, distances = retriever.search(query)
            top_contexts = [contexts[i] for i in indices]
            combined_context = top_contexts[0]


            # Generate answer and details
            answer, logits, attention = generator.generate_answer(
                query, combined_context, return_details=True
            )

            # Explain (without visualization)
            result = explainer.explain(query, combined_context, answer, logits, attention)

        # 📝 Answer
        st.subheader("📝 Answer")
        st.success(answer)

        # 🔍 Highlighted Context
        st.subheader("🔍 Highlighted Context")
        st.markdown(result["highlighted_context"])

        # 📊 Confidence Score
        st.subheader("📊 Confidence Score")
        st.metric(label="Confidence", value=f"{result['confidence']:.3f}")

        # 🎯 Attention Map (heatmap last)
        st.subheader("🎯 Attention Map")
        explainer.visualize_attention(attention, explainer.tokenizer.tokenize(query))
