import sys, os, warnings, asyncio, torch, base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.explainability.explain import Explainability
import streamlit as st
import json

if sys.platform == "darwin":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

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


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_file = "dashboard/static/bg.png"
bg_image = get_base64_of_bin_file(bg_file)

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Happy+Monkey&family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&family=Quicksand:wght@300..700&display=swap');

    .stApp {{
        background-image: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.7)),
                          url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}

    div[data-testid="stTextInput"] label p {{
        font-weight: 700 !important;
        color: #111 !important;
        font-size: 1.05rem !important;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}

    .run-separator {{
        width: 100%;
        height: 1px;
        background-color: rgba(0,0,0,0.12);
    }}

    .content-card {{
        background: rgba(255,255,255,0.78);
        border-radius: 12px;
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
        padding: 0 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        border-left: 6px solid #7b4bff;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}
    .content-card h3 {{
        font-size: 1.4rem;
        font-weight: 700;
        color: #222;
        margin-bottom: 1rem;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}

    .data-box {{
        background-color: rgba(238,248,239,0.95);
        border-left: 4px solid #4CAF50;
        padding: 12px;
        border-radius: 6px;
        font-size: 16px;
        color: #222;
        margin-bottom: 1rem;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}
    .context-box {{
        background-color: rgba(250,250,250,0.9);
        border-left: 4px solid #bbb;
        padding: 12px;
        border-radius: 6px;
        color: #333;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}
    .score-box {{
        background-color: rgba(250,250,250,0.9);
        border-left: 4px solid #bbb;
        padding: 12px;
        border-radius: 6px;
        color: #333;
        font-size: 1.2rem;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}

    .title {{
        text-align: center;
        color: #00006B !important;
        font-size: 2.8rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.3rem;
        font-family: 'Quicksand', cursive !important;
    }}

    .subtitle {{
        text-align: center;
        color: #555;
        font-size: 1.4rem;
        margin-bottom: 2.5rem;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
    }}

    .run-btn button {{
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
        font-family: 'Lato', 'Quicksand', sans-serif !important;
        margin-top: -5rem !important;
    }}
    .run-btn button:hover {{
        background-color: #e03e3e !important;
    }}
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>📑 RAG Explainability Dashboard </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze how the model retrieves, generates, and explains answers with interpretability tools.</div>", unsafe_allow_html=True)


query = st.text_input("Enter your question💡", "")
st.markdown("<div class='run-btn'>", unsafe_allow_html=True)
run_clicked = st.button("Run", key="run_btn")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='run-separator'></div>", unsafe_allow_html=True)


if run_clicked:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Running retrieval and generation..."):
            indices, distances = retriever.search(query)
            top_contexts = [contexts[i] for i in indices]
            combined_context = top_contexts[0]
            answer, logits, attention = generator.generate_answer(query, combined_context, return_details=True)
            result = explainer.explain(query, combined_context, answer, logits, attention)

        st.markdown("<div class='content-card'><h3>Answer</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-box'>{answer}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        highlighted_html = (
            result["highlighted_context"]
            .replace("**[", "<mark style='background-color:#fff59d;'>")
            .replace("]**", "</mark>")
            .replace("[", "<mark style='background-color:#fff59d;'>")
            .replace("]", "</mark>")
        )
        st.markdown("<div class='content-card'><h3>Highlighted Context</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='context-box'>{highlighted_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='content-card'><h3>Confidence Score</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='score-box'>{result['confidence']:.3f}</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:14px; color:#666; font-style:italic;'>"
            "This score represents the model’s certainty in its generated answer, "
            "where values closer to 1.0 indicate higher confidence and reliability."
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='content-card'><h3>Attention Map</h3>", unsafe_allow_html=True)
        explainer.visualize_attention(attention, explainer.tokenizer.tokenize(query))
        st.markdown(
            "<p style='font-size:14px; color:#666; font-style:italic; margin-top:0.5rem;'>"
            "This visualization illustrates how the model attends to different input tokens "
            "while generating each output token. Darker regions indicate stronger attention "
            "weights, highlighting the parts of the input most influential in forming the answer."
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
