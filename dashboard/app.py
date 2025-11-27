import sys, os, warnings, asyncio, torch, base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.explainability.explain import Explainability
import streamlit as st
import json
import pandas as pd
import time
import plotly.express as px
import time

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
    @import url('https://fonts.googleapis.com/css2?family=Happy+Monkey&family=Lato:wght@100;300;400;700;900&family=Quicksand:wght@300..700&family=Saira+Stencil+One&display=swap');

    .stApp {{
        background-image: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
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

    div[data-testid="stTextInput"] > div {{
        border: 2px solid rgba(0,0,0,0.4) !important;
        border-radius: 10px !important;
        background-color: rgba(255,255,255,0.90) !important;
    }}

    div[data-testid="stTextInput"] > div:focus-within {{
        border: 2px solid #00006B !important;
        box-shadow: 0 0 6px rgba(0,0,107,0.35) !important;
        background-color: rgba(255,255,255,0.98) !important;
    }}

    div[data-testid="stTextInput"] input {{
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        color: #111 !important;
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
        font-family: 'Saira Stencil One', sans-serif !important;
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

st.markdown("""
<style>

/* Tab container */
.stTabs [data-testid="stTabs"] {
    margin-top: 1rem;
}

/* Each Tab */
.stTabs [data-testid="stTab"] {
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.4rem !important;
    border-radius: 10px 10px 0 0 !important;
    color: #666 !important;
    background: rgba(255,255,255,0.5);
    transition: all .2s ease-in-out;
}

/* Hover */
.stTabs [data-testid="stTab"]:hover {
    color: #222 !important;
    background: rgba(255,255,255,0.8) !important;
}

/* Remove default Streamlit tab underline */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
    border-bottom: none !important;
}


/* Selected tab */
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.5) !important;
    color: #00006B !important;
    border-bottom: 3px solid #00006B !important;
    font-weight: 00 !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>📑 RAG Explainability Dashboard </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze how the model retrieves, generates, and explains answers with interpretability tools.</div>", unsafe_allow_html=True)
st.markdown("""
<style>
[data-testid="stRadio"] {
    margin-top: -2rem !important; 
    margin-bottom: -1rem !important; 
}
""", unsafe_allow_html=True)

tabs = st.tabs(["RAG Dashboard", "Question-Type Behavior Analysis"])

with tabs[0]:

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
                combined_context = " ".join(top_contexts[:5])
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
                "<p style='font-size:14px; color:#666; font-style:italic;'>This score represents the model’s certainty.</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='content-card'><h3>Attention Map</h3>", unsafe_allow_html=True)
            explainer.visualize_attention(attention, explainer.tokenizer.tokenize(query))
            st.markdown(
                "<p style='font-size:14px; color:#666; font-style:italic;'>Token attention visualization.</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 5])

    with col1:
        st.markdown(
        "<p style='font-size:18px; margin-top:7px;'><strong>"
        "Run Full Explainability Evaluation by Question Type"
        "</p>",
        unsafe_allow_html=True
    )

    with col2:
        eval_clicked = st.button("Run Evaluation")

    st.markdown(
        "<p style='font-size:15px; color:#777; margin-top:-10px;'>"
        "Uses: <code>question_set.csv</code> (500 queries, 9 types)"
        "</p>",
        unsafe_allow_html=True
    )

    if eval_clicked:
        start_time = time.time()
        time_display = st.empty()

        df = pd.read_csv("data/question_set.csv")
        logs = []
        progress = st.progress(0)
        total = len(df)

        preview_df = df.head(20)
        st.markdown("""
<div style="
    background: rgba(255,255,255,0.78);
    border-radius: 12px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    padding: 0.6rem 1.2rem;
    margin-bottom: 1rem;
    border-left: 6px solid #7b4bff;
    font-size: 1.35rem;
    font-weight: 700;
    color: #555;
">
    Preview of Evaluation Dataset (first 20 rows)
</div>
""", unsafe_allow_html=True)

        st.dataframe(preview_df)
        for idx, row in df.iterrows():
            q = row["question"]
            q_type = row["type"]
            gold_context = row["context"]
            if not isinstance(gold_context, str) or not gold_context.strip():
                gold_context = ""

            indices, distances = retriever.search(q)

            answer, logits, attention = generator.generate_answer(q, gold_context, return_details=True)
            result = explainer.explain(q, gold_context, answer, logits, attention)

            logs.append({
                "question": q,
                "type": q_type,
                "mean_similarity": float(sum(distances) / len(distances)),
                "variance_similarity": float(pd.Series(distances).var()),
                "generation_length": len(answer.split()),
                "context_attention": float(result["confidence"]),
                "answer": answer
            })

            progress.progress((idx + 1) / total)
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            if minutes > 0:
                time_display.markdown(
                    f"⏳ Running full evaluation... **{minutes}m {seconds}s elapsed**"
                )
            else:
                time_display.markdown(
                    f"⏳ Running full evaluation... **{seconds}s elapsed**"
                )

            time.sleep(0.1)

        pd.DataFrame(logs).to_csv("data/experiment_log.csv", index=False)
        st.success("Complete. Saved to data/experiment_log.csv")

with tabs[1]:

    log_file = "data/experiment_log.csv"

    if not os.path.exists(log_file):
        st.warning("Run full evaluation first")
    else:
        df = pd.read_csv(log_file)

        group_map = {
            "DATE": "FACTUAL",
            "LOCATION": "FACTUAL",
            "PERSON": "FACTUAL",
            "FACT": "FACTUAL",
            "STATISTICS": "FACTUAL",
            "ENTITY": "FACTUAL",
            "REASON": "REASONING",
            "PROCESS": "REASONING",
            "OPINION": "OPINION"
        }
        df["type_group"] = df["type"].map(group_map)

        view_mode = st.radio(
            "",
            ["3 Groups", "9 Categories"],
            index=0,
            horizontal=True
        )

        st.markdown(f"""
<div style="
    background: rgba(255,255,255,0.6);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid #00006B;
    font-size: 1.05rem;
    color: #000;
    line-height: 1.5;
">

<span style="font-size:1.2rem; font-weight:700;">3 Groups</span><br>
• FACTUAL (Date, Location, Person, Fact, Statistics, Entity)<br>
• REASONING (Why, How, Process)<br>
• OPINION (Subjective, Opinion based)

<span style="font-size:1.2rem; font-weight:700;">9 Categories</span><br>
• DATE, LOCATION, PERSON, FACT, REASON, STATISTICS, ENTITY, PROCESS, OPINION

</div>
""", unsafe_allow_html=True)

        x_col = "type_group" if view_mode == "3 Groups" else "type"

        st.markdown("""
        <div style="background:rgba(255,255,255,0.78);border-radius:12px;
        box-shadow:0 6px 14px rgba(0,0,0,0.08);padding:0.5rem 1.2rem;
        margin-top:1.5rem;margin-bottom:1.2rem;border-left:6px solid #7b4bff;">
        <h3 style="font-size:1.4rem;font-weight:700;color:#222;margin:0;">Mean Similarity Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        fig1 = px.box(df, x=x_col, y="mean_similarity", color=x_col)
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.78);border-radius:12px;
        box-shadow:0 6px 14px rgba(0,0,0,0.08);padding:0.5rem 1.2rem;
        margin-top:1.5rem;margin-bottom:1.2rem;border-left:6px solid #7b4bff;">
        <h3 style="font-size:1.4rem;font-weight:700;color:#222;margin:0;">Retrieval Variance Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        fig2 = px.box(df, x=x_col, y="variance_similarity", color=x_col)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.78);border-radius:12px;
        box-shadow:0 6px 14px rgba(0,0,0,0.08);padding:0.5rem 1.2rem;
        margin-top:1.5rem;margin-bottom:1.2rem;border-left:6px solid #7b4bff;">
        <h3 style="font-size:1.4rem;font-weight:700;color:#222;margin:0;">Generation Length Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        fig3 = px.box(df, x=x_col, y="generation_length", color=x_col)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.78);border-radius:12px;
        box-shadow:0 6px 14px rgba(0,0,0,0.08);padding:0.5rem 1.2rem;
        margin-top:1.5rem;margin-bottom:1.2rem;border-left:6px solid #7b4bff;">
        <h3 style="font-size:1.4rem;font-weight:700;color:#222;margin:0;">Context Attention Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        fig4 = px.box(df, x=x_col, y="context_attention", color=x_col)
        st.plotly_chart(fig4, use_container_width=True)
