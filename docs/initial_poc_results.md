# Initial PoC Report — Vector Search + Top-k Retrieval

## Experiment Setup

- **Dataset:** SQuAD v2 (5% sample, ~6.5k QA pairs)
- **Embedder:** intfloat/e5-small (384 dimensions)
- **Vector Store:** FAISS (Flat index)
- **Generator:** google/flan-t5-small
- **Retriever:** Top-k = 3

## Example Query

- **Question:** "Who wrote the Declaration of Independence?"
- **Generated Answer:** _James Madison_
- **Retrieved Contexts:** Top-k passages were mostly related to James Madison, not Thomas Jefferson.

## Interpretation

- The pipeline runs successfully end-to-end: embedding → vector search → generation.
- The incorrect answer is due to retrieval not surfacing Jefferson-related passages.
- This highlights a limitation in **retrieval quality**, not in the generator itself.

## Implications

- Current PoC establishes a functional RAG pipeline but shows the need for accuracy improvements.
- Potential next steps:
  - Tune retrieval parameters (e.g., increase _k_)
  - Experiment with different FAISS index types
  - Add a reranker for improved ranking of candidate passages
  - Apply prompt engineering to better ground answers in retrieved contexts

## Next Actions

- Record additional sample queries and outcomes to expand the evaluation set.
- Prepare for October milestones:
  - Complete explainability module (highlighting, confidence scores, attention maps)
  - Build dashboard MVP (Streamlit/Gradio)
  - Collect experiment logs and examples for reporting
