import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import streamlit as st
import numpy as np

class Explainability:
    def __init__(self, generator_model, tokenizer):
        self.model = generator_model
        self.tokenizer = tokenizer

    def highlight_answer_context(self, answer: str, context: str) -> str:
        if not answer or not context:
            return context
        answer_lower = answer.lower()
        context_lower = context.lower()
        if answer_lower in context_lower:
            start = context_lower.index(answer_lower)
            end = start + len(answer)
            return context[:start] + f"**[{context[start:end]}]**" + context[end:]
        return context

    def compute_confidence(self, logits: torch.Tensor) -> float:
        if logits is None:
            return 0.0
        probs = torch.softmax(logits, dim=-1)
        return probs.max().item()

    def visualize_attention(self, attention: List[torch.Tensor], input_tokens: List[str], output_path: str = None):
        if not attention or not isinstance(attention, (list, tuple)):
            print("[WARN] No attention data available.")
            return None

        attn = attention[-1].mean(dim=1)[0].detach().cpu().numpy()
        n_tokens = min(len(input_tokens), attn.shape[0])
        attn = attn[:n_tokens, :n_tokens]
        tokens = input_tokens[:n_tokens]

        avg_scores = attn.mean(axis=0)
        top_k = min(10, n_tokens)
        top_idxs = np.argsort(avg_scores)[-top_k:]
        top_idxs = np.sort(top_idxs)

        attn_subset = attn[np.ix_(top_idxs, top_idxs)]
        tokens_subset = [tokens[i] for i in top_idxs]

        fig, ax = plt.subplots(figsize=(2.5, 2.5))

        sns.heatmap(
            attn_subset,
            cmap="YlOrRd",
            ax=ax,
            cbar=True,
            square=True,
            xticklabels=tokens_subset,
            yticklabels=tokens_subset,
            linewidths=0.3,
            linecolor="white"
        )

        ax.set_title("Top-10 Attention Map", fontsize=8, pad=6)
        ax.set_xlabel("Input Tokens", fontsize=7, labelpad=5)
        ax.set_ylabel("Output Tokens", fontsize=7, labelpad=5)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)
        plt.tight_layout(pad=0.5)

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
        else:
            st.pyplot(fig, clear_figure=True, use_container_width=False)

        plt.close(fig)

    def explain(self, query: str, context: str, answer: str, logits: torch.Tensor, attention: List[torch.Tensor]):
        highlighted = self.highlight_answer_context(answer, context)
        confidence = self.compute_confidence(logits)
        return {
            "highlighted_context": highlighted,
            "confidence": confidence
        }
