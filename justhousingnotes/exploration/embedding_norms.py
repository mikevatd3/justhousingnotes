import csv
import random

import plotly.graph_objects as go
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer

load_dotenv()

N_SAMPLES = 20

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embeddings = model.transformer.wte.weight.detach()

norms = embeddings.norm(dim=1).numpy()

with open("data/token_l2_norms.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "token", "l2_norm"])
    for tok_id in norms.argsort():
        w.writerow([tok_id, tokenizer.decode([tok_id]), norms[tok_id]])

centered = embeddings - embeddings.mean(dim=0)
centered_norms = centered.norm(dim=1).numpy()

random.seed(76)
sample_ids = random.sample(range(len(norms)), N_SAMPLES)
sample_ids.sort(key=lambda i: norms[i])

STAGGER_LEVELS = 5


def add_token_annotations(fig, values, sample_ids):
    for i, tok_id in enumerate(sample_ids):
        token = tokenizer.decode([tok_id]).strip() or "<space>"
        fig.add_vline(x=values[tok_id], line_dash="dot", opacity=0.25)
        fig.add_annotation(
            x=values[tok_id],
            y=0,
            yref="paper",
            text=token,
            showarrow=False,
            textangle=-90,
            yshift=-10 - 18 * (i % STAGGER_LEVELS),
            font=dict(size=10),
        )


# Plot 1: raw embedding norms
fig = go.Figure()
fig.add_trace(go.Histogram(x=norms, nbinsx=100))
fig.update_layout(
    title=f"L2 norms of GPT-2 token embeddings (n={len(norms)})",
    xaxis_title="L2 norm",
    yaxis_title="Count",
    margin={"l": 40, "r": 40, "t": 60, "b": 170},
)
add_token_annotations(fig, norms, sample_ids)

# Plot 2: norms after subtracting the mean embedding vector
fig_centered = go.Figure()
fig_centered.add_trace(go.Histogram(x=centered_norms, nbinsx=100))
fig_centered.update_layout(
    title=f"L2 norms of GPT-2 token embeddings, mean-centered (n={len(centered_norms)})",
    xaxis_title="L2 norm (centered)",
    yaxis_title="Count",
    margin={"l": 40, "r": 40, "t": 60, "b": 170},
)
add_token_annotations(fig_centered, centered_norms, sample_ids)

# Plot 3: raw vs. centered norms overlaid
fig_combined = go.Figure()
fig_combined.add_trace(go.Histogram(x=norms, nbinsx=100, name="raw", opacity=0.6))
fig_combined.add_trace(go.Histogram(x=centered_norms, nbinsx=100, name="centered", opacity=0.6))
fig_combined.update_layout(
    title="GPT-2 token embedding L2 norms: raw vs. mean-centered",
    xaxis_title="L2 norm",
    yaxis_title="Count",
    barmode="overlay",
    margin={"l": 40, "r": 40, "t": 60, "b": 40},
)

print(f"raw:      mean={norms.mean():.3f} std={norms.std():.3f} min={norms.min():.3f} max={norms.max():.3f}")
print(f"centered: mean={centered_norms.mean():.3f} std={centered_norms.std():.3f} min={centered_norms.min():.3f} max={centered_norms.max():.3f}")

fig.show()
fig_centered.show()
fig_combined.show()
