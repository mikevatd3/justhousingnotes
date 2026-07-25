import csv

import plotly.graph_objects as go
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel

load_dotenv()

ids, tokens, counts, norms = [], [], [], []
with open("data/norm_counts.csv") as f:
    for row in csv.DictReader(f):
        ids.append(int(row["id"]))
        tokens.append(row["token"])
        counts.append(int(row["count"]))
        norms.append(float(row["l2_norm"]))

model = GPT2LMHeadModel.from_pretrained("gpt2")
embeddings = model.transformer.wte.weight.detach()
centered = embeddings - embeddings.mean(dim=0)
centered_norms = centered.norm(dim=1)[ids].numpy()


def make_scatter(x, x_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=counts,
        mode="markers",
        marker=dict(size=5, opacity=0.5),
        text=tokens,
        hovertemplate=f"token=%{{text}}<br>{x_title}=%{{x}}<br>count=%{{y}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"GPT-2 token frequency vs. {x_title}",
        xaxis_title=x_title,
        yaxis_title="Count in WikiText-103 (log scale)",
        yaxis_type="log",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


# Plot 1: raw embedding norm vs. frequency
fig = make_scatter(norms, "L2 norm")

# Plot 2: mean-centered embedding norm vs. frequency
fig_centered = make_scatter(centered_norms, "L2 norm (centered)")

# Plot 3: both norms overlaid on one scatter
fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(
    x=norms, y=counts, mode="markers", name="raw",
    marker=dict(size=5, opacity=0.5),
    text=tokens,
    hovertemplate="token=%{text}<br>l2_norm=%{x}<br>count=%{y}<extra></extra>",
))
fig_combined.add_trace(go.Scatter(
    x=centered_norms, y=counts, mode="markers", name="centered",
    marker=dict(size=5, opacity=0.5),
    text=tokens,
    hovertemplate="token=%{text}<br>centered l2_norm=%{x}<br>count=%{y}<extra></extra>",
))
fig_combined.update_layout(
    title="GPT-2 token frequency vs. embedding L2 norm: raw vs. mean-centered",
    xaxis_title="L2 norm",
    yaxis_title="Count in WikiText-103 (log scale)",
    yaxis_type="log",
    margin={"l": 40, "r": 40, "t": 60, "b": 40},
)

fig.show()
fig_centered.show()
fig_combined.show()
