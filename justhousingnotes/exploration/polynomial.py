import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# MODEL
torch.manual_seed(76)

HIDDEN_SIZE = 8
DOMAIN = [-2, 4]

to_learn = lambda ins: ins ** 3 - 3 * ins ** 2 - ins

model = nn.Sequential(
    nn.Linear(1, HIDDEN_SIZE), # These are linear transforms
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, 1),
)

xs = torch.FloatTensor(5000).uniform_(*DOMAIN).unsqueeze(1)
ys = to_learn(xs)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(5000):
    pred = model(xs)
    loss = loss_fn(pred, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Visualization

sd = model.state_dict()
w = sd['0.weight'].squeeze().numpy()
b = sd['0.bias'].numpy()
x = np.linspace(-2, 4, 200)
w2 = sd['2.weight'].squeeze().numpy()

COLS = 2
ROWS = len(w) + 1

grid_xs = torch.arange(*DOMAIN, step=0.01).unsqueeze(1)
grid_ys = model(grid_xs).squeeze().detach().numpy()

graph_xs = grid_xs.squeeze().detach().numpy()
function_ys = to_learn(grid_xs).squeeze().detach().numpy()

ymin = min(np.maximum(0, w[i]*x + b[i]).min() for i in range(len(w)))
ymax = max(np.maximum(0, w[i]*x + b[i]).max() for i in range(len(w)))

funcmax = max(function_ys) * 1.1
funcmin = -funcmax

fig = make_subplots(
    rows=ROWS,
    cols=COLS,
    shared_xaxes=True,
    shared_yaxes=True,
    column_widths=[0.5, 0.05],
    row_heights=[150] * (ROWS - 1) + [300],
    vertical_spacing=0.02,
    horizontal_spacing=0.02,
)

fig.add_trace(
    go.Scatter(x=graph_xs, y=grid_ys, name="Modeled"),
    row=ROWS,
    col=1,
)

fig.add_trace(
    go.Scatter(x=graph_xs, y=function_ys, name="Actual"),
    row=ROWS,
    col=1,
)

for i in range(len(w)):
    y = np.maximum(0, w[i]*x + b[i])
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'), row=i+1, col=1)
    fig.add_trace(go.Scatter(
        x=[0,0],
        y=[0,w2[i]],
        marker={
            "symbol":[
                'circle', 
                'triangle-up' if w2[i] > 0 else 'triangle-down'
            ], 
            "size": [0, 10],
            "color": "black"
        },
    ), row=i+1, col=2)

    knot_point = (-b[i]/w[i])
    if (-2 <= knot_point <= 4):
        fig.add_vline(x=knot_point, row=i+1, col=1, line_dash="dot")
        fig.add_vline(x=knot_point, row=ROWS, col=1, line_dash="dot")

fig.update_yaxes(range=[ymin, ymax], col=1)
fig.update_yaxes(range=[-max(w2) * 1.1, max(w2) * 1.1], col=1)
fig.update_yaxes(range=[funcmin, funcmax], row=ROWS)

fig.add_annotation(
    text="Model output for each neuron (one linear transform & ReLU)", 
    x=-0.07, y=0.87, xref='paper', yref='paper',
    showarrow=False, textangle=-90, font=dict(size=14),
)
fig.add_annotation(
    text="Combined Output", x=-0.07, y=0.0, xref='paper', yref='paper',
    showarrow=False, textangle=-90, font=dict(size=14),
)

fig.add_annotation(
    text="Layer two (multiplies the model output before combination)", 
    x=1.04, y=0.87, xref='paper', yref='paper',
    showarrow=False, textangle=-90, font=dict(size=14),
)

fig.update_layout(
    margin={"l": 40, "r": 40, "t": 20, "b": 10},
    width=650,
    height=750,
    showlegend=False
)

fig.show()

