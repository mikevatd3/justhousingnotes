"""
This file is a test case for average precision, used in the paper 
@tripathy2026fair to validate their outputs from different language models.

Lower vs higher values mean what exactly?
"""

import torch
import plotly.express as px



def organize_inputs(
    actual_classification: torch.Tensor, # 1-d
    model_probabilities: torch.Tensor, # 1-d same len as actual_classification
):
    # Model probabilities is monotonically increasing so this basically reverses
    # the list 
    i = model_probabilities.argsort(descending=True) 
     # 0s and 1s -- 1 if the cubic function defined below is > 0, otherwise, 0
     # here we sort by the model prediction, so a reverse of the list
    actual = actual_classification[i]
    
    cumsum = actual.cumsum(0) # How many 1s have been observed at that point, counting down
    precision = cumsum / torch.arange(1, actual.shape[0] + 1) # How many 1s have been observed over number of observations at that point

    return actual, cumsum, precision


def average_precision(
    actual_classification: torch.Tensor, # 1-d
    model_probabilities: torch.Tensor, # 1-d same len as actual_classification
):
    actual, _, precision = organize_inputs(
        actual_classification,
        model_probabilities,
    )

    return sum(precision[actual == 1]) / actual.sum() 


def precision_recall_curve(
    actual_classification: torch.Tensor, # 1-d
    model_probabilities: torch.Tensor, # 1-d same len as actual_classification
):
    actual, cumsum, precision = organize_inputs(
        actual_classification,
        model_probabilities,
    )
    
    recall = cumsum / actual.sum() # How many 1s been observed over total 1s

    return precision, recall


def main():
    START_POS = 5
    t = torch.arange(-5,5,0.25)
    f_t = ((t**3 - 2*t**2 + 0.4) > 0).to(torch.float32)
    f_t_model = t.sigmoid()
    
    avg_prec = average_precision(f_t, f_t_model)
    precision, recall = precision_recall_curve(f_t, f_t_model)

    t = t[START_POS:]
    precision = precision[START_POS:]
    recall = recall[START_POS:]


    fig = px.line(x=t, y=precision)
    fig.add_traces(px.line(x=t, y=recall, color_discrete_sequence=["orange"]).data)
    fig.add_traces(px.line(x=t, y=precision/recall, color_discrete_sequence=["red"]).data)
    fig.add_hline(avg_prec, line_dash="dot")
    fig.show()


if __name__ == "__main__":
    main()
