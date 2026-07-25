# Running ideas list

- Density: how does the model handle longer prompts that basically say the same
  thing as shorter prompts?
    - For example: "The capital of France is" vs "The city that serves as the
      seat of government of France is"
- Invariant output: given a likeliest output token, is there a way to map the
  space of possible inputs? What if you relax to 'invariant top ten'?
    - What if you removing 'stop tokens'?
        - Is there such a thing as 'stop tokens'?
- Saturation: Sort of the opposite of density -- is there anything in the model
  that measures jargon-rich or highly elided phrases?
- Distribution: How do particular prompts change the distribution away from the
  'null' distribution where the model just spits out tokens basically at the
  same rate as what is observed in the training data?
- What is the distribution of embedding norm lengths in GPT2?
    - Fairly normal, centered around 4 (this is before mean-centering)
    - Roughly more common tokens are actually *shorter* than less common ones.
      This is covered apparently in 

## Human language topics

- Language Play
    - Formulaic sequences

