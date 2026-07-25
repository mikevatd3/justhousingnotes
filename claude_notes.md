# GPT-2 token embedding norms — session notes (2026-07-14)

## What we built

- `justhousingnotes/exploration/embedding_norms.py` — loads GPT-2's token
  embedding matrix (`model.transformer.wte.weight`), computes the L2 norm of
  every one of the 50,257 vocab vectors, plots a Plotly histogram, and
  annotates ~20 randomly sampled tokens directly on the chart (dotted vline +
  rotated token label, staggered to avoid overlap). Also writes
  `data/token_l2_norms.csv` (`id,token,l2_norm`, sorted ascending by norm).
- `data/norm_counts.csv` — user-created join of `data/token_counts.csv`
  (WikiText-103 frequency counts) and `data/token_l2_norms.csv`.
- `justhousingnotes/exploration/norm_vs_frequency.py` — scatter plot of L2
  norm (x) vs. WikiText-103 frequency count (y, log scale) from
  `norm_counts.csv`, hover text shows the token.

## Empirical finding

Norms: mean≈3.96, std≈0.43, range [2.45, 6.32]. Lowest-norm tokens are common
function words (` at`, ` in`, ` on`, ` for`); highest-norm tokens are
rare/garbled tokens (`SPONSORED`, byte-fragment tokens). **More frequent
tokens have smaller embedding norm** — counter to the naive intuition that a
bigger norm should make it easier to get a large dot product (logit) with a
token.

## Why this happens (the mechanism)

The per-step gradient on a token's embedding is roughly
`(softmax_prob − 1{token is the target}) × hidden_state`. A common word like
"the" actually *is* the target often, so it gets frequent positive pulls that
balance the negative pushes it gets the rest of the time — it settles at a
moderate norm aligned with the many contexts where it's correct.

A rare/junk token is almost never the target, so nearly every step it only
gets the negative term (softmax pushing its probability down in whatever
context it appears). That's a consistent push in one direction with almost
nothing pulling back, so its norm grows roughly unbounded over training. The
large norm isn't "useful" for prediction — it's the residue of a token that
spent training being told "not this one," accumulating in one direction. This
is the **representation degeneration** phenomenon: a pathology of training
dynamics, not an optimal geometric solution, and it's why the embedding space
becomes anisotropic (narrow-cone) rather than using its full dimensionality.

## Key papers (chronological)

1. **FRAGE: Frequency-Agnostic Word Representation** — Gong et al., NeurIPS
   2018, [arXiv:1809.06858](https://arxiv.org/abs/1809.06858). First to flag
   that frequent vs. rare word embeddings occupy visibly different regions of
   the space; proposes adversarial training to make embeddings
   frequency-agnostic.
2. **Representation Degeneration Problem in Training Natural Language
   Generation Models** — Gao et al., ICLR 2019,
   [arXiv:1907.12009](https://arxiv.org/abs/1907.12009). Gives the
   theoretical account of *why* it happens (the gradient-imbalance mechanism
   above). Developed independently of FRAGE around the same time
   (~Sept 2018 submissions), but presented later (ICLR May 2019 vs. NeurIPS
   Dec 2018).
3. **Controlled Experiments for Word Embeddings** — Schakel & Wilson, 2015,
   [arXiv:1510.02675](https://arxiv.org/pdf/1510.02675). Classic word2vec-era
   paper: norm tracks word "significance" rather than raw frequency
   (non-monotonic — very frequent function words and very rare words can both
   end up with small norms, for different reasons).

## Follow-up: has word2vec-style explicit structure modeling been modernized?

Two distinct modern threads, both moving away from word2vec's implicit
"predict-context" objective toward directly targeting semantic/syntactic
structure (rather than getting structure as a side effect of a generative
objective, as in GPT-2/LLMs):

**Explicit-structure static embeddings** (direct descendants of word2vec):
- Retrofitting — Faruqui et al. 2015. Post-adjusts embeddings against WordNet
  relations so geometry matches lexical semantics, not just co-occurrence.
- Counter-fitting — Mrkšić et al. 2016,
  [arXiv:1603.00892](https://arxiv.org/pdf/1603.00892). Uses synonym/antonym
  constraints to pull/push vectors.
- Poincaré / hyperbolic embeddings — Nickel & Kiela 2017
  ([OpenReview](https://openreview.net/forum?id=Ske5r3AqK7)). Embeds words in
  hyperbolic space instead of Euclidean, because tree-like hierarchies
  (WordNet is-a relations) fit naturally there.

**Probing implicit structure in LLMs** (rather than engineering it in):
- Structural probe — Hewitt & Manning 2019,
  [ACL Anthology](https://aclanthology.org/N19-1419/). Shows syntax trees are
  linearly recoverable from BERT/ELMo hidden states: a learned linear map
  makes squared L2 distance encode parse-tree distance, and squared L2 norm
  encode tree depth — even though the training objective never mentioned
  syntax.

**Tradeoff**: explicit-structure embeddings are interpretable and directly
optimized for the structure you care about, but static (one vector per word)
and weaker on context-dependent tasks. LLM representations are richer and
context-sensitive, but structure is only recoverable after the fact via
probing — an emergent side effect, not a training target.
