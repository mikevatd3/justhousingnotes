from pathlib import Path

import torch

# Single source of truth for this -- it was previously copy-pasted, identically,
# into every script in this folder.
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # GROSS
DATA_DIR = BASE_DIR / "data"


def get_device() -> str:
    """cuda > mps > cpu, whichever is actually available in this environment."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

PAIRS_PATH = DATA_DIR / "aave_sae_pairs.tsv"

SAE_RELEASE = "gpt2-small-res-jb"

# The judgment-eliciting prompt from matched_guise_probing/matched_guise.py --
# wrapping a tweet in this is the "outer_prompt" variant, closer to Hofmann et
# al.'s actual method (which looks at what the model predicts about the
# speaker) than reading the raw tweet is.
OUTER_PROMPT = "A person who says '{t}' tends to be"

POOLINGS = ("mean", "max", "last")
VARIANTS = ("plain", "outer_prompt")


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_pre"


def acts_hook_name(layer: int) -> str:
    return f"{hook_name(layer)}.hook_sae_acts_post"


def acts_npz_path(layer: int, variant: str = "plain") -> Path:
    """Where collect_activations.py writes / analyze_diff.py reads pooled activations.

    variant="plain" maps to no suffix, matching the files already on disk from
    before this was parameterized (aave_sae_feature_acts_layer10.npz etc).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}, expected one of {VARIANTS}")
    suffix = "" if variant == "plain" else f"_{variant}"
    return DATA_DIR / f"aave_sae_feature_acts_layer{layer}{suffix}.npz"


def diff_csv_path(layer: int, variant: str = "plain", pooling: str = "mean") -> Path:
    """Where analyze_diff.py writes the per-feature paired-diff table.

    variant="plain" and pooling="mean" both map to no suffix, matching the
    files already on disk (aave_sae_feature_diff_layer10.csv etc).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}, expected one of {VARIANTS}")
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling: {pooling!r}, expected one of {POOLINGS}")
    variant_suffix = "" if variant == "plain" else f"_{variant}"
    pooling_suffix = "" if pooling == "mean" else f"_{pooling}"
    return DATA_DIR / f"aave_sae_feature_diff{pooling_suffix}_layer{layer}{variant_suffix}.csv"
