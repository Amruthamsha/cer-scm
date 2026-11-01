"""CER-SCM: Counterfactual Experience Replay via Structural Causal Models."""

from .causal_structure import (
    CAUSAL_VARS,
    build_causal_dataset,
    learn_causal_structure,
    diagnose_action_descendants,
)
from .scm import StructuralCausalModel
from .cer import (
    ACTION_TO_SCM,
    CounterfactualGenerator,
    UniformReplayBuffer,
    mixed_batch,
    variance_penalized_target,
)
from .metrics import compute_metrics

__all__ = [
    "CAUSAL_VARS",
    "build_causal_dataset",
    "learn_causal_structure",
    "diagnose_action_descendants",
    "StructuralCausalModel",
    "ACTION_TO_SCM",
    "CounterfactualGenerator",
    "UniformReplayBuffer",
    "mixed_batch",
    "variance_penalized_target",
    "compute_metrics",
]
