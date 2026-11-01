"""
Causal structure learning over financial time-series features (PC / NOTEARS-MLP).

Includes a post-analysis fix (not used for the results reported in the paper):
the synthetic 'action' proxy used for structure learning must have explicit,
non-empty descendants, or do-interventions on it collapse to a no-op.
See `_ensure_action_has_descendants` below. This is a stopgap. Handling
the case properly is an open research question.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from castle.algorithms import PC

try:
    from castle.algorithms import NotearsNonlinear
    _NOTEARS_OK = True
except ImportError:
    _NOTEARS_OK = False

CAUSAL_VARS = [
    "returns", "volatility", "volume_ratio",
    "momentum_5", "rsi", "bb_position", "macd_diff",
    "atr_ratio", "obv_signal", "price_vs_sma50",
]

# Variables 'action' is forced to have as structural children when the
# automatic learner finds none (see _ensure_action_has_descendants).
ACTION_FORCED_CHILDREN = ["volume_ratio", "returns_lag1"]


def _ensure_action_has_descendants(graph: np.ndarray, var_names: list):
    """
    Force edges action -> {volume_ratio, returns_lag1} if the learned graph
    gives 'action' no descendants at all.

    Why this is needed: 'action' is constructed as np.sign(momentum_5), a
    deterministic function of a variable already in the graph. PC/NOTEARS
    assume faithfulness, which breaks under exact determinism: action carries
    zero information beyond momentum_5, so the learner routinely assigns it
    no outgoing edges. A do-intervention on a childless node changes nothing
    downstream, so the "counterfactual" state collapses back to the observed
    state. Forcing a small, domain-justified edge set (a trade mechanically
    affects next-period volume and realized return) makes the intervention
    well-defined regardless of what the automatic learner finds elsewhere.

    Returns (graph, was_forced).
    """
    if "action" not in var_names:
        return graph, False
    a_idx = var_names.index("action")
    has_child = bool(np.any(graph[a_idx, :] != 0))
    if has_child:
        return graph, False
    for child in ACTION_FORCED_CHILDREN:
        if child in var_names:
            graph[a_idx, var_names.index(child)] = 1
    return graph, True


def diagnose_action_descendants(graph: np.ndarray, var_names: list) -> int:
    """Size of descendants(action) in the learned graph. Diagnostic
    intended to be logged per-asset to confirm the fix holds across the
    full ticker universe."""
    if "action" not in var_names:
        return 0
    a_idx = var_names.index("action")
    visited, queue = set(), [a_idx]
    while queue:
        cur = queue.pop(0)
        for j in range(len(var_names)):
            if graph[cur, j] != 0 and j not in visited:
                visited.add(j)
                queue.append(j)
    return len(visited)


def learn_causal_structure(features_df: pd.DataFrame, alpha: float = 0.05, method: str = "pc"):
    """
    Learn a causal graph from financial time-series features.

    Returns (causal_matrix, var_names, scaled_data_df).
    """
    dm = features_df[CAUSAL_VARS].copy()
    dm["action"] = np.sign(features_df["momentum_5"])
    dm["returns_lag1"] = features_df["returns"].shift(1)
    dm = dm.dropna()

    var_names = list(dm.columns)
    scaler = StandardScaler()
    dm_scaled = scaler.fit_transform(dm.values)
    dm_scaled_df = pd.DataFrame(dm_scaled, columns=var_names)

    if method == "notears" and _NOTEARS_OK:
        try:
            model = NotearsNonlinear()
            model.learn(dm_scaled)
            graph = model.causal_matrix
            graph, forced = _ensure_action_has_descendants(graph, var_names)
            n_edges = int(np.sum(graph != 0))
            n_desc = diagnose_action_descendants(graph, var_names)
            print(
                f"    [NOTEARS-MLP] {len(var_names)} nodes, {n_edges} edges  |  "
                f"descendants(action)={n_desc}" + ("  [forced]" if forced else "")
            )
            return graph, var_names, dm_scaled_df
        except Exception as e:
            print(f"    [NOTEARS failed -> PC fallback]: {e}")

    pc = PC(alpha=alpha)
    pc.learn(dm_scaled)
    graph = pc.causal_matrix
    graph, forced = _ensure_action_has_descendants(graph, var_names)
    n_desc = diagnose_action_descendants(graph, var_names)
    print(
        f"    [PC alpha={alpha}] {len(var_names)} nodes, {int(np.sum(graph != 0))} edges  |  "
        f"descendants(action)={n_desc}" + ("  [forced]" if forced else "")
    )
    return graph, var_names, dm_scaled_df
