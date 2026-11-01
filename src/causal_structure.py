"""
Causal structure learning over financial time-series features (PC / NOTEARS-MLP).

Matches Section 4.2 of the paper. The 12 causal variables are ten state
features, an action proxy (the sign of 5-day momentum) and the previous day's
return. All variables are standardized before discovery.

`diagnose_action_descendants` reports how many variables an intervention on
the action can reach. When it is zero, a counterfactual leaves the state
unchanged (Section 7.2 of the paper). `force_action_edges` is an optional
extension, off by default, that adds two domain-motivated edges in that case.
It was not used for any reported result.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CAUSAL_VARS = [
    "returns", "volatility", "volume_ratio",
    "momentum_5", "rsi", "bb_position", "macd_diff",
    "atr_ratio", "obv_signal", "price_vs_sma50",
]

# Children added to 'action' when force_action_edges=True and the learner finds none.
ACTION_FORCED_CHILDREN = ["volume_ratio", "returns_lag1"]


def diagnose_action_descendants(graph: np.ndarray, var_names: list) -> int:
    """Number of variables downstream of 'action' in the graph.

    Zero means do(action = a') changes nothing, so the counterfactual state
    equals the observed state.
    """
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
    visited.discard(a_idx)
    return len(visited)


def _force_action_descendants(graph: np.ndarray, var_names: list):
    """Optional extension: add action -> {volume_ratio, returns_lag1} when
    'action' has no children. Returns (graph, was_forced)."""
    if "action" not in var_names:
        return graph, False
    a_idx = var_names.index("action")
    if np.any(graph[a_idx, :] != 0):
        return graph, False
    graph = graph.copy()
    for child in ACTION_FORCED_CHILDREN:
        if child in var_names:
            graph[a_idx, var_names.index(child)] = 1
    return graph, True


def build_causal_dataset(features_df: pd.DataFrame) -> pd.DataFrame:
    """The 12 causal variables: ten features, the action proxy and returns_lag1."""
    dm = features_df[CAUSAL_VARS].copy()
    dm["action"] = np.sign(features_df["momentum_5"])
    dm["returns_lag1"] = features_df["returns"].shift(1)
    return dm.dropna()


def learn_causal_structure(
    features_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "pc",
    force_action_edges: bool = False,
    verbose: bool = True,
):
    """Learn a causal graph over the 12 causal variables.

    method: "pc" (Fisher's Z at level alpha; undirected CPDAG edges are kept in
            both directions) or "notears" (NOTEARS-MLP, falls back to PC).
    Returns (causal_matrix, var_names, standardized_data_df), where
    causal_matrix[i, j] != 0 means an edge i -> j.
    """
    from castle.algorithms import PC  # imported lazily: only needed for discovery

    dm = build_causal_dataset(features_df)
    var_names = list(dm.columns)
    dm_scaled = StandardScaler().fit_transform(dm.values)
    dm_scaled_df = pd.DataFrame(dm_scaled, columns=var_names)

    graph, label = None, None
    if method == "notears":
        try:
            from castle.algorithms import NotearsNonlinear

            model = NotearsNonlinear()
            model.learn(dm_scaled)
            graph, label = model.causal_matrix, "NOTEARS-MLP"
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"    [NOTEARS failed, falling back to PC]: {e}")
    if graph is None:
        pc = PC(alpha=alpha)
        pc.learn(dm_scaled)
        graph, label = pc.causal_matrix, f"PC alpha={alpha}"

    graph = np.asarray(graph)
    forced = False
    if force_action_edges:
        graph, forced = _force_action_descendants(graph, var_names)
    if verbose:
        n_desc = diagnose_action_descendants(graph, var_names)
        print(
            f"    [{label}] {len(var_names)} nodes, {int(np.sum(graph != 0))} edges | "
            f"descendants(action)={n_desc}" + (" [forced]" if forced else "")
        )
    return graph, var_names, dm_scaled_df
