"""
Structural Causal Model over the learned causal graph, linear or neural
structural equations, fit via Pearl's abduction-action-prediction do-calculus.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


class StructuralCausalModel:
    """
    Fits one structural equation per endogenous variable (its parents in the
    learned causal graph predict it; linear or a small MLP) and exposes
    Pearl's 3-step do-calculus for generating counterfactual states:

    1. Abduction:  infer exogenous noise U_i from the observed row
    2. Action:     apply do(intervention_var = intervention_value)
    3. Prediction: propagate through the graph in topological order
    """

    def __init__(self, causal_graph: np.ndarray, var_names: list, mode: str = "linear"):
        self.graph = causal_graph
        self.var_names = var_names
        self.n = len(var_names)
        self.mode = mode
        self.models = {}
        self._r2_scores = {}
        self._topo_order = None

    def fit(self, data_matrix: pd.DataFrame):
        self._topo_order = self._topological_sort()
        for i, var in enumerate(self.var_names):
            parent_idx = [j for j in range(self.n) if self.graph[j, i] != 0]
            if not parent_idx:
                self.models[var] = {
                    "type": "exogenous",
                    "mean": float(data_matrix.iloc[:, i].mean()),
                    "std": float(data_matrix.iloc[:, i].std()) + 1e-8,
                }
                self._r2_scores[var] = None
            else:
                parent_names = [self.var_names[p] for p in parent_idx]
                X = data_matrix[parent_names].values
                y = data_matrix[var].values
                reg = (
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1,
                    )
                    if self.mode == "neural"
                    else LinearRegression()
                )
                reg.fit(X, y)
                y_pred = reg.predict(X)
                residuals = y - y_pred
                self.models[var] = {
                    "type": "endogenous",
                    "parent_names": parent_names,
                    "reg": reg,
                    "residual_std": float(np.std(residuals)) + 1e-8,
                }
                self._r2_scores[var] = float(r2_score(y, y_pred))

    def fit_report(self) -> dict:
        endo = {k: v for k, v in self._r2_scores.items() if v is not None}
        if not endo:
            return {}
        return {
            "mean_r2": float(np.mean(list(endo.values()))),
            "min_r2": float(min(endo.values())),
            "per_var": endo,
        }

    def generate_counterfactual_state(
        self, observed_row: dict, intervention_var: str, intervention_value: float
    ) -> dict:
        # Step 1, Abduction: recover exogenous noise
        noise = {}
        for var in self.var_names:
            m = self.models.get(var)
            if m is None:
                noise[var] = 0.0
                continue
            if m["type"] == "exogenous":
                noise[var] = observed_row.get(var, m["mean"])
            else:
                pv = np.array([observed_row.get(p, 0.0) for p in m["parent_names"]]).reshape(1, -1)
                pred = float(m["reg"].predict(pv)[0])
                noise[var] = observed_row.get(var, pred) - pred

        # Step 2, Action: apply intervention
        cf = dict(observed_row)
        cf[intervention_var] = intervention_value

        # Step 3, Prediction: propagate forward in topological order,
        # restricted to descendants of the intervened variable
        int_idx = self.var_names.index(intervention_var)
        descendants = self._get_descendants(int_idx)
        for var in self._topo_order:
            if var == intervention_var:
                continue
            if self.var_names.index(var) not in descendants:
                continue
            m = self.models.get(var)
            if m and m["type"] == "endogenous":
                pv = np.array([cf.get(p, 0.0) for p in m["parent_names"]]).reshape(1, -1)
                pred = float(m["reg"].predict(pv)[0])
                cf[var] = pred + noise.get(var, 0.0)
        return cf

    def _topological_sort(self) -> list:
        in_deg = {v: 0 for v in self.var_names}
        for i in range(self.n):
            for j in range(self.n):
                if self.graph[i, j] != 0:
                    in_deg[self.var_names[j]] += 1
        queue = [v for v, d in in_deg.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            idx = self.var_names.index(node)
            for j in range(self.n):
                if self.graph[idx, j] != 0:
                    in_deg[self.var_names[j]] -= 1
                    if in_deg[self.var_names[j]] == 0:
                        queue.append(self.var_names[j])
        # Fall back gracefully if the graph has cycles (can happen with NOTEARS)
        return order if len(order) == self.n else list(self.var_names)

    def _get_descendants(self, node_idx: int) -> set:
        visited, queue = set(), [node_idx]
        while queue:
            cur = queue.pop(0)
            for j in range(self.n):
                if self.graph[cur, j] != 0 and j not in visited:
                    visited.add(j)
                    queue.append(j)
        return visited
