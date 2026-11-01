"""
Synthetic check of the counterfactual machinery, runs on a CPU in seconds.

We simulate markets whose true structural causal model is known, so the true
counterfactual of every observed day can be computed exactly and compared
with what CER-SCM produces.

Case A. The decision affects the market (action -> volume -> returns).
        With the true graph, abduction-action-prediction recovers the true
        counterfactual return. With a graph learned by PC, it does too,
        whenever PC finds the action's descendants.

Case B. The decision is a proxy that the market does not respond to, as with
        sign(momentum_5) on historical data (Section 4.2 of the paper).
        - With the true graph the action has no descendants, so every
          counterfactual equals the observed day: the transition is only
          relabelled, as for AAPL in Section 7.2 of the paper.
        - PC cannot orient the momentum-action edge, and CER-SCM keeps
          undirected edges in both directions. The action then acquires
          descendants it does not really have, and the counterfactuals
          differ from the observed day but are wrong.

Both failure modes are invisible to SCM fit statistics, which is why the
paper's open question is when counterfactual experience carries real
causal information.

Usage:  python examples/synthetic_counterfactual_demo.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import StructuralCausalModel, diagnose_action_descendants  # noqa: E402

VARS = ["momentum_5", "action", "volume_ratio", "returns"]
N = 3000


def simulate(case: str, rng):
    m = rng.normal(0, 1, N)
    if case == "A":
        a = rng.choice([-1.0, 0.0, 1.0], N)   # decisions vary independently of the market
        v = 0.8 * a + rng.normal(0, 0.5, N)   # trading moves volume
    else:
        a = np.sign(m)                        # trend-following proxy, as in the paper
        v = rng.normal(0, 0.5, N)             # the market ignores the proxy
    r = 0.5 * v + 0.3 * m + rng.normal(0, 0.5, N)
    return pd.DataFrame({"momentum_5": m, "action": a, "volume_ratio": v, "returns": r})


def true_cf_return(row, alt, case):
    if case == "A":
        return row["returns"] + 0.5 * 0.8 * (alt - row["action"])
    return row["returns"]


def true_graph(case):
    g = np.zeros((4, 4))
    idx = {v: i for i, v in enumerate(VARS)}
    g[idx["momentum_5"], idx["returns"]] = 1
    g[idx["volume_ratio"], idx["returns"]] = 1
    if case == "A":
        g[idx["action"], idx["volume_ratio"]] = 1
    else:
        g[idx["momentum_5"], idx["action"]] = 1
    return g


def pc_graph(df):
    import contextlib
    import io
    import logging

    logging.disable(logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()):
        from castle.algorithms import PC

        z = (df - df.mean()) / df.std()
        pc = PC(alpha=0.05)
        pc.learn(z.values)
    return np.asarray(pc.causal_matrix)


def cf_error(df, graph, case, n_eval=500):
    scm = StructuralCausalModel(graph, VARS, mode="linear")
    scm.fit(df)
    errs, moved = [], 0
    for _, row in df.head(n_eval).iterrows():
        obs = row.to_dict()
        for alt in (-1.0, 0.0, 1.0):
            if alt == obs["action"]:
                continue
            cf = scm.generate_counterfactual_state(obs, "action", alt)
            errs.append(abs(cf["returns"] - true_cf_return(obs, alt, case)))
            moved += abs(cf["returns"] - obs["returns"]) > 1e-9
    return float(np.mean(errs)), moved / len(errs)


def main():
    rng = np.random.default_rng(0)
    print("Synthetic counterfactual check (linear SCM, n = 3000 days)\n")
    for case, title in [("A", "Case A: the decision moves the market"),
                        ("B", "Case B: the decision is a proxy the market ignores")]:
        df = simulate(case, rng)
        print(title)
        for name, g in [("true graph", true_graph(case)), ("PC-learned graph", pc_graph(df))]:
            mae, frac = cf_error(df, g, case)
            print(f"  {name:17s} descendants(action) = {diagnose_action_descendants(g, VARS)} | "
                  f"MAE vs true counterfactual return = {mae:.4f} | "
                  f"counterfactuals that differ from the observed day = {frac:.0%}")
        print()
    print("Case A: when the decision really moves the market, the learned SCM recovers the")
    print("true counterfactual returns. Case B: when it does not, the true graph gives pure")
    print("relabelling (no descendants), while the PC graph, with its unoriented")
    print("momentum-action edge kept both ways, gives the action spurious descendants and")
    print("produces counterfactuals that differ from the observed day but are wrong.")


if __name__ == "__main__":
    main()
