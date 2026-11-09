"""
What do CER-SCM's counterfactual rewards actually contain?

CPU only, no RL training; roughly a minute per asset (NOTEARS dominates).
For every asset and both SCMs (linear-PC, neural-NOTEARS) it walks the
training window with a random policy and, at every step, compares the
counterfactual reward CER-SCM stores for each untaken action against the
true one-step reward that action would have earned (read from the
environment, fees included).

It runs the notebook's own CERMixin.generate_and_store_counterfactuals, so
it measures exactly what the agents were trained on.

    python diagnose_cf.py                 # all 24 equities
    python diagnose_cf.py --tickers AAPL,JPM

Output: cf_diagnostics.csv (one row per asset x SCM) and a printed summary.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nbdefs import load_notebook_defs  # noqa: E402
from cf_controls import peek_reward  # noqa: E402
from run_controls import DEFAULT_NB, load_or_cache_data  # noqa: E402


class _Sink:
    def __init__(self):
        self.items = []

    def push(self, *t):
        self.items.append(t)

    def __len__(self):
        return len(self.items)


def make_probe(nb, scm, vn):
    """The notebook's CERMixin, with a sink in place of the replay buffer,
    so we see exactly the counterfactual transitions the agents stored."""

    class Probe(nb.CERMixin):
        def __init__(self):
            self.init_cer(scm, vn)
            self.cf_memory = _Sink()

    return Probe()


def diagnose(nb, scm, vn, train_df, rng):
    probe = make_probe(nb, scm, vn)
    act_idx = vn.index("action")
    desc = {vn[j] for j in scm._get_descendants(act_idx)}
    ret_idx = list(nb.STATE_COLS).index("returns")
    feat_sd = train_df[nb.STATE_COLS].std().values + 1e-12

    env = nb.TradingEnvironment(train_df)
    state = env.reset()
    done = False
    rows = []
    while not done:
        action = int(rng.integers(3))
        alt_true = {a: peek_reward(env, a) for a in range(3)}
        ns, reward, done = env.step(action)
        probe.cf_memory.items = []
        probe.generate_and_store_counterfactuals(state, action, reward, ns, done)
        for cf_s, alt_a, cf_r, _, _ in probe.cf_memory.items:
            rows.append({
                "taken": action,
                "alt": alt_a,
                "cf_r": float(cf_r),
                "true_r": float(alt_true[alt_a]),
                "real_r": float(reward),
                "lagged_ret_x_beta": float(nb.CF_BETA * state[ret_idx]),
                "state_shift_sd": float(np.max(np.abs((np.asarray(cf_s) - state) / feat_sd))),
            })
        state = ns

    d = pd.DataFrame(rows)
    # the two alternatives for the same real step are consecutive pairs
    pair = d["cf_r"].values.reshape(-1, 2)
    informative = d["true_r"].abs() > 1e-9  # skip steps where the alternative changes nothing
    sub = d[informative]
    corr = float(np.corrcoef(sub["cf_r"], sub["true_r"])[0, 1]) if len(sub) > 10 and sub["cf_r"].std() > 0 else np.nan
    sign = float(np.mean(np.sign(sub["cf_r"]) == np.sign(sub["true_r"]))) if len(sub) else np.nan
    return {
        "action_has_descendants": len(desc) > 0,
        "returns_is_descendant": "returns" in desc,
        "n_descendants": len(desc),
        "pct_cf_equals_beta_lagged_return": float(np.mean(np.isclose(d["cf_r"], d["lagged_ret_x_beta"], atol=1e-7)) * 100),
        "pct_both_alternatives_same_cf_r": float(np.mean(np.isclose(pair[:, 0], pair[:, 1], atol=1e-9)) * 100),
        "corr_cf_r_vs_true_r": corr,
        "sign_agreement_cf_vs_true": sign,
        "median_abs_cf_r": float(d["cf_r"].abs().median()),
        "median_abs_real_r": float(d["real_r"].abs().median()),
        "pct_abs_cf_r_over_5pct": float((d["cf_r"].abs() > 0.05).mean() * 100),
        "max_abs_cf_r": float(d["cf_r"].abs().max()),
        "median_state_shift_sd": float(d["state_shift_sd"].median()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notebook", default=DEFAULT_NB)
    p.add_argument("--tickers", default=None)
    p.add_argument("--data-cache", default="data_cache")
    p.add_argument("--out", default="cf_diagnostics.csv")
    args = p.parse_args()

    nb = load_notebook_defs(args.notebook)
    tickers = ([t.strip() for t in args.tickers.split(",")] if args.tickers else
               [t for s, lst in nb.ASSET_UNIVERSE.items() if s != "Commodities" for t in lst])

    out = []
    for t in tickers:
        data = load_or_cache_data(nb, t, args.data_cache)
        if data is None:
            print(f"{t}: data unavailable")
            continue
        train = data["train"]
        g_pc, vn, dm = nb.learn_causal_structure(train, method="pc")
        g_nt, _, _ = nb.learn_causal_structure(train, method="notears")
        for label, g, mode in (("linear_pc", g_pc, "linear"), ("neural_notears", g_nt, "neural")):
            scm = nb.StructuralCausalModel(g, vn, mode=mode)
            scm.fit(dm)
            r = diagnose(nb, scm, vn, train, np.random.default_rng(0))
            r.update({"ticker": t, "scm": label})
            out.append(r)
            print(f"{t:<10} {label:<15} desc={r['n_descendants']:2d} ret_desc={r['returns_is_descendant']!s:<5} "
                  f"=beta*lagged {r['pct_cf_equals_beta_lagged_return']:5.1f}%  "
                  f"corr(cf,true)={r['corr_cf_r_vs_true_r']:+.3f}  "
                  f"|cf| med {r['median_abs_cf_r']:.4f} vs real {r['median_abs_real_r']:.4f}  "
                  f">5%: {r['pct_abs_cf_r_over_5pct']:.1f}%")

    df = pd.DataFrame(out)
    cols = ["ticker", "scm"] + [c for c in df.columns if c not in ("ticker", "scm")]
    df[cols].to_csv(args.out, index=False)
    print(f"\nSaved {args.out}\n")
    print(df.groupby("scm")[["action_has_descendants", "returns_is_descendant",
                              "pct_cf_equals_beta_lagged_return", "pct_both_alternatives_same_cf_r",
                              "corr_cf_r_vs_true_r", "sign_agreement_cf_vs_true",
                              "pct_abs_cf_r_over_5pct"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
