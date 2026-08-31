# CER-SCM: Counterfactual Experience Replay via Structural Causal Models

A causal reinforcement learning framework for financial trading. Integrates a
**Structural Causal Model (SCM)** — fit over a causal graph learned with PC /
NOTEARS-MLP — with **Counterfactual Experience Replay (CER)**: for every
transition an RL agent experiences, the SCM generates counterfactual
transitions under alternative actions via Pearl's do-calculus, giving the
agent additional (causally-grounded) training signal beyond what it actually
observed.

Evaluated on 30 assets across 5 sectors (US tech, US finance, Asian markets,
EU markets, commodities), comparing CER-augmented Dueling-DQN and discrete
SAC against Dueling-DQN, SAC, PPO, and buy-and-hold baselines, with an
ablation study, sector-clustered bootstrap tests, and bootstrap effect-size
confidence intervals.

## Method

```
Market data (12 features)
        │
        ▼
Causal structure learning (PC / NOTEARS-MLP)
        │
        ▼
Structural Causal Model               do-calculus: abduction → action → prediction
  (linear or neural structural
   equations per variable)
        │
        ▼
Counterfactual Experience Replay      for each real transition, generate N_CF_ACTIONS
  (CER)                               counterfactual transitions under alternative actions
        │
        ▼
Dueling-DQN / SAC  (+ CQL, PER)       trained on real + counterfactual experience
```

## Repo structure

```
cer-scm/
├── src/
│   ├── causal_structure.py   # PC/NOTEARS graph learning + degenerate-action fix
│   ├── scm.py                 # StructuralCausalModel — do-calculus counterfactual generation
│   └── metrics.py             # Trading performance metrics (Sharpe/Sortino/Calmar/etc.)
├── notebooks/
│   └── CER_SCM_experiment.ipynb   # Full 30-asset × N-trial experiment pipeline
├── requirements.txt
└── README.md
```

`src/` holds the causal-inference core — the reusable methodological
contribution. The notebook is the full experiment harness (data loading,
trading environment, DQN/SAC/CER agents, ablations, statistical tests,
publication figures) and is intentionally kept as one pipeline, matching how
the RTX 3090 experiment run is structured end-to-end; it uses the same fixed
logic as `src/`.

## Fixes applied (manuscript audit)

Two issues found during a two-pass audit against the paper and implementation
were still present in code despite being marked "patched," and are fixed here
at the source (verified against the original notebook's exact logic, with the
before/after difference checked numerically):

- **Sortino ratio.** Was computing standard deviation over only the
  negative-return subset (`np.std(neg)`), which is statistically invalid on
  small samples — a single large loss could produce a Sortino ratio in the
  millions. Fixed to the standard downside-deviation definition,
  `sqrt(mean(min(r, 0)²))`, over all returns.
- **Degenerate counterfactual mechanism.** The synthetic `action` variable
  used for causal structure learning was `sign(momentum_5)` — a deterministic
  function of a variable already in the graph. PC/NOTEARS assume
  faithfulness, which breaks under exact determinism, so `action` routinely
  learned zero outgoing edges; a `do(action = a')` intervention then had no
  downstream effect and the "counterfactual" state collapsed back to the
  observed state. Fixed by forcing domain-justified edges
  (`action → volume_ratio, returns_lag1`) whenever the automatic learner
  finds none, plus a `diagnose_action_descendants()` check to confirm the fix
  holds per-asset across the full ticker universe.

A third, cosmetic issue (a section header reading "20 Trials" against an
actual `N_TRIALS = 10` used consistently throughout the code) is also
corrected.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/cer-scm.git
cd cer-scm
pip install -r requirements.txt
```

Requires a CUDA GPU for the full 30-asset × 10-trial run in reasonable time
(notebook is tuned for an RTX 3090 — AMP, cudnn.benchmark, generous replay
buffer); CPU works for smaller-scale runs.

## Usage

Open `notebooks/CER_SCM_experiment.ipynb`. Checkpoints are saved per ticker
to `journal_checkpoints/`, so a run is safe to interrupt and resume.

As a library:
```python
from src import learn_causal_structure, StructuralCausalModel, compute_metrics

graph, var_names, data = learn_causal_structure(features_df, method="pc")
scm = StructuralCausalModel(graph, var_names, mode="linear")
scm.fit(data)

cf_state = scm.generate_counterfactual_state(observed_row, "action", alt_action_value)
```

## Status

Journal manuscript in revision. Framework validated across 30 assets / 5
sectors with statistical significance testing (Wilcoxon, Friedman, bootstrap
CIs, Bonferroni correction) and ablations isolating the contribution of CER,
CQL, and PER individually.

## License

MIT
