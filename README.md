# CER-SCM: Counterfactual Experience Replay via Structural Causal Models

Code for the manuscript

> **CER-SCM: Counterfactual Experience Replay via Structural Causal Models for Off-Policy Financial Reinforcement Learning**
> Amruthamsha Prashanth Raju, R. Viswajit, K. Sudhakar, Archana Mathur, Lakshmana
> Nitte Meenakshi Institute of Technology, Bengaluru. *Under review at IEEE Transactions on Artificial Intelligence.*

An off-policy trading agent only ever sees the outcome of the action it took. CER-SCM gives it the outcomes of the actions it did not take. It learns a causal graph for each asset over 12 causal variables (PC or NOTEARS-MLP), fits a structural causal model (SCM) on that graph, and uses Pearl's abduction-action-prediction procedure to generate counterfactual transitions for every untaken action. These are replayed alongside real experience, so the agent learns from both observed and generated samples.

## Key results

Out of sample over 2023-2024, net of transaction costs, on 24 equities across 4 global sectors with 10 trials per method:

- **CausalSAC raised mean total return from 24.6% to 35.0%**, a **42.1% relative gain** over Discrete SAC (Bonferroni-corrected Wilcoxon p = 0.044), and **raised the Sharpe ratio by 39.5%**, from 0.524 to 0.731 (asset-level p = 0.003).
- **The gain is robust.** It holds under a sector-clustered bootstrap (+10.38 percentage points, 95% CI [+2.46, +24.52]) and replicates on an independent 30-asset run that includes commodities (22 of 30 assets).
- **Recent-window SCMs matter.** Refitting the causal model on a recent window raised CausalDQN's return from 22.5% to 33.3% (+48%) and its Sharpe from 0.446 to 0.625, improving 21 of 24 assets (Wilcoxon p = 0.0032). A graph frozen on 2018-2021 describes a market that no longer holds.
- **Structure matters.** Replacing SCM counterfactuals with random ones gives the lowest return of any RL method (17.4%), consistent with the gain coming from causal structure rather than from extra replay data.

## Method

```
Market data (12 features per asset)
        |
        v
Causal structure learning (PC / NOTEARS-MLP)
        |
        v
Structural causal model           abduction -> action -> prediction
(linear or neural equations)
        |
        v
Counterfactual Experience Replay  for each real transition, generate transitions
                                  under the alternative actions
        |
        v
Dueling DQN (+ PER, variance-penalized CQL target) or Discrete SAC,
trained on real and counterfactual experience
```

## Repository contents

```
cer-scm/
├── notebooks/
│   ├── CER_SCM_main_24_assets.ipynb          # main run: code and outputs behind the reported results
│   └── CER_SCM_replication_30_assets.ipynb   # independent 30-asset replication run
├── src/                                      # reusable core, with post-analysis fixes (see below)
│   ├── causal_structure.py                   # PC / NOTEARS-MLP structure learning
│   ├── scm.py                                # StructuralCausalModel and counterfactual generation
│   └── metrics.py                            # trading metrics
├── requirements.txt
├── LICENSE
└── README.md
```

The notebooks are kept **exactly as run**, so their code and recorded outputs match each other. `src/` contains the same core logic, refactored into modules and including two fixes made after the reported runs. The two are deliberately kept separate. See [Known issues and ongoing work](#known-issues-and-ongoing-work).

## Setup

```bash
git clone https://github.com/Amruthamsha/cer-scm.git
cd cer-scm
pip install -r requirements.txt
```

Data is downloaded from Yahoo Finance via `yfinance` on first run. A CUDA GPU is needed for the full experiments (the notebooks are tuned for an RTX 3090). Checkpoints are saved per ticker, so runs can be interrupted and resumed.

Using the core modules:

```python
from src import learn_causal_structure, StructuralCausalModel, compute_metrics

graph, var_names, data = learn_causal_structure(features_df, method="pc")
scm = StructuralCausalModel(graph, var_names, mode="linear")
scm.fit(data)
cf_state = scm.generate_counterfactual_state(observed_row, "action", alt_action_value)
```

## Experimental setup

- **Assets.** Configured for 30 tickers in 5 sectors. In the main run, all 6 commodity tickers failed to download, so the reported results cover **24 equities across 4 sectors** (US tech, US finance, Asian markets, EU markets).
- **Periods.** Train 2018-2021, validation 2022, out-of-sample test 2023-2024, net of transaction costs.
- **Agents.** CausalDQN (Dueling DQN + PER with a variance-penalized CQL target) and CausalSAC (Discrete SAC), each compared against the same agent without counterfactual replay.
- **Trials.** 10 per method, 100 episodes each, with validation-based early stopping.
- **Statistics.** Wilcoxon signed-rank tests with Bonferroni correction, Friedman tests, sector-clustered bootstrap confidence intervals, and ablations isolating CER, CQL and PER.

## Full results (main run, 24 assets, out of sample 2023-2024)

Mean total return and Sharpe ratio across assets, from `notebooks/CER_SCM_main_24_assets.ipynb`:

| Method | Mean total return | Sharpe |
|---|---|---|
| **CausalSAC** (CER + neural SCM + Discrete SAC) | **35.01% ± 52.30** | **0.731** |
| Discrete SAC (baseline) | 24.64% ± 40.57 | 0.524 |
| CausalDQN, recent-window SCM | 33.29% ± 58.70 | 0.625 |
| CausalDQN, fixed SCM | 22.48% ± 46.87 | 0.446 |
| Dueling DQN (baseline) | 30.46% ± 50.51 | 0.564 |
| Random counterfactuals (ablation) | 17.36% ± 30.25 | 0.370 |

**Replication (30 assets, including commodities):** CausalSAC 35.64% against Discrete SAC 20.90%, from `notebooks/CER_SCM_replication_30_assets.ipynb`.

**Scope.** The claim is that counterfactual replay improves an off-policy learner relative to the same learner without it. Passive buy and hold returned 91.4% over 2023-2024, a strong bull market for these assets, and outperforms every learned policy in that window. The notebook also contains a PPO run, which is on-policy and outside the scope of an experience-replay method, so it is not part of the reported comparison.

## Known issues and ongoing work

Both issues below were identified after the reported runs. The notebooks are left unchanged so that their code matches their outputs. The fixes live in `src/`, and a re-run with them is pending.

1. **Sortino ratio.** The notebooks compute downside deviation as the standard deviation of only the negative returns. On small samples this is unstable and produces meaningless values (in the thousands) in the recorded outputs, so **ignore the Sortino columns in the notebook outputs**. `src/metrics.py` uses the standard definition, `sqrt(mean(min(r, 0)^2))` over all returns.

2. **Action nodes without descendants.** For structure learning, the action variable is a deterministic function of `momentum_5`. This violates the faithfulness assumption behind PC and NOTEARS, and in several learned graphs the action node received no outgoing edges. For those assets, an intervention on the action changes nothing downstream, so the generated "counterfactuals" simply relabel the observed transition. `src/causal_structure.py` adds `diagnose_action_descendants()` to detect this, and a stopgap that adds two domain-motivated edges (action to next-period volume and to realized return) when none are found. This is a stopgap, not a solution. When counterfactual experience carries real causal information, and how errors in a learned causal model propagate into the policy, are open questions and the focus of my ongoing research.

## Citation

```bibtex
@unpublished{raju_cerscm,
  author = {Raju, Amruthamsha Prashanth and Viswajit, R. and Sudhakar, K. and Mathur, Archana and Lakshmana},
  title  = {{CER-SCM}: Counterfactual Experience Replay via Structural Causal Models for Off-Policy Financial Reinforcement Learning},
  note   = {Under review at IEEE Transactions on Artificial Intelligence},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.
