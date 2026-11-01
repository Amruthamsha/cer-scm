"""Risk-adjusted trading performance metrics."""

import numpy as np
import pandas as pd


def compute_metrics(history: list, initial_balance: float, final_pv: float) -> dict:
    """
    Compute total return, Sharpe, Sortino, Calmar, max drawdown, win rate,
    and turnover from a list of {'step', 'action', 'pv', 'reward'} records.

    Sharpe is sqrt(252) * mean / std of daily rewards (Section 5.4 of the paper).
    The Sortino ratio uses the standard downside deviation over all returns.
    The paper does not report Sortino, because with few losing days the ratio
    becomes unstable; the notebooks computed it with an earlier formula.
    """
    if not history:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "final_value": 0.0,
        }

    h = pd.DataFrame(history)
    rets = h["reward"].values
    pv = h["pv"].values

    tr = (final_pv - initial_balance) / initial_balance

    sh = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)

    # Post-analysis fix: downside deviation computed over ALL returns (positive
    # returns contribute zero downside), not std() over only the negative
    # subset. The latter is statistically invalid on small/degenerate
    # negative-return samples, e.g. a single negative return gives std=0,
    # producing a spuriously huge Sortino ratio.
    downside_sq = np.minimum(rets, 0.0) ** 2
    downside_deviation = np.sqrt(np.mean(downside_sq))
    so = (np.mean(rets) / (downside_deviation + 1e-9)) * np.sqrt(252)

    peak = np.maximum.accumulate(pv)
    mdd = float(np.min((pv - peak) / (peak + 1e-9)))

    ann = tr * 252 / max(len(rets), 1)
    cal = ann / (abs(mdd) + 1e-9)

    return {
        "total_return": tr,
        "sharpe_ratio": sh,
        "sortino_ratio": so,
        "calmar_ratio": cal,
        "max_drawdown": mdd,
        "win_rate": float(np.mean(rets > 0)),
        "turnover": float(np.mean(h["action"] != 1)),
        "final_value": final_pv,
    }
