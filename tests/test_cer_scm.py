"""Unit tests for the counterfactual machinery. Run with:  pytest -q"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (  # noqa: E402
    StructuralCausalModel,
    CounterfactualGenerator,
    UniformReplayBuffer,
    mixed_batch,
    variance_penalized_target,
    diagnose_action_descendants,
)

VARS = ["momentum_5", "action", "volume_ratio", "returns"]


def _linear_world(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1, n)
    a = rng.choice([-1.0, 0.0, 1.0], n)
    v = 0.8 * a + rng.normal(0, 0.5, n)
    r = 0.5 * v + 0.3 * m + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"momentum_5": m, "action": a, "volume_ratio": v, "returns": r})
    g = np.zeros((4, 4))
    g[0, 3] = g[2, 3] = g[1, 2] = 1  # momentum->returns, volume->returns, action->volume
    return df, g


@pytest.fixture(scope="module")
def fitted():
    df, g = _linear_world()
    scm = StructuralCausalModel(g, VARS, mode="linear")
    scm.fit(df)
    return df, g, scm


def test_consistency_observed_action_returns_observed_day(fitted):
    """Intervening with the action actually taken reproduces the observation."""
    df, _, scm = fitted
    obs = df.iloc[5].to_dict()
    cf = scm.generate_counterfactual_state(obs, "action", obs["action"])
    for v in VARS:
        assert cf[v] == pytest.approx(obs[v], abs=1e-9)


def test_recovers_true_counterfactual(fitted):
    """With the true graph, abduction-action-prediction matches the ground truth."""
    df, _, scm = fitted
    errs = []
    for _, row in df.head(300).iterrows():
        obs = row.to_dict()
        for alt in (-1.0, 0.0, 1.0):
            cf = scm.generate_counterfactual_state(obs, "action", alt)
            truth = obs["returns"] + 0.4 * (alt - obs["action"])
            errs.append(abs(cf["returns"] - truth))
    assert np.mean(errs) < 0.02


def test_non_descendants_are_unchanged(fitted):
    """An intervention can only change descendants of the action (Section 3.2)."""
    df, _, scm = fitted
    obs = df.iloc[7].to_dict()
    cf = scm.generate_counterfactual_state(obs, "action", 1.0 if obs["action"] != 1.0 else -1.0)
    assert cf["momentum_5"] == obs["momentum_5"]


def test_no_descendants_means_relabelling():
    """If 'action' has no descendants, the counterfactual equals the observed day (Section 7.2)."""
    df, g = _linear_world()
    g = g.copy()
    g[1, 2] = 0  # remove action -> volume
    assert diagnose_action_descendants(g, VARS) == 0
    scm = StructuralCausalModel(g, VARS, mode="linear")
    scm.fit(df)
    obs = df.iloc[3].to_dict()
    cf = scm.generate_counterfactual_state(obs, "action", -obs["action"] or 1.0)
    assert cf["returns"] == pytest.approx(obs["returns"])
    assert cf["volume_ratio"] == pytest.approx(obs["volume_ratio"])


def test_generator_produces_k2_scaled_transitions(fitted):
    df, g, scm = fitted
    state_cols = ["returns", "volume_ratio", "momentum_5"]
    gen = CounterfactualGenerator(scm, VARS, state_cols, beta=0.9, k=2)
    row = df.iloc[11]
    state = row[state_cols].values
    action = {-1.0: 0, 0.0: 1, 1.0: 2}[row["action"]]
    out = gen.generate(state, action, 0.01, state, False)
    assert len(out) == 2
    assert {t[1] for t in out} == set(range(3)) - {action}
    for s_cf, a_alt, r_cf, s_next, d in out:
        obs = row.to_dict()
        cf = scm.generate_counterfactual_state(obs, "action", {0: -1.0, 1: 0.0, 2: 1.0}[a_alt])
        assert r_cf == pytest.approx(0.9 * cf["returns"])


def test_mixed_batch_is_half_real_half_counterfactual():
    real, cf = UniformReplayBuffer(), UniformReplayBuffer()
    for i in range(300):
        real.push("real", i)
        cf.push("cf", i)
    batch = mixed_batch(real, cf, 256)
    assert sum(t[0] == "real" for t in batch) == 128
    assert sum(t[0] == "cf" for t in batch) == 128


def test_variance_penalized_target_matches_eq8():
    r = np.array([0.01, -0.02])
    q_online = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])
    q_target = np.array([[1.1, 1.9, 0.4], [0.2, -0.8, 2.5]])
    done = np.array([0.0, 1.0])
    y = variance_penalized_target(r, q_online, q_target, done, gamma=0.99, alpha_cql=0.01)
    expected0 = 0.01 + 0.99 * (1.9 - 0.01 * np.std([1.1, 1.9, 0.4], ddof=1))
    assert y[0] == pytest.approx(expected0)
    assert y[1] == pytest.approx(-0.02)  # terminal transition
