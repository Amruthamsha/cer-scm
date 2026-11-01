"""
Counterfactual Experience Replay (Sections 4.4 and 4.5 of the paper).

Framework-agnostic reference implementation of the pieces that make up
CER-SCM, written to match the experiment notebooks line for line:

* `CounterfactualGenerator`  turns one real transition into K = 2
  counterfactual transitions (one per untaken action) via abduction, action
  and prediction on a fitted StructuralCausalModel (Eqs. 5-7).
* `UniformReplayBuffer` and `mixed_batch` give the separate counterfactual
  buffer and the 50/50 real/counterfactual training batch.
* `variance_penalized_target` is the Variance-Penalized CQL target used by
  CausalDQN (Eq. 8).
"""

from collections import deque
import random

import numpy as np

# Action encoding (Eq. 1 and Section 4.3): 0 = Sell, 1 = Hold, 2 = Buy.
ACTION_TO_SCM = {0: -1.0, 1: 0.0, 2: 1.0}
N_ACTIONS = 3


class CounterfactualGenerator:
    """Generate counterfactual transitions for every untaken action.

    Parameters
    ----------
    scm : StructuralCausalModel, already fitted.
    var_names : the SCM's variable names (12 causal variables).
    state_cols : names of the state features, in state-vector order.
    beta : trust factor on counterfactual rewards (Eq. 7), default 0.9.
    k : counterfactuals per transition, default 2 (all untaken actions).
    """

    def __init__(self, scm, var_names, state_cols, beta: float = 0.9, k: int = 2):
        self.scm = scm
        self.var_names = list(var_names)
        self.state_var_map = {v: i for i, v in enumerate(state_cols) if v in self.var_names}
        self.beta = beta
        self.k = k

    def _state_to_obs(self, state) -> dict:
        obs = {v: float(state[i]) for v, i in self.state_var_map.items()}
        for v in self.var_names:
            obs.setdefault(v, 0.0)
        return obs

    def _obs_to_state(self, obs: dict, original_state):
        cf = np.array(original_state, dtype=float).copy()
        for v, i in self.state_var_map.items():
            if v in obs:
                cf[i] = float(np.nan_to_num(obs[v], nan=0.0, posinf=1.0, neginf=-1.0))
        return cf

    def generate(self, state, action: int, reward: float, next_state, done):
        """Return a list of (s_cf, a_alt, r_cf, next_state, done) tuples.

        Abduction recovers each variable's noise from the observed day, the
        action node is set to the alternative, and only its descendants are
        recomputed. The counterfactual reward is beta times the SCM's
        counterfactual return; the next state is the observed one.
        """
        obs = self._state_to_obs(state)
        if "action" in self.var_names:
            obs["action"] = ACTION_TO_SCM[action]
        out = []
        alternatives = [a for a in range(N_ACTIONS) if a != action][: self.k]
        for alt in alternatives:
            cf_obs = self.scm.generate_counterfactual_state(obs, "action", ACTION_TO_SCM[alt])
            cf_state = self._obs_to_state(cf_obs, state)
            cf_r_raw = cf_obs.get("returns", reward)
            cf_r = float(np.nan_to_num(cf_r_raw, nan=0.0, posinf=0.1, neginf=-0.1)) * self.beta
            out.append((cf_state, alt, cf_r, next_state, done))
        return out


class UniformReplayBuffer:
    """Plain FIFO buffer with uniform sampling (used for B_CF)."""

    def __init__(self, capacity: int = 200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, n: int):
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


def mixed_batch(real_buffer, cf_buffer, batch_size: int = 256):
    """Half real, half counterfactual (128 + 128 in the paper).

    Real transitions come from `real_buffer.sample` (uniform or prioritized);
    counterfactuals are drawn uniformly from their own buffer, so model error
    cannot capture the priority meant for real experience. Falls back to the
    real half alone until the counterfactual buffer is large enough.
    """
    real_n = batch_size // 2
    cf_n = batch_size - real_n
    real = real_buffer.sample(real_n)
    if len(cf_buffer) >= cf_n:
        return list(real) + cf_buffer.sample(cf_n)
    return list(real)


def variance_penalized_target(r, q_next_online, q_next_target, done, gamma=0.99, alpha_cql=0.01):
    """Variance-Penalized CQL target (Eq. 8), for arrays of shape (batch, n_actions).

        a*    = argmax_a Q_online(s', a)                       (Double DQN selection)
        y     = r + gamma * (1 - done) * [ Q_target(s', a*) - alpha * std_a Q_target(s', a) ]

    The standard deviation is the sample standard deviation across actions,
    matching torch.Tensor.std(dim=1) in the notebooks.
    """
    r = np.asarray(r, dtype=float)
    done = np.asarray(done, dtype=float)
    q_next_online = np.asarray(q_next_online, dtype=float)
    q_next_target = np.asarray(q_next_target, dtype=float)
    a_star = q_next_online.argmax(axis=1)
    q_eval = q_next_target[np.arange(len(a_star)), a_star]
    spread = q_next_target.std(axis=1, ddof=1)
    return r + gamma * (q_eval - alpha_cql * spread) * (1.0 - done)
