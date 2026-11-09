"""
SAC-backbone controls for CER-SCM.

Each control is CausalSAC with ONLY the counterfactual generator swapped.
Everything else is inherited unchanged from the notebook: the Discrete SAC
networks and update, the separate uniform counterfactual buffer, the 128 + 128
mixed batch (CausalSACAgent.train_step), K = 2 counterfactuals per transition,
the training loop, early stopping on validation Sharpe and the test evaluation.

    random_cf_sac   Gaussian noise (sigma = 0.05) on the observed state and
                    reward, labelled with the untaken action. Mirrors the
                    notebook's RandomCFDQNAgent exactly.
    relabel_sac     No SCM at all: (s, a', beta * returns_t, s'). This is what
                    CER-SCM produces whenever the action node has no
                    descendants in the learned graph (Section 7.2).
    exact_cf_sac    The true one-step counterfactual: the portfolio return the
                    environment would have paid for a' from the same cash /
                    share position, fees included. No model error, so beta = 1.

Optional reference reruns (same cached data, same code path, same seeds as
the controls, so all five SAC arms are directly comparable):

    causal_sac_rerun     CausalSACAgent with the neural SCM (the paper's CausalSAC)
    discrete_sac_rerun   DiscreteSACAgent (the paper's Discrete SAC)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exact one-step counterfactual reward, read off the environment without
# mutating it. Mirrors TradingEnvironment.step line for line.
# ---------------------------------------------------------------------------
def peek_reward(env, action):
    if env.step_idx >= len(env.data) - 1:
        return 0.0
    price = float(env.data[env.step_idx][-1])
    next_price = float(env.data[env.step_idx + 1][-1])
    balance, shares = env.balance, env.shares
    if action == 2 and balance > 0:
        shares += balance / (price * (1 + env.tc))
        balance = 0.0
    elif action == 0 and shares > 0:
        balance += shares * price * (1 - env.tc)
        shares = 0.0
    pv = balance + shares * next_price
    return (pv - env.pv) / (env.pv + 1e-9)


def build_control_classes(nb):
    """Create the control agent class against the notebook's own classes."""

    class CFControlSACAgent(nb.CausalSACAgent):
        """CausalSAC with a non-SCM counterfactual generator.

        train_step is inherited from CausalSACAgent untouched, so the only
        difference from CausalSAC is what goes into self.cf_memory.
        """

        MODES = ("random", "relabel", "exact")

        def __init__(self, mode, noise_scale=0.05, beta=None, exact_beta=1.0, **kwargs):
            if mode not in self.MODES:
                raise ValueError(mode)
            # Skip CERMixin.init_cer (needs an SCM); set up only what train_step uses.
            nb.DiscreteSACAgent.__init__(self, **kwargs)
            self.cf_memory = nb.UniformReplayBuffer(nb.BUFFER_CAP)
            self.mode = mode
            self.noise_scale = noise_scale
            self.beta = nb.CF_BETA if beta is None else beta
            self.exact_beta = exact_beta
            self._ret_idx = list(nb.STATE_COLS).index("returns")
            self.needs_env_peek = (mode == "exact")
            self._alt_rewards = None

        def generate_and_store_counterfactuals(self, state, action, reward, next_state, done):
            alts = [a for a in range(3) if a != action][: nb.N_CF_ACTIONS]
            for alt_a in alts:
                if self.mode == "random":
                    cf_s = (state + np.random.normal(0, self.noise_scale, state.shape)).astype(np.float32)
                    cf_r = float(reward + np.random.normal(0, self.noise_scale))
                elif self.mode == "relabel":
                    cf_s = np.array(state, dtype=np.float32).copy()
                    cf_r = float(self.beta * state[self._ret_idx])
                else:  # exact
                    if self._alt_rewards is None:
                        raise RuntimeError("exact mode needs train_agent_controls (env peek)")
                    cf_s = np.array(state, dtype=np.float32).copy()
                    cf_r = float(self.exact_beta * self._alt_rewards[alt_a])
                self.cf_memory.push(cf_s, alt_a, cf_r, next_state, done)
            self._alt_rewards = None

    return CFControlSACAgent


def make_train_agent_controls(nb):
    """train_agent from the notebook, plus one hook: before env.step, an agent
    that needs exact counterfactuals reads all three one-step rewards from the
    environment. For every other agent this is identical to nb.train_agent."""

    import copy

    def train_agent_controls(agent, train_data, val_data, n_episodes=nb.N_EPISODES,
                             use_cer=False, patience=30, verbose=False):
        eps, eps_min, eps_decay = 0.8, 0.02, 0.99
        best_val_sharpe = -np.inf
        best_state_dict = None
        no_improve = 0
        eval_every = 10
        peek = getattr(agent, "needs_env_peek", False)

        for ep in range(n_episodes):
            env = nb.TradingEnvironment(train_data)
            state = env.reset()
            done = False

            while not done:
                action = agent.act(state, eps)
                if peek:
                    agent._alt_rewards = {a: peek_reward(env, a) for a in range(3)}
                ns, reward, done = env.step(action)
                if peek:
                    assert abs(agent._alt_rewards[action] - reward) < 1e-9, "peek_reward drifted from env.step"
                agent.push(state, action, reward, ns, float(done))

                if use_cer and hasattr(agent, "generate_and_store_counterfactuals"):
                    agent.generate_and_store_counterfactuals(state, action, reward, ns, done)

                agent.train_step()
                state = ns

            eps = max(eps_min, eps * eps_decay)

            if hasattr(agent, "sync_target") and (ep + 1) % nb.TARGET_UPDATE_FREQ == 0:
                agent.sync_target()

            if val_data is not None and (ep + 1) % eval_every == 0:
                val_sharpe = nb.evaluate_on_val(agent, val_data)
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    if hasattr(agent, "q_net"):
                        best_state_dict = copy.deepcopy(agent.q_net.state_dict())
                    elif hasattr(agent, "actor"):
                        best_state_dict = copy.deepcopy(agent.actor.state_dict())
                    no_improve = 0
                else:
                    no_improve += eval_every
                    if no_improve >= patience:
                        if verbose:
                            print(f"      Early stop ep {ep+1}  best_val_sharpe={best_val_sharpe:.3f}")
                        break

        if best_state_dict is not None:
            if hasattr(agent, "q_net"):
                agent.q_net.load_state_dict(best_state_dict)
            elif hasattr(agent, "actor"):
                agent.actor.load_state_dict(best_state_dict)

    return train_agent_controls


CONTROL_METHODS = ["random_cf_sac", "relabel_sac", "exact_cf_sac"]
REFERENCE_METHODS = ["causal_sac_rerun", "discrete_sac_rerun"]


def make_control_agent(method, nb, ControlCls, scm_neural=None, vn=None):
    """Returns (agent, use_cer)."""
    if method == "random_cf_sac":
        return ControlCls("random"), True
    if method == "relabel_sac":
        return ControlCls("relabel"), True
    if method == "exact_cf_sac":
        return ControlCls("exact"), True
    if method == "causal_sac_rerun":
        return nb.CausalSACAgent(scm_neural, vn), True
    if method == "discrete_sac_rerun":
        return nb.DiscreteSACAgent(), False
    raise ValueError(method)
