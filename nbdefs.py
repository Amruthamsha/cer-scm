"""
Load every definition from the main CER-SCM notebook (data loader, causal
structure learner, SCM, environment, networks, agents, training loop) without
running any experiment.

All code cells before the "Main Experiment Loop" heading are executed in one
namespace, with shell lines (`!pip ...`, `!nvidia-smi`) stripped. This keeps the
control experiments on exactly the code that produced the paper's numbers,
rather than on a re-implementation of it.

Usage:
    from nbdefs import load_notebook_defs
    nb = load_notebook_defs("notebooks/CER_SCM_main_24_assets.ipynb")
    nb.CausalSACAgent, nb.train_agent, nb.TradingEnvironment, ...
"""

import json
import types


STOP_MARKER = "Main Experiment Loop"


def load_notebook_defs(nb_path, quiet=True):
    with open(nb_path) as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and STOP_MARKER in src:
            break
        if cell["cell_type"] != "code":
            continue
        lines = [ln for ln in src.split("\n") if not ln.lstrip().startswith("!")]
        code = "\n".join(lines).strip()
        if code:
            code_cells.append(code)

    if not code_cells:
        raise RuntimeError(f"No code cells found before '{STOP_MARKER}' in {nb_path}")

    mod = types.ModuleType("cer_scm_notebook")
    mod.__dict__["__name__"] = "cer_scm_notebook"

    import contextlib
    import io

    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        for code in code_cells:
            exec(compile(code, nb_path, "exec"), mod.__dict__)

    required = ["CausalSACAgent", "DiscreteSACAgent", "UniformReplayBuffer",
                "TradingEnvironment", "TradingDataLoader", "train_agent",
                "evaluate_agent", "learn_causal_structure", "StructuralCausalModel",
                "set_seed", "STATE_COLS", "N_CF_ACTIONS", "CF_BETA", "BUFFER_CAP",
                "ASSET_UNIVERSE", "TRAIN_START", "TEST_END", "N_TRIALS", "N_EPISODES"]
    missing = [r for r in required if r not in mod.__dict__]
    if missing:
        raise RuntimeError(f"Notebook is missing expected definitions: {missing}")
    return mod
