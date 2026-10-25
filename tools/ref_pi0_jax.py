#!/usr/bin/env python3
"""Run the OFFICIAL openpi Pi0 (pi05=True) sample_actions in JAX (CPU, float32) as a
reference for tools/export_pi0.py.  Usage: ref_pi0_jax.py CKPT IN.npz OUT.npy

Needs jax[cpu] + flax on PYTHONPATH and openpi-comet/src.  Import-only dependencies
that the inference path never touches (augmax, orbax, the pytorch port) are stubbed;
array_typing's runtime type checker is disabled (beartype does not support py3.14).
"""
import os
import sys
from unittest import mock

for m in ("augmax", "orbax", "orbax.checkpoint", "openpi.models_pytorch", "openpi.models_pytorch.pi0_pytorch"):
    sys.modules[m] = mock.MagicMock()
import openpi.shared.array_typing as at  # noqa: E402

at.typecheck = lambda f: f

import jax  # noqa: E402
import numpy as np  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_pi0 import CAMS, Params  # noqa: E402


def main():
    ckpt, npz, out = sys.argv[1:4]
    P = Params(ckpt, "cpu")
    params = {}
    for name in P.names:
        node = params
        parts = name.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = P.numpy(name)
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=32, dtype="float32")
    model = cfg.load(params, remove_extra_params=False)
    d = np.load(npz)
    names = [n for n, _, _ in CAMS]
    obs = _model.Observation.from_dict({
        "image": {n: d[n][None] for n in names},
        "image_mask": {n: np.array([True]) for n in names},
        "state": d["state"][None],
        "tokenized_prompt": d["tokens"][None],
        "tokenized_prompt_mask": d["mask"][None],
    })
    act = model.sample_actions(jax.random.key(0), obs, num_steps=10, noise=d["noise"][None])
    np.save(out, np.asarray(act[0], np.float32))
    print("ok", np.asarray(act[0, 0, :5]))


if __name__ == "__main__":
    main()
