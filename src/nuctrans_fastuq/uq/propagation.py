"""
Uncertainty propagation through a saved engine: draw input samples, evaluate the surrogate
and write inputs, predicted mean and predictive std to one HDF5 file (axes under `axes/`,
a `dims` attribute on every dataset).

    python -m nuctrans_fastuq.uq.propagation --engine engine.h5 --config propagation_opts.yaml --output out.h5

Settings: n_samples and seed (required), sampling_method (default random). Any failure stops
the task with the error and writes no file.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from ..surrogate.engine_io import load_engine
from ..surrogate.scaling import clip_scaled, clipped_fraction, descale_prediction, scaled_range
from .sensitivity import _STR, _dataset, _to_float, read_header

KEYS = ("n_samples", "seed")
SAMPLING_METHODS = ("random", "latin_hypercube", "sobol", "halton", "hammersley")


def check_settings(settings):
    """The settings of the task: n_samples (> 0) and seed, both required, sampling_method optional, nothing else."""
    settings = dict(settings or {})
    missing = [k for k in KEYS if k not in settings]
    if missing:
        raise ValueError(f"missing propagation settings: {missing}")
    unknown = sorted(set(settings) - set(KEYS) - {"sampling_method"})
    if unknown:
        raise ValueError(f"unknown propagation settings: {unknown}")
    n, seed = settings["n_samples"], settings["seed"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    method = settings.get("sampling_method", "random")
    if method not in SAMPLING_METHODS:
        raise ValueError(f"sampling_method must be one of {list(SAMPLING_METHODS)}, got {method!r}")
    return {"n_samples": n, "seed": seed, "sampling_method": method}


def propagate(engine, n_samples, seed, sampling_method="random"):
    """Sample the inputs and evaluate the surrogate: X [sample, param], mean and std [sample, time, depth], and
    clipped {"below": [depth], "above": [depth]}, the fraction of predictions clipped to the training range
    (log scaling only, else 0)."""
    np.random.seed(seed)  # chaospy draws from the global numpy generator
    x = np.asarray(engine.exp_design.generate_samples(n_samples, sampling_method), dtype=float)
    mean, std = engine.meta_model.eval_metamodel(x)  # per depth: [sample, time], in the scaled units of the training
    config = getattr(engine, "scaling_config", None)
    fractions = {}
    if config is not None:  # back to physical units: the median prediction and the exact std (lognormal for log)
        train_range = scaled_range({d: engine.exp_design.y[d] for d in engine.out_names})
        fractions = clipped_fraction(mean, config, train_range)
        mean = clip_scaled(mean, config, train_range)  # log: no overshoot above the training range
        mean, std = descale_prediction(mean, std, config)
    stack = lambda res: np.stack([np.asarray(res[d], dtype=float) for d in engine.out_names], axis=-1)  # noqa: E731
    below, above = np.array([fractions.get(d, (0.0, 0.0)) for d in engine.out_names]).reshape(-1, 2).T
    return x, stack(mean), stack(std), {"below": below, "above": above}


def write_propagation(engine, path, header, settings):
    """Propagate the inputs through the engine and write path (kept untouched if this fails).

    Returns the clipped fractions of propagate (also written as output/clipped_below and output/clipped_above).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        settings = check_settings(settings)
        x, mean, std, clipped = propagate(engine, **settings)
        with h5py.File(tmp, "w") as f:
            for key, value in header.items():
                f.attrs[key] = value
            f.attrs["n_samples"] = settings["n_samples"]
            f.attrs["seed"] = settings["seed"]
            f.attrs["sampling_method"] = settings["sampling_method"]
            f.attrs["settings_json"] = json.dumps(settings)
            f.attrs["created"] = datetime.now().isoformat()

            axes = f.create_group("axes")
            axes.create_dataset("parameter_names", data=np.array(engine.meta_model.par_names, dtype=object), dtype=_STR)
            axes.create_dataset("time", data=np.asarray(engine.exp_design.x_values, dtype=float))
            axes.create_dataset("depth", data=np.array(engine.out_names, dtype=object), dtype=_STR)
            axes.create_dataset("depth_z", data=np.array([_to_float(d) for d in engine.out_names]))

            _dataset(f.create_group("input"), "X", x, ["sample", "param"])
            out = f.create_group("output")
            for name, data in (("mean", mean), ("std", std)):
                _dataset(out, name, data, ["sample", "time", "depth"], compression="gzip", compression_opts=4)
            for side in ("below", "above"):  # fraction of the [sample, time] predictions per depth
                _dataset(out, f"clipped_{side}", clipped[side], ["depth"])
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return clipped


def main():
    parser = argparse.ArgumentParser(description="Propagate the input uncertainty through a saved engine.")
    parser.add_argument("--engine", required=True, help="engine.h5 written by the surrogate task.")
    parser.add_argument("--config", required=True, help="yaml with n_samples and seed.")
    parser.add_argument("--output", required=True, help="HDF5 file to write.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    clipped = write_propagation(load_engine(args.engine), args.output, read_header(args.engine), settings)
    print(f"wrote {args.output}")
    # every depth has the same number of predictions, so the mean over depths is the overall fraction
    print(f"predictions clipped to the training range: {clipped['below'].mean():.2%} below, "
          f"{clipped['above'].mean():.2%} above (worst depth: {clipped['below'].max():.2%}, "
          f"{clipped['above'].max():.2%})")


if __name__ == "__main__":
    main()
