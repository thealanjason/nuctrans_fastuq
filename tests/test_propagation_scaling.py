"""propagate() returns physical units for an engine trained on scaled outputs."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from nuctrans_fastuq.surrogate.scaling import descale_prediction, fit_scaling, scale_data
from nuctrans_fastuq.uq.propagation import propagate, write_propagation


def _engine(config, mean_scaled, std_scaled):
    meta = SimpleNamespace(eval_metamodel=lambda x: ({"-10.0": mean_scaled}, {"-10.0": std_scaled}),
                           par_names=["a", "b"])
    design = SimpleNamespace(generate_samples=lambda n, method: np.zeros((n, 2)), y={"-10.0": mean_scaled},
                             x_values=np.arange(3.0))
    return SimpleNamespace(meta_model=meta, exp_design=design, out_names=["-10.0"], scaling_config=config)


KINDS = [None, "div_by_max", "standardize", "minmax", "log"]
Y = np.array([[0.0, 1e-9, 1e-6], [0.0, 3e-9, 1e-3]])


@pytest.mark.parametrize("kind", KINDS)
def test_propagate_descales(kind):
    config = fit_scaling({"-10.0": Y}, kind) if kind else None
    mean_scaled = scale_data({"-10.0": Y}, config)["-10.0"] if config else Y
    _, mean, std, clipped = propagate(_engine(config, mean_scaled, np.full_like(Y, 0.3)), n_samples=2, seed=0)
    np.testing.assert_allclose(mean[:, :, 0], Y, rtol=1e-9, atol=1e-18)  # the median: not shifted by the std
    assert np.all(std > 0)
    np.testing.assert_array_equal(clipped["below"], [0.0])  # inside the training range: nothing clipped
    np.testing.assert_array_equal(clipped["above"], [0.0])


@pytest.mark.parametrize("kind", KINDS)
def test_propagate_returns_the_exact_mean_and_std(kind):
    config = fit_scaling({"-10.0": Y}, kind) if kind else None
    mean_scaled = scale_data({"-10.0": Y}, config)["-10.0"] if config else Y
    std_scaled = np.full_like(Y, 0.3)
    _, mean, std, _ = propagate(_engine(config, mean_scaled, std_scaled), n_samples=2, seed=0)
    want_mean, want_std = descale_prediction(mean_scaled, std_scaled, config)
    np.testing.assert_array_equal(mean[:, :, 0], want_mean)
    np.testing.assert_array_equal(std[:, :, 0], want_std)


def _overshooting_engine():
    """Log engine whose prediction overshoots the training range in one of its 6 values."""
    config = fit_scaling({"-10.0": Y}, "log")
    train = scale_data({"-10.0": Y}, config)["-10.0"]
    wild = train.copy()
    wild[0, 2] = train.max() + 50.0  # a polynomial overshoot: exp of this is ~1e21 times too large
    engine = _engine(config, wild, np.full_like(Y, 0.1))
    engine.exp_design.y = {"-10.0": train, "x_values": np.arange(3.0)}
    return engine, config, train


def test_propagate_clips_log_overshoot():
    engine, config, train = _overshooting_engine()
    _, mean, std, clipped = propagate(engine, n_samples=2, seed=0)
    ceiling_mean, ceiling_std = descale_prediction(train.max() + 1.0, 0.1, config)  # mean one log unit above
    assert mean.max() <= ceiling_mean * (1 + 1e-9)
    assert np.all(np.isfinite(std)) and std.max() <= ceiling_std * (1 + 1e-9)
    np.testing.assert_allclose(clipped["above"], [1 / 6])
    np.testing.assert_array_equal(clipped["below"], [0.0])


def test_write_propagation_stores_the_clipped_fractions(tmp_path):
    engine, _, _ = _overshooting_engine()
    path = tmp_path / "propagation.h5"
    clipped = write_propagation(engine, path, {}, {"n_samples": 2, "seed": 0})
    np.testing.assert_allclose(clipped["above"], [1 / 6])
    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["output/clipped_above"][()], [1 / 6])
        np.testing.assert_array_equal(f["output/clipped_below"][()], [0.0])
        assert list(f["output/clipped_above"].attrs["dims"]) == ["depth"]
