"""Output scaling: scale -> descale, the std, and the scaling stored in engine.h5."""
import h5py
import numpy as np
import pytest

from nuctrans_fastuq.surrogate.engine_io import _attr, _write_scaling, build_scaling
from nuctrans_fastuq.surrogate.scaling import descale_data, descale_prediction, descale_std, fit_scaling, scale_data

LINEAR = ["div_by_max", "standardize", "minmax"]
KINDS = LINEAR + ["log"]


def _outputs():
    """Concentration-like outputs: exact zeros (early times) and several orders of magnitude."""
    rng = np.random.default_rng(0)
    a = 10.0 ** rng.uniform(-12, -4, (50, 6))
    a[:, :2] = 0.0
    b = 10.0 ** rng.uniform(-9, -5, (50, 6))
    b[rng.random(b.shape) < 0.2] = 0.0
    return {"-10.0": a, "-20.0": b}


@pytest.mark.parametrize("kind", KINDS)
def test_descale_undoes_scale(kind):
    y = _outputs()
    cfg = fit_scaling(y, kind)
    scaled = scale_data(y, cfg)
    back = descale_data(scaled, cfg)
    for k in y:
        assert np.all(np.isfinite(scaled[k]))
        np.testing.assert_allclose(back[k], y[k], rtol=1e-9, atol=1e-12 * y[k].max())


@pytest.mark.parametrize("kind", LINEAR)
def test_linear_std_is_the_slope_of_descale(kind):
    y = _outputs()
    cfg = fit_scaling(y, kind)
    mean = scale_data(y, cfg)
    std = {k: np.full_like(v, 1e-3) for k, v in mean.items()}
    got = descale_std(std, cfg)
    h = 1e-3  # descale is linear, so any step gives the exact slope
    for k in y:
        slope = (descale_data(mean[k] + h, cfg) - descale_data(mean[k] - h, cfg)) / (2 * h)
        np.testing.assert_allclose(got[k], slope * std[k], rtol=1e-6)


def test_minmax_std_with_constant_training_outputs():
    """data_range_ is 0 here, but inverse_transform still has slope 1/scale_ = 1: the std must not become 0."""
    cfg = fit_scaling({"a": np.full((4, 3), 3e-6)}, "minmax")
    assert cfg["scaler"].data_range_[0] == 0.0
    slope = (descale_data(np.array([0.4]), cfg) - descale_data(np.array([0.2]), cfg))[0] / 0.2
    assert descale_std(0.1, cfg) == pytest.approx(0.1 * slope)
    assert descale_std(0.1, cfg) > 0


@pytest.mark.parametrize("kind", [None] + KINDS)
def test_descale_prediction_matches_sampling(kind):
    """y ~ N(mu, s^2) in scaled units: the median and std of descale(y) over many draws are what
    descale_prediction gives. For log the std needs the exact lognormal formula: with s = 0.5 the first-order
    std (scale * exp(mu) * s) is 17% too small."""
    cfg = fit_scaling(_outputs(), kind)
    mu = scale_data(np.array([0.0, 1e-9, 1e-6, 1e-4]), cfg)
    s = np.full_like(mu, 0.5 if kind == "log" else 0.1)
    median, std = descale_prediction(mu, s, cfg)

    n = 400_000
    draws = descale_data(mu + s * np.random.default_rng(1).standard_normal((n, mu.size)), cfg)
    np.testing.assert_allclose(draws.std(axis=0), std, rtol=0.02)
    assert np.all(np.abs(np.median(draws, axis=0) - median) < 5 * std / np.sqrt(n))


@pytest.mark.parametrize("kind", [None] + LINEAR)
def test_descale_prediction_linear_is_descale_data_and_descale_std(kind):
    y = _outputs()
    cfg = fit_scaling(y, kind)
    mean = scale_data(y, cfg)
    std = {k: np.full_like(v, 0.3) for k, v in mean.items()}
    got_mean, got_std = descale_prediction(mean, std, cfg)
    for k in y:
        np.testing.assert_array_equal(got_mean[k], descale_data(mean, cfg)[k])
        np.testing.assert_array_equal(got_std[k], descale_std(std, cfg)[k])
    unscaled = descale_prediction(mean, std, None)  # an engine without scaling: returned as is
    assert unscaled[0] is mean and unscaled[1] is std


@pytest.mark.parametrize("kind", KINDS)
def test_scaling_survives_the_engine_file(tmp_path, kind):
    y = _outputs()
    cfg = fit_scaling(y, kind)
    path = tmp_path / "engine.h5"
    with h5py.File(path, "w") as f:
        _write_scaling(f, cfg)
    with h5py.File(path, "r") as f:
        loaded = build_scaling({k: _attr(v) for k, v in f["scaling"].attrs.items()})

    assert loaded["type"] == kind
    scaled = scale_data(y, cfg)
    std = {k: np.full_like(v, 1e-3) for k, v in scaled.items()}
    for k in y:
        np.testing.assert_array_equal(scale_data(y, loaded)[k], scaled[k])
        np.testing.assert_array_equal(descale_data(scaled, loaded)[k], descale_data(scaled, cfg)[k])
        for got, want in zip(descale_prediction(scaled, std, loaded), descale_prediction(scaled, std, cfg)):
            np.testing.assert_array_equal(got[k], want[k])


# ---- log only

def test_log_keeps_zeros_at_zero():
    y = {"a": np.array([[0.0, 1e-9, 1e-6], [0.0, 3e-9, 1e-3]])}
    cfg = fit_scaling(y, "log")
    assert cfg["scale"] == 1e-9  # the smallest positive value

    scaled = scale_data(y, cfg)["a"]
    assert np.all(np.isfinite(scaled))  # a plain log would give -inf here
    assert np.all(scaled[:, 0] == 0.0)
    assert np.all(np.diff(scaled, axis=1) > 0)  # increasing, so the order is kept

    back = descale_data({"a": scaled}, cfg)["a"]
    assert np.all(back[:, 0] == 0.0)
    np.testing.assert_allclose(back, y["a"], rtol=1e-12)


def test_log_takes_a_float():
    cfg = fit_scaling({"a": np.array([[0.0, 1e-9, 1e-6]])}, "log")
    assert descale_data(scale_data(2e-7, cfg), cfg) == pytest.approx(2e-7, rel=1e-12)


def test_log_without_any_positive_value_does_not_divide_by_zero():
    cfg = fit_scaling({"a": np.zeros((3, 2))}, "log")
    assert cfg["scale"] == 1.0
    assert np.all(scale_data({"a": np.zeros((3, 2))}, cfg)["a"] == 0.0)


def test_log_rejects_negative_outputs():
    with pytest.raises(ValueError, match="log scaling"):
        fit_scaling({"a": np.array([[-1e-6, 1e-9, 1e-6]])}, "log")

    # noise below the smallest positive value is inside the domain
    fit_scaling({"a": np.array([[-1e-12, 0.0, 1e-9, 1e-6]])}, "log")

    # validation data outside the domain of the fitted scaling
    cfg = fit_scaling({"a": np.array([[0.0, 1e-9, 1e-6]])}, "log")
    with pytest.raises(ValueError, match="log scaling"):
        scale_data({"a": np.array([[-1e-6]])}, cfg)


def test_log_std_exact_formula():
    cfg = fit_scaling(_outputs(), "log")
    c, mu, s = cfg["scale"], np.array([0.0, 2.0, 5.0]), np.array([0.05, 0.5, 1.0])
    _, std = descale_prediction(mu, s, cfg)
    np.testing.assert_allclose(std ** 2, c ** 2 * np.exp(2 * mu + s ** 2) * (np.exp(s ** 2) - 1), rtol=1e-10)


@pytest.mark.parametrize("s", [0.0, 0.5, 2.0])
def test_log_prediction_is_the_median_whatever_the_std(s):
    """Not the mean scale * expm1(mu + s^2/2): that would shift the percentiles over samples up."""
    cfg = fit_scaling(_outputs(), "log")
    mu = {"a": np.array([0.0, 2.0, 5.0])}
    median, std = descale_prediction(mu, {"a": np.full(3, s)}, cfg)
    np.testing.assert_array_equal(median["a"], cfg["scale"] * np.expm1(mu["a"]))
    assert np.all(std["a"] == 0.0) if s == 0.0 else np.all(std["a"] > 0.0)


def test_log_prediction_takes_floats_and_tiny_stds():
    cfg = fit_scaling(_outputs(), "log")
    median, std = descale_prediction(2.0, 1e-9, cfg)
    assert float(median) == pytest.approx(cfg["scale"] * np.expm1(2.0), rel=1e-12)
    assert float(std) == pytest.approx(cfg["scale"] * np.exp(2.0) * 1e-9, rel=1e-8)  # no cancellation in expm1


def test_log_std_is_larger_where_the_prediction_is_larger():
    cfg = fit_scaling(_outputs(), "log")
    _, std = descale_prediction(np.array([0.0, 5.0, 10.0]), np.full(3, 0.1), cfg)
    assert np.all(np.diff(std) > 0)


# ---- clipping of log-space overshoot

def test_clip_scaled_log_limits_to_training_range_plus_margin():
    from nuctrans_fastuq.surrogate.scaling import clip_scaled, scaled_range
    y = _outputs()
    cfg = fit_scaling(y, "log")
    train = scale_data(y, cfg)
    lo, hi = scaled_range({**train, "x_values": np.arange(3.0)})  # x_values is skipped
    assert lo == 0.0
    out = clip_scaled({"a": np.array([-5.0, 0.5, hi, hi + 0.5, hi + 40.0])}, cfg, (lo, hi), margin=1.0)["a"]
    np.testing.assert_allclose(out, [0.0, 0.5, hi, hi + 0.5, hi + 1.0])
    assert clip_scaled(hi + 40.0, cfg, (lo, hi)) == hi + 1.0  # a float works too


def test_clipped_fraction_counts_what_clip_scaled_changes():
    from nuctrans_fastuq.surrogate.scaling import clip_scaled, clipped_fraction
    cfg = fit_scaling(_outputs(), "log")
    lo, hi = 0.0, 10.0
    pred = {"a": np.array([[-1.0, 0.0, 5.0, 11.0]]), "b": np.array([[1.0, 2.0], [3.0, 12.0]])}
    got = clipped_fraction(pred, cfg, (lo, hi), margin=1.0)
    assert got == {"a": (0.25, 0.0), "b": (0.0, 0.25)}  # 11 is inside hi + margin, 12 is not
    changed = {k: np.mean(clip_scaled(v, cfg, (lo, hi), margin=1.0) != v) for k, v in pred.items()}
    assert changed == {k: sum(f) for k, f in got.items()}


@pytest.mark.parametrize("kind", ["div_by_max", "standardize", "minmax", None])
def test_clipped_fraction_is_empty_for_linear_scalings(kind):
    from nuctrans_fastuq.surrogate.scaling import clipped_fraction
    cfg = fit_scaling(_outputs(), kind) if kind else None
    assert clipped_fraction({"a": np.array([-100.0, 100.0])}, cfg, (0.0, 1.0)) == {}


def test_training_summary_shows_the_clipped_fractions():
    from nuctrans_fastuq.surrogate.runner import _build_summary_lines
    metrics = {m: {"-10.0": np.array([0.5, 0.7])} for m in
               ("r2", "rmse", "pearson_r", "nse", "mean_error", "std_error")}
    without = "\n".join(_build_summary_lines("PCE", 10, 1.0, metrics, ["-10.0"]))
    assert "clipped" not in without  # linear scalings: no clipping, no section

    metrics.update(clipped_below={"-10.0": 0.0}, clipped_above={"-10.0": 1 / 6})
    lines = _build_summary_lines("PCE", 10, 1.0, metrics, ["-10.0"])
    assert "clipped to the training range" in "\n".join(lines)
    assert lines[-1].split() == ["-10.0", "0.00%", "16.67%"]


@pytest.mark.parametrize("kind", ["div_by_max", "standardize", "minmax", None])
def test_clip_scaled_leaves_linear_scalings_alone(kind):
    from nuctrans_fastuq.surrogate.scaling import clip_scaled
    cfg = fit_scaling(_outputs(), kind) if kind else None
    x = {"a": np.array([-100.0, 100.0])}
    assert clip_scaled(x, cfg, (0.0, 1.0))["a"] is x["a"]
