import numpy as np

from nuctrans_fastuq.surrogate.validation import validation_error


def test_metrics_for_small_qoi():
    rng = np.random.default_rng(0)
    t = rng.random((20, 3)) * 1e-9
    out = validation_error({"A": t}, {"A": t + 1e-12 * rng.random((20, 3))})
    assert np.all(np.isfinite(out["r2"]["A"]))
