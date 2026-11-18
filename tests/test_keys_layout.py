import h5py
import numpy as np
import pytest

from nuctrans_fastuq.surrogate.data_io import SECONDS_PER_YEAR, load_and_extract_data


@pytest.mark.parametrize("group", [True, False])
def test_field_keys_both_layouts(tmp_path, group):
    cols = [np.zeros(2), np.zeros(2), np.array([0.0, -10.0]), np.full(2, SECONDS_PER_YEAR)]
    p = tmp_path / "field.h5"
    with h5py.File(p, "w") as h:
        h["sampled_data/uncertain_parameters/rock_data/Host_rock/porosity"] = np.array([0.1, 0.2])
        if group:
            for i, c in enumerate(cols):
                h[f"simulation_results/conc/keys/{i}"] = c
        else:
            h["simulation_results/conc/keys"] = np.column_stack(cols)
        h["simulation_results/conc/values"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    _, y, _, _ = load_and_extract_data(p, "conc", z_interest=[-10.0])
    assert np.allclose(y["-10.0"], [[3.0], [4.0]])
