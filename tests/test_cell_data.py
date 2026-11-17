"""Aggregated (text-keyed) QoIs: one value per time step per label."""
import h5py
import numpy as np
import pytest

from nuctrans_fastuq.surrogate.data_io import SECONDS_PER_YEAR, load_and_extract_data

QOI = "total_release_rate"


def _write(path, labels, t_years, values, qoi=QOI):
    with h5py.File(path, "w") as f:
        f["sampled_data/uncertain_parameters/rock_data/Host_rock/porosity"] = np.linspace(0.1, 0.2, values.shape[1])
        keys = f.create_group(f"simulation_results/{qoi}/keys")
        keys.create_dataset("0", data=labels, dtype=h5py.string_dtype())
        keys.create_dataset("1", data=np.array(t_years) * SECONDS_PER_YEAR)
        f[f"simulation_results/{qoi}/values"] = values


def test_read_labels_and_time_selection(tmp_path):
    p = tmp_path / "d.h5"
    values = np.arange(24, dtype=float).reshape(6, 4)
    _write(p, ["Host_rock"] * 3 + ["Clay"] * 3, [1.0, 2.0, 3.0] * 2, values)
    _, y, _, t = load_and_extract_data(p, QOI)
    assert list(y) == ["Host_rock", "Clay"]
    assert np.allclose(t, [1, 2, 3])
    assert np.allclose(y["Host_rock"], values[:3].T)
    assert np.allclose(y["Clay"], values[3:].T)
    _, _, _, t = load_and_extract_data(p, QOI, t_interest=[2.4])
    assert np.allclose(t, [2.0])


def test_tiny_and_negative_values_survive(tmp_path):
    p = tmp_path / "d.h5"
    v = np.array([[1e-15, 2e-15], [3e-15, -4e-15]])
    _write(p, ["A", "A"], [1.0, 2.0], v)
    _, y, _, _ = load_and_extract_data(p, QOI)
    assert np.allclose(y["A"], v.T)


def test_missing_time_step_names_the_label(tmp_path):
    p = tmp_path / "d.h5"
    _write(p, ["A", "A", "B"], [1.0, 2.0, 1.0], np.ones((3, 2)))
    with pytest.raises(ValueError, match="label 'B'"):
        load_and_extract_data(p, QOI)


def test_aggregated_qoi_rejects_z_interest(tmp_path):
    p = tmp_path / "agg.h5"
    _write(p, ["A"], [1.0], np.ones((1, 2)))
    with pytest.raises(ValueError, match="aggregated"):
        load_and_extract_data(p, QOI, z_interest=[-100.0])



def test_parse_z_config_none():
    from nuctrans_fastuq.surrogate.experiment import Experiment
    exp = Experiment.__new__(Experiment)
    exp.name = "x"
    exp.raw = {"surrogate": {"output_qoi": QOI}}
    assert exp._parse_z_config() == ([], [])
    assert exp.qoi["factor"] == 1.0


def test_qoi_settings_come_from_config(tmp_path):
    p = tmp_path / "d.h5"
    _write(p, ["A", "A"], [1.0, 2.0], np.array([[2.0, -3.0], [1e-3, -5.0]]))
    _, y, _, _ = load_and_extract_data(p, QOI, qoi={"factor": 10.0, "floor": 0.5})
    assert np.allclose(y["A"], [[20.0, 0.0], [-30.0, -50.0]])
