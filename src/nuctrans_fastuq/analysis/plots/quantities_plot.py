from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import h5py
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

from .stratigraphy_plot import boundary_key, resolve_depth

SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass
class Ensemble:
    """Samples of one quantity vs. time at a fixed depth: ``values`` has shape (n_samples, n_times); ``time`` in years."""

    time: np.ndarray
    values: np.ndarray
    label: str = ""
    unit: str = ""
    color: str | None = None
    linestyle: str = "-"
    depth: float | None = None


def ensemble_from_surrogate(
    path,
    depth,
    site=None,
    factor=1.0,
    unit=None,
    label="Surrogate",
    color="#2ca02c",
    linestyle="-",
):
    """Surrogate output (``output/mean``) at a depth of ``axes/depth_z``/``axes/depth``.

    Args:
        path: Path to the surrogate ``propagation.h5`` file.
        depth: Either a number: the nearest value in ``axes/depth_z`` is used (must be within
            1 m, else a ``ValueError`` is raised), or a name (e.g. ``"Host_rock_bottom"``) matched
            against ``axes/depth``.
        site: Site to resolve a name ``depth`` to a nominal numeric depth with, when it isn't found
            verbatim in ``axes/depth``.
        factor: Multiplier applied to the raw values (e.g. unit conversion).
        unit: Unit label; defaults to the file's ``qoi_unit`` attribute when ``None``.
        label: Legend label for the resulting ``Ensemble``.
        color: Line/band colour for the resulting ``Ensemble``.
        linestyle: Median-line style for the resulting ``Ensemble``.

    Returns:
        Ensemble: Time series of samples at the matched depth (or a named location
        whose depth varies per sample).
    """
    with h5py.File(path, "r") as h:
        dz = h["axes/depth_z"][()]
        raw_names = h["axes/depth"][()] if "axes/depth" in h else None
        names = (
            [n.decode() if isinstance(n, bytes) else str(n) for n in raw_names]
            if raw_names is not None
            else None
        )
        if isinstance(depth, str) and names is not None and depth in names:
            i = names.index(depth)
            matched = None if np.isnan(dz[i]) else float(dz[i])
        else:
            if isinstance(depth, str):
                if site is None:
                    raise ValueError(
                        f"depth {depth!r} not in {path} (available: {names}); pass "
                        "site=<Site> to resolve it to a nominal depth"
                    )
                depth = resolve_depth(site, depth)
            if len(dz) == 1 and np.isnan(dz[0]):
                i, matched = 0, None
            else:
                finite = np.isfinite(dz)
                if not finite.any():
                    raise ValueError(
                        f"depth {depth} is numeric, but {path} only has named depths; "
                        f"available: {names}"
                    )
                i = int(np.argmin(np.where(finite, np.abs(dz - depth), np.inf)))
                if abs(dz[i] - depth) > 1.0:
                    raise ValueError(
                        f"depth {depth} not in {path}; available: {dz.tolist()}"
                    )
                matched = float(dz[i])
        vals = h["output/mean"][:, :, i] * factor
        unit = unit if unit is not None else h.attrs.get("qoi_unit", "")
        return Ensemble(
            h["axes/time"][()], vals, label, unit, color, linestyle, matched
        )


def _high_fidelity_group(h, name):
    """Read one ``simulation_results`` group as its raw ``(z, t, values)`` triples.

    Args:
        h: Open ``h5py.File`` for the high-fidelity simulation file.
        name: Name of the group under ``simulation_results`` (e.g. ``"<nuclide>"`` or
            ``"<nuclide>Flux"``).

    Returns:
        tuple: ``(k0, k1, vals)`` where ``k0`` is the depth (or zone) key per row (decoded to
        ``str`` if stored as bytes), ``k1`` is the time key per row, and ``vals`` has shape
        ``(n_rows, n_samples)``. Uses the *last* two of the group's ``keys/`` columns: a
        zone-integrated group keys each row by just ``(zone, time)``, while a depth-resolved field
        keys each row by the full ``(x, y, z, time)``, putting z/time in ``keys/2``/``keys/3``
        rather than ``keys/0``/``keys/1``. A vector-valued quantity (e.g. ``Flux``, stored as its
        ``(x, y, z)`` components) is reduced to its z-component, to match the vertical axis
        everything else here works in.
    """
    g = h["simulation_results"][name]
    n = len(g["keys"])
    k0, k1, vals = g[f"keys/{n - 2}"][()], g[f"keys/{n - 1}"][()], g["values"][()]
    if vals.ndim == 3:
        vals = vals[:, :, 2]
    if k0.dtype.kind in "OS":
        k0 = np.array([k.decode() if isinstance(k, bytes) else str(k) for k in k0])
    return k0, k1, vals


def _resolve_floor(site, floor, signed):
    """(floor, signed) to use: explicit args win; otherwise ``site.surrogate_qoi`` (0.0/``True`` -- no
    floor -- if ``site`` is ``None``)."""
    qoi = site.surrogate_qoi if site is not None else {}
    return (
        floor if floor is not None else qoi.get("floor", 0.0),
        signed if signed is not None else qoi.get("signed", True),
    )


def _apply_floor(vals, floor, signed):
    """Zero out raw values below ``floor``, exactly as the surrogate's own training data is floored: ``signed`` floors ``|vals|`` (small
    positive and negative values alike), unsigned floors ``vals`` directly. No-op for ``floor <= 0``.
    """
    if not floor:
        return vals
    small = np.abs(vals) < floor if signed else vals < floor
    vals = np.array(vals, copy=True)
    vals[small] = 0.0
    return vals


def _resolve_factor(site, factor):
    """Factor to use: explicit ``factor`` wins; otherwise ``site.surrogate_qoi["factor"]`` (the
    surrogate's own training-time unit conversion) times ``site.analysis["qoi"]["factor"]`` (a
    further display-time conversion some cases apply on top); 1.0 for either/both when missing.
    """
    if factor is not None:
        return factor
    if site is None:
        return 1.0
    surrogate_factor = site.surrogate_qoi.get("factor", 1.0)
    analysis_factor = (site.analysis.get("qoi") or {}).get("factor", 1.0)
    return surrogate_factor * analysis_factor


def _z_per_sample(h, site, z, n_samples):
    """Per-sample interpolation target(s) for ``z``, and a nominal depth for ``Ensemble.depth``/titles.

    ``z`` is a number (the same target for every sample), an array of ``n_samples`` per-sample targets,
    or a ``"<unit>_top"``/``"<unit>_bottom"`` name resolved against ``site``.
    """
    if isinstance(z, str):
        if site is None:
            raise ValueError(
                f"z={z!r} is a named depth; pass site=<Site> to resolve it"
            )
        nominal = resolve_depth(site, z)
        key = boundary_key(site, z)
        dset_path = f"sampled_data/uncertain_parameters/geometry/rock_interface/{key}"
        if key is not None and dset_path in h:
            arr = h[dset_path][()]
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"{dset_path} has {arr.shape[0]} samples, expected {n_samples}"
                )
            return arr, nominal
        return np.full(n_samples, nominal), nominal
    arr = np.atleast_1d(np.asarray(z, dtype=float))
    if arr.size == 1:
        return np.full(n_samples, float(arr[0])), float(arr[0])
    if arr.shape[0] != n_samples:
        raise ValueError(f"z array has {arr.shape[0]} entries, expected {n_samples}")
    return arr, None


def ensemble_from_high_fidelity(
    path,
    nuclide,
    quantity="concentration",
    z=None,
    site=None,
    floor=None,
    signed=None,
    factor=None,
    label="OGS",
    unit="",
    color="#1f4fd8",
    linestyle="--",
):
    """High-fidelity concentration (``quantity='concentration'``) or flux (``'flux'``) at depth ``z``.

    Each sample is linearly interpolated in z over its own finite nodes, so sample-specific meshes are
    handled (when ``z`` is a name and the ensemble's geometry is uncertain, a different
    target z per sample (the sample's own sampled boundary depth).

    Args:
        path: Path to the high-fidelity simulation HDF5 file.
        nuclide: Nuclide name; selects the ``simulation_results`` group (with ``"Flux"``
            appended when ``quantity="flux"``).
        quantity: ``"concentration"`` or ``"flux"``.
        z: Depth (metres) at which to interpolate each sample, or a name (e.g. ``"Host_rock_bottom"``).
           Samples whose finite nodes don't bracket their target z are left as NaN at that time step.
        site: Site used to resolve a name ``z`` (required when ``z`` is a name), and, unless
            overridden by ``floor``/``signed``/``factor``, the defaults below.
        floor: Raw values below this (in raw model units, before any unit conversion) are set to 0,
            the same as the surrogate's own training data (see ``_apply_floor``).
        signed: Whether ``floor`` applies to ``|value|`` (``True``) or ``value`` directly (``False``);
            ``None`` uses ``site.surrogate_qoi["signed"]`` (``True`` without a ``site``).
        factor: Multiplier applied to the raw values after flooring; ``None`` uses
            ``site.surrogate_qoi["factor"] * site.analysis["qoi"]["factor"]`` (1.0 for either/both
            when missing -- see ``_resolve_factor``).
        label: Legend label for the resulting ``Ensemble``.
        unit: Unit label for the resulting ``Ensemble``.
        color: Line/band colour for the resulting ``Ensemble``.
        linestyle: Median-line style for the resulting ``Ensemble``.

    Returns:
        Ensemble: Time series of samples interpolated to depth ``z`` (``Ensemble.depth`` is ``None``
        when ``z`` was given as a per-sample array with no single nominal value).
    """
    name = nuclide + ("Flux" if quantity == "flux" else "")
    with h5py.File(path, "r") as h:
        zk, tk, vals = _high_fidelity_group(h, name)
        z_target, depth = _z_per_sample(h, site, z, vals.shape[1])
    vals = _apply_floor(vals, *_resolve_floor(site, floor, signed)) * _resolve_factor(
        site, factor
    )
    times = np.unique(tk)
    out = np.full((vals.shape[1], times.size), np.nan)
    for j, t in enumerate(times):
        m = tk == t
        zs, v = zk[m], vals[m]
        order = np.argsort(zs)
        zs, v = zs[order], v[order]
        for s in range(vals.shape[1]):
            ok = np.isfinite(v[:, s])
            zt = z_target[s]
            if ok.sum() >= 2 and zs[ok][0] <= zt <= zs[ok][-1]:
                out[s, j] = np.interp(zt, zs[ok], v[ok, s])
    return Ensemble(times / SECONDS_PER_YEAR, out, label, unit, color, linestyle, depth)


def ensemble_from_leakage(
    path,
    zone="host rock",
    nuclide="total",
    kind="discharged",
    site=None,
    floor=None,
    signed=None,
    factor=None,
    label="OGS",
    unit="",
    color="#1f4fd8",
    linestyle="--",
):
    """Leakage out of one zone: ``kind`` in {'discharged', 'frac_discharged', 'release_rate'}.

    Args:
        path: Path to the high-fidelity simulation HDF5 file.
        zone: Zone name to select (must match a value in the group's depth/zone key).
        nuclide: Nuclide name, or ``"total"``; selects the ``simulation_results`` group
            ``f"{nuclide}_{kind}"``.
        kind: One of ``"discharged"``, ``"frac_discharged"``, ``"release_rate"``.
        site: Site whose ``surrogate_qoi`` gives the default ``floor``/``signed``/``factor`` below.
        floor: Raw values below this (in raw model units, before any unit conversion) are set to 0,
            the same as the surrogate's own training data (see ``_apply_floor``); ``None`` uses
            ``site.surrogate_qoi["floor"]`` (0.0 -- no floor -- without a ``site``).
        signed: Whether ``floor`` applies to ``|value|`` (``True``) or ``value`` directly (``False``);
            ``None`` uses ``site.surrogate_qoi["signed"]`` (``True`` without a ``site``).
        factor: Multiplier applied to the raw values after flooring; ``None`` uses
            ``site.surrogate_qoi["factor"] * site.analysis["qoi"]["factor"]`` (1.0 for either/both
            when missing -- see ``_resolve_factor``).
        label: Legend label for the resulting ``Ensemble``.
        unit: Unit label for the resulting ``Ensemble``.
        color: Line/band colour for the resulting ``Ensemble``.
        linestyle: Median-line style for the resulting ``Ensemble``.

    Returns:
        Ensemble: Time series of samples for ``zone`` (no fixed depth).
    """
    with h5py.File(path, "r") as h:
        zk, tk, vals = _high_fidelity_group(h, f"{nuclide}_{kind}")
    vals = _apply_floor(vals, *_resolve_floor(site, floor, signed)) * _resolve_factor(
        site, factor
    )
    m = zk == zone
    if not m.any():
        raise ValueError(
            f"zone {zone!r} not found; available: {sorted(set(zk.tolist()))}"
        )
    order = np.argsort(tk[m])
    return Ensemble(
        tk[m][order] / SECONDS_PER_YEAR, vals[m][order].T, label, unit, color, linestyle
    )


@dataclass
class QuantityPanel:
    """Styling/layout config for one time-series axes (scales, thresholds, which Ensembles to overlay).

    depth: z of the quantity (for connection lines to the stratigraphy): a number, or a name (e.g.
        "Host_rock_bottom") resolved via the Site passed to plot_quantity()/make_figure(). zone: unit
        name or (z1, z2) for leakage.
    xscale/yscale: 'linear', 'log' or 'symlog'. threshold: value or list of values drawn as dashed lines.
    absolute: plot |values| (useful for log scales with downward, negative fluxes).
    """

    ensembles: list[Ensemble]
    depth: float | str | None = None
    zone: str | tuple | None = None
    title: str | None = None
    ylabel: str | None = None
    xscale: str = "linear"
    yscale: str = "linear"
    threshold: float | list | None = None
    threshold_label: str = "Threshold"
    absolute: bool = False
    quantiles: tuple = (5, 95)
    linthresh: float | None = None
    xlim: tuple | None = None
    ylim: tuple | None = None
    legend: bool | None = None
    extra: dict = field(default_factory=dict)


def plot_quantity(ax, panel, site=None, show_xlabel=True, legend=True):
    """Median line and quantile band for each ensemble in ``panel``, plus optional thresholds.

    Args:
        ax: Matplotlib axes to draw into.
        panel: QuantityPanel describing the ensembles and styling to plot.
        site: Site used to resolve ``panel.depth`` to a named label (``site.analysis["labels"]``)
            for the panel title; if ``None``, the title falls back to ``"z = ... m"``.
        show_xlabel: Whether to draw the "Time [years]" x-axis label.
        legend: Whether to draw the legend (overridden by ``panel.legend`` when it is not ``None``).

    Returns:
        The same ``ax``, for chaining.
    """
    lo_q, hi_q = panel.quantiles
    all_vals = []
    for k, e in enumerate(panel.ensembles):
        t, v = np.asarray(e.time, float), np.asarray(e.values, float)
        if panel.absolute:
            v = np.abs(v)
        keep = t > 0 if panel.xscale == "log" else np.ones_like(t, bool)
        t, v = t[keep], v[:, keep]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(v, axis=0)
            lo, hi = np.nanpercentile(v, [lo_q, hi_q], axis=0)
        color = e.color or f"C{k}"
        name = e.label or f"series {k + 1}"
        ax.fill_between(
            t,
            lo,
            hi,
            color=color,
            alpha=0.3,
            linewidth=0,
            label=f"{name} [{lo_q}%–{hi_q}%]",
        )
        ax.plot(t, lo, color=color, linewidth=0.3, alpha=0.8)
        ax.plot(t, hi, color=color, linewidth=0.3, alpha=0.8)
        ax.plot(
            t,
            med,
            color=color,
            linestyle=e.linestyle,
            linewidth=1.3,
            label=f"{name} median",
        )
        all_vals.append(np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]]))

    thresholds = (
        [] if panel.threshold is None else np.atleast_1d(panel.threshold).tolist()
    )
    for i, th in enumerate(thresholds):
        ax.axhline(
            th,
            color="#d62a3c",
            linestyle="--",
            linewidth=1.0,
            label=panel.threshold_label if i == 0 else None,
        )

    ax.set_xscale(panel.xscale)
    if panel.yscale == "symlog":
        vals = (
            np.concatenate(all_vals + [np.abs(thresholds)])
            if all_vals
            else np.array([1.0])
        )
        nz = np.abs(vals[vals != 0])
        lt = panel.linthresh or (
            10 ** np.floor(np.log10(nz.max())) * 1e-2 if nz.size else 1.0
        )
        ax.set_yscale("symlog", linthresh=lt)
        ax.yaxis.set_minor_locator(NullLocator())
    elif panel.yscale == "log":
        ax.set_yscale("log", nonpositive="mask")
        ax.yaxis.set_minor_locator(LogLocator(subs="auto"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if panel.xlim:
        ax.set_xlim(panel.xlim)
    if panel.ylim:
        ax.set_ylim(panel.ylim)

    unit = next((e.unit for e in panel.ensembles if e.unit), "")
    ylabel = panel.ylabel if panel.ylabel is not None else (f"[{unit}]" if unit else "")
    if panel.absolute and panel.ylabel is None and ylabel:
        ylabel = f"|value| {ylabel}"
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("Time [years]")
    if panel.title is not None:
        title = panel.title
    elif panel.zone is not None:
        title = (
            panel.zone.replace("_", " ")
            if isinstance(panel.zone, str)
            else f"zone {max(panel.zone):g} to {min(panel.zone):g} m"
        )
    elif panel.depth is not None:
        labels = (site.analysis.get("labels") if site is not None else None) or {}
        if isinstance(panel.depth, str):
            name = next((k for k, v in labels.items() if v == panel.depth), None)
            title = name or panel.depth.replace("_", " ")
        else:
            name = next(
                (
                    k
                    for k, v in labels.items()
                    if isinstance(v, (int, float)) and abs(v - panel.depth) < 1.0
                ),
                None,
            )
            title = (
                f"{name} (z = {panel.depth:g} m)"
                if name
                else f"z = {panel.depth:.1f} m"
            )
    else:
        title = ""
    ax.set_title(title)
    ax.grid(True, which="major", linewidth=0.4, color="#dddddd")
    if panel.legend if panel.legend is not None else legend:
        ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="#cccccc")
    return ax


@dataclass
class DepthProfile:
    """Samples of a quantity over depth at one time: ``values`` has shape (n_samples, n_depths); ``depth`` in metres."""

    depth: np.ndarray
    values: np.ndarray
    time: float | None = None
    label: str = ""
    unit: str = ""
    color: str | None = None
    linestyle: str = "-"


def depth_profile_from_surrogate(
    path, time, factor=1.0, unit=None, label="Surrogate", color="#2ca02c", linestyle="-"
):
    """Surrogate output (``output/mean``) over depth (``axes/depth_z``), at the time in ``axes/time`` closest to ``time``.

    Args:
        path: Path to the surrogate ``propagation.h5`` file.
        time: Target time (years); the nearest value in ``axes/time`` is used.
        factor: Multiplier applied to the raw values (e.g. unit conversion).
        unit: Unit label; defaults to the file's ``qoi_unit`` attribute when ``None``.
        label: Legend label for the resulting ``DepthProfile``.
        color: Line/band colour for the resulting ``DepthProfile``.
        linestyle: Median-line style for the resulting ``DepthProfile``.

    Returns:
        DepthProfile: Depth profile of samples at the matched time.
    """
    with h5py.File(path, "r") as h:
        t = h["axes/time"][()]
        i = int(np.argmin(np.abs(t - time)))
        dz = h["axes/depth_z"][()]
        order = np.argsort(dz)
        vals = h["output/mean"][:, i, :][:, order] * factor
        unit = unit if unit is not None else h.attrs.get("qoi_unit", "")
        return DepthProfile(dz[order], vals, float(t[i]), label, unit, color, linestyle)


def depth_profile_from_high_fidelity(
    path,
    nuclide,
    quantity="concentration",
    time=0.0,
    site=None,
    floor=None,
    signed=None,
    factor=None,
    label="OGS",
    unit="",
    color="#1f4fd8",
    linestyle="--",
):
    """High-fidelity concentration/flux profile over depth, at the time in the data closest to ``time`` (years).

    Args:
        path: Path to the high-fidelity simulation HDF5 file.
        nuclide: Nuclide name; selects the ``simulation_results`` group (with ``"Flux"``
            appended when ``quantity="flux"``).
        quantity: ``"concentration"`` or ``"flux"``.
        time: Target time (years); the nearest available time in the data is used.
        site: Site whose ``surrogate_qoi`` gives the default ``floor``/``signed``/``factor`` below.
        floor: Raw values below this (in raw model units, before any unit conversion) are set to 0,
            the same as the surrogate's own training data (see ``_apply_floor``); ``None`` uses
            ``site.surrogate_qoi["floor"]``.
        signed: Whether ``floor`` applies to ``|value|`` (``True``) or ``value`` directly (``False``);
            ``None`` uses ``site.surrogate_qoi["signed"]`` (``True`` without a ``site``).
        factor: Multiplier applied to the raw values after flooring; ``None`` uses
            ``site.surrogate_qoi["factor"] * site.analysis["qoi"]["factor"]`` (1.0 for either/both
            when missing -- see ``_resolve_factor``).
        label: Legend label for the resulting ``DepthProfile``.
        unit: Unit label for the resulting ``DepthProfile``.
        color: Line/band colour for the resulting ``DepthProfile``.
        linestyle: Median-line style for the resulting ``DepthProfile``.

    Returns:
        DepthProfile: Depth profile of samples at the matched time.
    """
    name = nuclide + ("Flux" if quantity == "flux" else "")
    with h5py.File(path, "r") as h:
        zk, tk, vals = _high_fidelity_group(h, name)
    vals = _apply_floor(vals, *_resolve_floor(site, floor, signed)) * _resolve_factor(
        site, factor
    )
    times = np.unique(tk)
    t = times[np.argmin(np.abs(times / SECONDS_PER_YEAR - time))]
    m = tk == t
    order = np.argsort(zk[m])
    return DepthProfile(
        zk[m][order],
        vals[m][order].T,
        float(t / SECONDS_PER_YEAR),
        label,
        unit,
        color,
        linestyle,
    )


@dataclass
class DepthPanel:
    """One depth-profile panel: quantity vs depth at a fixed time (or several profiles overlaid).

    z_limit: (z_a, z_b) depth window, in either order; None shows the profiles' own depth range. When composed
        with a stratigraphy column (see compose_figure.SiteBlock.depth_panels), the stratigraphy's own z_limit is used
        instead, so the two share one depth axis.
    """

    profiles: list[DepthProfile]
    z_limit: tuple | None = None
    title: str | None = None
    xlabel: str | None = None
    xscale: str = "linear"
    xlim: tuple | None = None
    quantiles: tuple = (5, 95)
    legend: bool | None = None


def plot_depth_profile(ax, panel, show_ylabel=True, legend=True):
    """Median line and quantile band vs depth for each profile in ``panel`` (x = quantity, y = depth).

    Args:
        ax: Matplotlib axes to draw into.
        panel: DepthPanel describing the profiles and styling to plot.
        show_ylabel: Whether to draw the "z [m]" y-axis label and its tick labels.
        legend: Whether to draw the legend (overridden by ``panel.legend`` when it is not ``None``).

    Returns:
        The same ``ax``, for chaining.
    """
    lo_q, hi_q = panel.quantiles
    for k, p in enumerate(panel.profiles):
        z, v = np.asarray(p.depth, float), np.asarray(p.values, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(v, axis=0)
            lo, hi = np.nanpercentile(v, [lo_q, hi_q], axis=0)
        color = p.color or f"C{k}"
        name = p.label or f"series {k + 1}"
        ax.fill_betweenx(
            z,
            lo,
            hi,
            color=color,
            alpha=0.3,
            linewidth=0,
            label=f"{name} [{lo_q}%–{hi_q}%]",
        )
        ax.plot(lo, z, color=color, linewidth=0.3, alpha=0.8)
        ax.plot(hi, z, color=color, linewidth=0.3, alpha=0.8)
        ax.plot(
            med,
            z,
            color=color,
            linestyle=p.linestyle,
            linewidth=1.3,
            label=f"{name} median",
        )

    ax.set_xscale(panel.xscale)
    if panel.xlim:
        ax.set_xlim(panel.xlim)
    if panel.z_limit:
        ax.set_ylim(min(panel.z_limit), max(panel.z_limit))

    unit = next((p.unit for p in panel.profiles if p.unit), "")
    xlabel = panel.xlabel if panel.xlabel is not None else (f"[{unit}]" if unit else "")
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("z [m]")
    else:
        ax.tick_params(labelleft=False)

    title = panel.title
    if title is None:
        t = next((p.time for p in panel.profiles if p.time is not None), None)
        title = f"t = {t:g} yr" if t is not None else ""
    ax.set_title(title)
    ax.grid(True, which="major", linewidth=0.4, color="#dddddd")
    if panel.legend if panel.legend is not None else legend:
        ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="#cccccc")
    return ax
