from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from nuctrans_fastuq.surrogate.data_io import QOI_DEFAULT

# Lithology fill colour and hatch pattern.
LITHOLOGY_STYLES = {
    "Mudstone": {"color": "#8c6e62", "hatch": ".."},
    "Sandstone": {"color": "#dcb466", "hatch": "///"},
    "Rocksalt": {"color": "#f5dce9", "hatch": "xx"},
    "Crystalline": {"color": "#b9b9b9", "hatch": "++"},
    "Limestone": {"color": "#e9e1d0", "hatch": "--"},
    "Conglomerate": {"color": "#cb8b4f", "hatch": "oo"},
    "Dolostone": {"color": "#b8d0a8", "hatch": "\\\\\\"},
}
# a grey fallback for anything not listed
_FALLBACK_STYLE = {"color": "#dddddd", "hatch": None}

# Host-rock lithology: map category, used only when neither a map location nor host_category is given.
DEFAULT_HOST_CATEGORY = {
    "Rocksalt": "Host rock salt",
    "Crystalline": "Crystalline host rock",
    "Mudstone": "Host rock clay",
}


@dataclass
class Unit:
    """One stratigraphic unit: its name, top/bottom depth (metres, negative downward), and lithologies."""

    name: str
    top: float
    bottom: float
    lithologies: list[str]


@dataclass
class Site:
    """A stratigraphic column: an ordered list of Units, which one is the host rock, and where the
    repository sits within it.

    boundary_keys: {(unit name, "top" or "bottom"): rock_interface.yaml key} for every unit edge --
        a unit's top shares its key with the unit above's bottom when there is no explicit "<unit>_top"
        entry (e.g. "Host_rock"'s top is keyed "Aptian_bottom").
    boundary_ranges: {rock_interface.yaml key: (value_min, value_max)}, only for the boundaries this
        site's case treats as uncertain (``uncertain_parameters.geometry.rock_interface``).
    surrogate_qoi: this case's ``surrogate.qoi`` settings (unit/factor/signed/floor fed to the
        surrogate's training data; see ``nuctrans_fastuq.surrogate.data_io.QOI_DEFAULT``), merged over
        the defaults -- used to floor raw high-fidelity values the same way before plotting them.
    """

    name: str
    units: list[Unit]
    host_unit: str = "Host_rock"
    repository_depth: float | None = None
    analysis: dict = field(default_factory=dict)
    boundary_keys: dict = field(default_factory=dict)
    boundary_ranges: dict = field(default_factory=dict)
    surrogate_qoi: dict = field(default_factory=lambda: dict(QOI_DEFAULT))

    def unit(self, name):
        """The Unit named ``name``; raises KeyError if the site has no such unit."""
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(
            f"Unit {name!r} not in site {self.name}; units: {[u.name for u in self.units]}"
        )

    @property
    def host(self):
        """The host-rock Unit (``self.unit(self.host_unit)``)."""
        return self.unit(self.host_unit)

    def zone_range(self, zone):
        """(top, bottom) of a zone given as a unit name or a (z1, z2) tuple."""
        if isinstance(zone, str):
            u = self.unit(zone)
            return u.top, u.bottom
        return max(zone), min(zone)


def _scalar(entry):
    """The scalar value of a rock_interface.yaml entry, whichever of its three shapes it's given in."""
    return (
        entry[0]["value"]
        if isinstance(entry, list)
        else entry["value"] if isinstance(entry, dict) else float(entry)
    )


def _scalar_field(entry, field_name):
    """``entry[field_name]`` for a rock_interface.yaml entry (list- or dict-shaped); ``None`` if absent."""
    e = entry[0] if isinstance(entry, list) else entry
    return e.get(field_name) if isinstance(e, dict) else None


def _split_boundary_name(z):
    """``(unit name, "top"/"bottom")`` from a ``"<unit>_top"``/``"<unit>_bottom"`` string."""
    for suffix, which in (("_top", "top"), ("_bottom", "bottom")):
        if z.endswith(suffix):
            return z[: -len(suffix)], which
    raise ValueError(f"depth name {z!r} must end with '_top' or '_bottom'")


def resolve_depth(site, z):
    """Numeric depth for ``z``: itself if already a number, or ``site``'s unit top/bottom named by a
    ``"<unit>_top"``/``"<unit>_bottom"`` string (e.g. ``"Host_rock_bottom"``)."""
    if isinstance(z, str):
        name, which = _split_boundary_name(z)
        unit = site.unit(name)
        return unit.top if which == "top" else unit.bottom
    return float(z)


def boundary_key(site, z):
    """The rock_interface.yaml key underlying a ``"<unit>_top"``/``"<unit>_bottom"`` location ``z``
    (e.g. ``"Aptian_bottom"`` for ``"Host_rock_top"``, since the two units share that boundary); ``None``
    for a numeric ``z`` or one with no backing geometry key (e.g. a hardcoded surface top).
    """
    if not isinstance(z, str):
        return None
    name, which = _split_boundary_name(z)
    return site.boundary_keys.get((name, which))


def load_site(case_dir, case_yaml=None, host_unit="Host_rock"):
    """Load a site from a case directory (``<case>.yaml``, ``site_data/<site>.yaml``, ``geometry/rock_interface.yaml``).

    case_dir: the case directory to load from.
    case_yaml: path to the case YAML; defaults to ``case_dir/<case_dir.name>.yaml``.
    host_unit: name of the host-rock unit within the case's units.
    """
    case_dir = Path(case_dir)
    case_yaml = Path(case_yaml) if case_yaml else case_dir / f"{case_dir.name}.yaml"
    case = yaml.safe_load(case_yaml.read_text()) if case_yaml.exists() else {}
    site_name = (case.get("communication_sdh") or {}).get("site_name")
    site_file = (
        case_dir / "site_data" / f"{site_name}.yaml"
        if site_name
        else next((case_dir / "site_data").glob("*.yaml"))
    )
    site_data = yaml.safe_load(site_file.read_text())
    geometry = yaml.safe_load(
        (case_dir / "geometry" / "rock_interface.yaml").read_text()
    )

    bottoms = {
        k[: -len("_bottom")]: _scalar(v)
        for k, v in geometry.items()
        if k.endswith("_bottom")
    }
    tops = {
        k[: -len("_top")]: _scalar(v) for k, v in geometry.items() if k.endswith("_top")
    }
    order = sorted(bottoms, key=lambda n: -bottoms[n])
    units, boundary_keys, prev, prev_name = [], {}, None, None
    for n in order:
        if n in tops:
            top, boundary_keys[n, "top"] = tops[n], f"{n}_top"
        elif prev is not None:
            top, boundary_keys[n, "top"] = prev, f"{prev_name}_bottom"
        else:
            top = 0.0
        boundary_keys[n, "bottom"] = f"{n}_bottom"
        lith = (site_data.get(n) or {}).get("simplified_lithology") or []
        if not lith:
            warnings.warn(f"No simplified_lithology for unit {n} in {site_file.name}")
        units.append(Unit(n, top, bottoms[n], list(lith)))
        prev, prev_name = bottoms[n], n

    uncertain_boundaries = (
        (case.get("uncertain_parameters") or {}).get("geometry") or {}
    ).get("rock_interface") or []
    boundary_ranges = {}
    for key in uncertain_boundaries:
        lo = _scalar_field(geometry.get(key), "value_min")
        hi = _scalar_field(geometry.get(key), "value_max")
        if lo is not None and hi is not None:
            boundary_ranges[key] = (float(lo), float(hi))

    surrogate_qoi = {**QOI_DEFAULT, **((case.get("surrogate") or {}).get("qoi") or {})}

    source = (case.get("model") or {}).get("source") or {}
    repo = source.get("loc") if source.get("type") == "point" else None
    return Site(
        site_data.get("name", site_file.stem),
        units,
        host_unit,
        repo,
        case.get("analysis") or {},
        boundary_keys,
        boundary_ranges,
        surrogate_qoi,
    )


BOUNDARY_RANGE_HATCH = "///"
BOUNDARY_RANGE_LABEL = "boundary range"
BOUNDARY_RANGE_COLOR = "#f2a900"
BOUNDARY_RANGE_ALPHA = 0.45


def _z_window(site, z_limit):
    """(ztop, zbot) for ``site``: its full depth range, or the ``z_limit`` crop (either order)."""
    if z_limit is None:
        return site.units[0].top, site.units[-1].bottom
    return max(z_limit), min(z_limit)


def plot_stratigraphy(
    ax,
    site,
    z_limit=None,
    host_color=None,
    color_mode="full",
    lithology_mode="dominant",
    unit_labels=None,
    show_labels=True,
    show_repository=True,
    show_boundary_ranges=True,
    hatch_linewidth=0.5,
    label_fontsize=5.5,
    ylabel="z [m]",
    label_min_height=0.11,
):
    """Draw a stratigraphic column of ``site`` on ``ax``.

    ax: matplotlib Axes to draw into.
    site: a Site, e.g. from load_site().
    z_limit: (z_a, z_b) depth window, in either order; None shows the site's full depth range.
    host_color: fill colour for the host-rock unit (e.g. the map colour at its location); None uses its
        lithology colour like any other unit.
    color_mode: "full" fills every unit with its lithology colour; "host_only" leaves non-host units white
        (pattern only). The host unit is always filled with ``host_color`` (the map colour) when given.
    lithology_mode: "dominant" uses the first listed lithology; "split" shows each lithology as a vertical strip.
    unit_labels: {unit name: display label} overrides; units not listed use a label derived from their name.
    show_labels: whether to draw unit name labels.
    show_repository: whether to draw the repository-depth star marker.
    show_boundary_ranges: whether to overlay a hatched band (``site.boundary_ranges``) at each boundary
        whose depth is uncertain for this site's case; no-op if the site has none.
    hatch_linewidth: line width of the hatch pattern strokes.
    label_fontsize: unit label text size, in points.
    ylabel: y-axis label; "" or None omits it.
    label_min_height: minimum unit height, as a fraction of the axes height in inches, for its label to be drawn.
    """
    ztop, zbot = _z_window(site, z_limit)
    labels = unit_labels or {}
    ax_h_in = ax.get_position().height * ax.figure.get_figheight()
    for u in site.units:
        top, bot = min(u.top, ztop), max(u.bottom, zbot)
        if top <= bot:
            continue
        liths = u.lithologies or [None]
        if lithology_mode == "dominant":
            liths = liths[:1]
        is_host = u.name == site.host_unit
        w = 1.0 / len(liths)
        for i, lith in enumerate(liths):
            st = {**_FALLBACK_STYLE, **LITHOLOGY_STYLES.get(lith, {})}
            if is_host and host_color is not None:
                face = host_color
            elif color_mode == "full":
                face = st["color"]
            else:
                face = "white"
            r = Rectangle(
                (i * w, bot),
                w,
                top - bot,
                facecolor=face,
                edgecolor="#262626",
                hatch=st["hatch"],
                linewidth=0.6,
            )
            r.set_hatch_linewidth(hatch_linewidth)
            ax.add_patch(r)
        ax.add_patch(
            Rectangle(
                (0, bot),
                1,
                top - bot,
                facecolor="none",
                edgecolor="#262626",
                linewidth=0.8,
            )
        )
        if show_labels and (top - bot) / (ztop - zbot) * ax_h_in >= label_min_height:
            if u.name in labels:
                label = labels[u.name]
            elif u.name == "Host_rock":
                label = "Host rock"
            else:
                words = u.name.split("_")
                label = "\n".join(words) if len(u.name) > 12 else " ".join(words)
            ax.text(
                0.58 if is_host and show_repository else 0.5,
                (top + bot) / 2,
                label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                linespacing=0.95,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4),
            )

    if show_boundary_ranges:
        for lo, hi in site.boundary_ranges.values():
            lo_c, hi_c = max(min(lo, hi), zbot), min(max(lo, hi), ztop)
            if lo_c >= hi_c:
                continue
            r = Rectangle(
                (0, lo_c),
                1,
                hi_c - lo_c,
                facecolor=to_rgba(BOUNDARY_RANGE_COLOR, BOUNDARY_RANGE_ALPHA),
                edgecolor="#262626",
                linestyle=(0, (3, 2)),
                hatch=BOUNDARY_RANGE_HATCH,
                linewidth=1.0,
                zorder=4,
            )
            r.set_hatch_linewidth(hatch_linewidth)
            ax.add_patch(r)

    if show_repository:
        z = site.repository_depth
        if z is None:
            z = (site.host.top + site.host.bottom) / 2
        if zbot <= z <= ztop:
            ax.plot(
                0.13,
                z,
                marker="*",
                color="#d62728",
                markeredgecolor="#7f0000",
                markeredgewidth=0.4,
                markersize=8,
                zorder=5,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(zbot, ztop)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#bbbbbb")
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def lithology_handles(
    sites,
    colored=True,
    show_repository=True,
    repository_label="Host unit",
    lithology_mode="dominant",
    show_boundary_ranges=None,
    boundary_range_label=BOUNDARY_RANGE_LABEL,
    z_limits=None,
):
    """Legend handles for the lithologies drawn for ``sites`` (order of LITHOLOGY_STYLES).

    sites: Sites whose units' lithologies to include.
    colored: whether patches show their lithology colour; False draws them white (pattern only).
    show_repository: whether to append a handle for the repository-depth star marker.
    repository_label: legend label for the repository-depth star marker.
    lithology_mode: "dominant" includes only each unit's first lithology; "split" includes all of them (should match the lithology_mode passed to plot_stratigraphy()).
    show_boundary_ranges: whether to append a handle for the boundary-range hatch; None appends one iff
        any of ``sites`` has a boundary range visible within its own ``z_limits`` crop.
    boundary_range_label: legend label for the boundary-range hatch handle.
    z_limits: optional list parallel to ``sites`` (each a ``(z_a, z_b)`` window or ``None``), matching
        the ``z_limit`` each site's column is cropped to. ``None`` (the default) assumes every site is shown
        at its full depth range.
    """
    if z_limits is None:
        z_limits = [None] * len(sites)
    used = []
    for s, zl in zip(sites, z_limits):
        ztop, zbot = _z_window(s, zl)
        for u in s.units:
            top, bot = min(u.top, ztop), max(u.bottom, zbot)
            if top <= bot:
                continue
            liths = u.lithologies[:1] if lithology_mode == "dominant" else u.lithologies
            used += [l for l in liths if l not in used]
    order = [l for l in LITHOLOGY_STYLES if l in used] + [
        l for l in used if l not in LITHOLOGY_STYLES
    ]
    handles = []
    for l in order:
        st = {**_FALLBACK_STYLE, **LITHOLOGY_STYLES.get(l, {})}
        handles.append(
            Patch(
                facecolor=st["color"] if colored else "white",
                edgecolor="#262626",
                hatch=st["hatch"],
                linewidth=0.6,
                label=l,
            )
        )
    if show_repository:
        handles.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="none",
                color="#d62728",
                markeredgecolor="#7f0000",
                markeredgewidth=0.4,
                markersize=8,
                label=repository_label,
            )
        )
    if show_boundary_ranges is None:
        show_boundary_ranges = False
        for s, zl in zip(sites, z_limits):
            ztop, zbot = _z_window(s, zl)
            if any(
                max(min(lo, hi), zbot) < min(max(lo, hi), ztop)
                for lo, hi in s.boundary_ranges.values()
            ):
                show_boundary_ranges = True
                break
    if show_boundary_ranges:
        handles.append(
            Patch(
                facecolor=to_rgba(BOUNDARY_RANGE_COLOR, BOUNDARY_RANGE_ALPHA),
                edgecolor="#262626",
                hatch=BOUNDARY_RANGE_HATCH,
                linewidth=1.0,
                label=boundary_range_label,
            )
        )
    return handles
