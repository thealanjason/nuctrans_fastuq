# Using `stratigraphy_plot.py`

`stratigraphy_plot.py` loads one site's stratigraphy (an ordered stack of rock units, each with a
depth range and lithologies) from a case directory, and draws it as a vertical column on a
matplotlib `Axes`: one filled/hatched box per unit, unit name labels, and an optional star marker
at the repository depth.

As with `map_plot.py`, you often won't call `plot_stratigraphy()` directly:
`sitefig.compose_figure.make_figure()` draws a `SiteBlock`'s stratigraphy column for you,
including the connector lines to its quantity panels and the host-rock colouring picked up from
the map. Call `plot_stratigraphy()` yourself when you want just the column, on its own `Axes`.

## Quickstart

```python
import matplotlib.pyplot as plt
from sitefig import load_site, plot_stratigraphy

site = load_site("Case3_Rocksalt")  # reads Case3_Rocksalt/{<case>.yaml, site_data/*.yaml, geometry/rock_interface.yaml}

fig, ax = plt.subplots(figsize=(1, 4))
plot_stratigraphy(ax, site)
fig.savefig("stratigraphy.png")
```

## The data model: `Unit` and `Site`

- **`Unit`** — one stratigraphic layer: `name`, `top`/`bottom` depth (metres, negative downward),
  and `lithologies` (a list; usually one entry, but a unit can mix lithologies).
- **`Site`** — the full column for one location: an ordered `units` list, which one is
  `host_unit` (default `"Host_rock"`), the repository's `repository_depth` (or `None`), a free
  `analysis` dict (e.g. `analysis["labels"]`, used by `quantities_plot.plot_quantity()` to name a
  panel's depth after a unit), the site's geometry-uncertainty bookkeeping (`boundary_keys` and
  `boundary_ranges`, see below), and `surrogate_qoi` -- this case's `surrogate.qoi` settings
  (`unit`/`factor`/`signed`/`floor`, as fed to the surrogate's training data; see
  `nuctrans_fastuq.surrogate.data_io.QOI_DEFAULT`), merged over the defaults. `quantities_plot.py`'s
  `*_from_high_fidelity`/`ensemble_from_leakage` readers use `surrogate_qoi["floor"]`/`["signed"]`/
  `["factor"]` by default, to floor and scale raw high-fidelity values the same way the surrogate
  itself was trained on (and thus the same way its own output is already floored/scaled).
  - `site.unit(name)` — the named `Unit`, or raises `KeyError`.
  - `site.host` — the host-rock `Unit` (shortcut for `site.unit(site.host_unit)`).
  - `site.zone_range(zone)` — `(top, bottom)` for a unit name or an explicit `(z1, z2)` tuple;
    used wherever a zone can be given either way (e.g. `QuantityPanel.zone`).

### Naming a depth: `"<unit>_top"` / `"<unit>_bottom"`

Anywhere in `quantities_plot.py`/`compose_figure.py` that takes a depth (`QuantityPanel.depth`,
`ensemble_from_surrogate`'s `depth`, `ensemble_from_high_fidelity`'s `z`) also accepts a name like
`"Host_rock_top"` or `"Host_rock_bottom"` instead of a number in metres — handy when the exact
depth isn't fixed (see "Geometry uncertainty" below) or you'd rather not hardcode it. Two helpers
resolve such a name against a `Site`:

- **`resolve_depth(site, z)`** — the nominal numeric depth: `z` itself if already a number, or
  `site.unit(name).top`/`.bottom` for a `"<unit>_top"`/`"<unit>_bottom"` string.
- **`boundary_key(site, z)`** — the `rock_interface.yaml` key underlying that location (e.g.
  `"Aptian_bottom"` for `"Host_rock_top"`, since the two units share that boundary), or `None` for
  a numeric `z` or one with no backing geometry key (e.g. a hardcoded surface top). Used by
  `ensemble_from_high_fidelity()` to find a boundary's per-sample sampled depth.

### Geometry uncertainty: `boundary_keys` and `boundary_ranges`

Some cases treat one or more interface depths themselves as uncertain (declared under
`uncertain_parameters.geometry.rock_interface` in the case YAML, e.g. `["Aptian_bottom",
"Host_rock_bottom"]`) rather than fixed — `load_site()` picks this up automatically:

- **`boundary_keys`** — `{(unit name, "top"/"bottom"): rock_interface.yaml key}` for every unit
  edge in the site, including shared ones (a unit's top with no explicit `"<unit>_top"` entry is
  keyed by the unit above's bottom).
- **`boundary_ranges`** — `{rock_interface.yaml key: (value_min, value_max)}`, only for the
  boundaries listed under `uncertain_parameters.geometry.rock_interface` for this case (empty for
  a case with deterministic geometry, even though `rock_interface.yaml` itself always carries
  `value_min`/`value_max` as prior metadata).

`plot_stratigraphy()` draws `boundary_ranges` as an amber, translucent, hatched band (fixed
`BOUNDARY_RANGE_COLOR`/`BOUNDARY_RANGE_ALPHA`, deliberately not tied to any unit's own lithology
colour or the red repository star) straddling each uncertain boundary (see `show_boundary_ranges`
below) -- visible even when the range is a thin slice of the column, e.g. tens of metres out of a
km-plus total depth. `quantities_plot.ensemble_from_high_fidelity()` uses `boundary_keys` to read a
boundary's own per-sample sampled depth out of an ensemble file.

## Loading a site: `load_site(case_dir, case_yaml=None, host_unit="Host_rock")`

Reads, from one case directory:

- `<case_dir.name>.yaml` (or `case_yaml`) — for the site name (`communication_sdh.site_name`) and
  the repository location (`model.source`, when its type is `"point"`).
- `site_data/<site_name>.yaml` (or the one `*.yaml` file in `site_data/`, if no site name is
  found) — for each unit's `simplified_lithology`.
- `geometry/rock_interface.yaml` — for each unit's top/bottom depth (`<unit>_top`/`<unit>_bottom`
  entries); units are ordered deepest-bottom-first.

It warns (not raises) if a unit has no `simplified_lithology` listed, and draws it with the grey
fallback style in that case.

## `plot_stratigraphy(ax, site, ...)` parameters

| Parameter | Default | Effect |
|---|---|---|
| `ax` | — | Matplotlib `Axes` to draw into. |
| `site` | — | A `Site`, e.g. from `load_site()`. |
| `z_limit` | `None` | `(z_a, z_b)` depth window, in either order; `None` shows the site's full depth range. |
| `host_color` | `None` | Fill colour for the host-rock unit (e.g. the map colour at its location); `None` uses its own lithology colour like any other unit. |
| `color_mode` | `"full"` | `"full"` fills every unit with its lithology colour; `"host_only"` leaves non-host units white (pattern only). The host unit is always filled with `host_color` when one is given, regardless of `color_mode`. |
| `lithology_mode` | `"dominant"` | `"dominant"` draws only each unit's first listed lithology; `"split"` divides the unit into equal vertical strips, one per lithology. |
| `unit_labels` | `None` | `{unit name: display label}` overrides; units not listed fall back to a label derived from their name (`"Host_rock"` → `"Host rock"`, others → their name with underscores turned into spaces/line breaks). |
| `show_labels` | `True` | Whether to draw unit name labels at all. |
| `show_repository` | `True` | Whether to draw the repository-depth star marker. |
| `show_boundary_ranges` | `True` | Whether to overlay a hatched band at each boundary in `site.boundary_ranges` (no-op if there are none — see "Geometry uncertainty" above). |
| `hatch_linewidth` | `0.5` | Line width of the hatch-pattern strokes. |
| `label_fontsize` | `5.5` | Unit label text size, in points. |
| `ylabel` | `"z [m]"` | Y-axis label; `""` or `None` omits it. |
| `label_min_height` | `0.11` | Minimum unit height (as a fraction of the axes height, in inches) below which its label is skipped, to avoid overlapping text in thin units. |

It returns the same `ax`, for chaining.

## Shared legend: `lithology_handles(sites, ...)`

Builds one de-duplicated legend (`list` of matplotlib handles) covering every lithology actually
used across several `Site`s, in `LITHOLOGY_STYLES`'s order (unknown lithologies, drawn with the
grey fallback, are appended after). Used by `compose_figure.make_figure()` to draw one shared
legend below several stratigraphy columns instead of repeating it per column.

By default it assumes each site is shown at its full depth range; pass `z_limits` (parallel to
`sites`) when a column is cropped, so lithologies/boundary ranges entirely outside that crop (e.g.
units above a `z_limit=(-300, -700)` window) don't get a legend entry nobody can see in the figure.

| Parameter | Default | Effect |
|---|---|---|
| `sites` | — | `Site`s whose units' lithologies to include. |
| `colored` | `True` | Whether patches show their lithology colour; `False` draws them white (pattern only) — pass the same value you used as `color_mode="host_only"` on the columns themselves. |
| `show_repository` | `True` | Whether to append a handle for the repository-depth star marker. |
| `repository_label` | `"Host unit"` | Legend label for that star-marker handle. |
| `lithology_mode` | `"dominant"` | Must match the `lithology_mode` passed to `plot_stratigraphy()`: `"dominant"` includes only each unit's first lithology, `"split"` includes all of them. |
| `show_boundary_ranges` | `None` | Whether to append a handle for the boundary-range hatch; `None` appends one iff any of `sites` has a boundary range visible within its own `z_limits` crop. |
| `z_limits` | `None` | `list` parallel to `sites` (each a `(z_a, z_b)` window or `None`), matching each site's own `z_limit` crop (e.g. a `SiteBlock.z_limit`) — restricts the legend to what's actually visible in that crop. `None` assumes every site is shown at its full range. |
| `boundary_range_label` | `"boundary range"` | Legend label for that handle. |

## What you can do with it

- **Draw a site's full column**, or **crop to a depth window**: `z_limit=(-1000, -1600)`.
- **Tie the host-rock colour to something external** (typically the map category colour at the
  site's location) via `host_color=...`, instead of its own lithology colour.
- **Mute non-host units to pattern-only** with `color_mode="host_only"`, so the host rock's colour
  stands out when it's carrying map-colour meaning.
- **Show every lithology a unit has**, side by side, with `lithology_mode="split"` (default
  `"dominant"` shows only the first).
- **Relabel units for display** (without touching the loaded data) via `unit_labels={...}`.
- **Hide labels or the repository marker** (`show_labels=False`, `show_repository=False`), or tune
  `label_fontsize`/`label_min_height` so labels don't overlap in thin units.
- **Build one shared lithology legend across multiple sites/columns** with `lithology_handles()`,
  rather than repeating a legend per column.
- **Pick out a specific interval** from the loaded `Site` — `site.unit(name)`, `site.host`,
  `site.zone_range(zone)` — e.g. to set a `QuantityPanel`'s `zone` without hardcoding depths.
- **Name a depth instead of hardcoding it** — `"Host_rock_top"`/`"Host_rock_bottom"` work anywhere
  a depth is expected (`QuantityPanel.depth`, `ensemble_from_surrogate`'s `depth`,
  `ensemble_from_high_fidelity`'s `z`); `resolve_depth(site, z)` resolves one yourself.
- **Show the uncertainty range of an interface's depth** as a hatched band, automatically, for any
  site whose case declares it uncertain (`uncertain_parameters.geometry.rock_interface`) — no
  extra arguments needed beyond loading the `Site` with `load_site()`.
- **Extend the lithology styling** by adding entries to `LITHOLOGY_STYLES` (colour + hatch) for a
  lithology not already listed, or to `DEFAULT_HOST_CATEGORY` (the lithology → map-category
  fallback `compose_figure.make_figure()` uses when a block has neither a map location nor an
  explicit `host_category`).
- **Combine with the map and quantity panels** in one composed figure — pass `Site`s inside
  `SiteBlock`s to `sitefig.compose_figure.make_figure()` rather than calling
  `plot_stratigraphy()`/`lithology_handles()` yourself; it handles host colouring, shared columns,
  and connector lines automatically.
