# Using `quantities_plot.py`

`quantities_plot.py` draws quantity-vs-time and quantity-vs-depth panels: a median line plus a
quantile band for one or more overlaid series, with optional threshold lines. It also reads the
sample data for those series out of two different file formats — the cheap **surrogate** model's
grid output, and the expensive **high-fidelity** (OGS) simulation's sparse, per-sample output.

As with `map_plot.py`/`stratigraphy_plot.py`, `sitefig.compose_figure.make_figure()` draws these
panels for you (and adds the connector lines from the stratigraphy to each panel's depth/zone) —
call `plot_quantity()`/`plot_depth_profile()` yourself only when you want a panel on its own
`Axes`, outside a composed figure.

## Quickstart

```python
import matplotlib.pyplot as plt
from sitefig import ensemble_from_surrogate, QuantityPanel, plot_quantity

e = ensemble_from_surrogate("propagation.h5", depth=-1200, unit="mol/L")
panel = QuantityPanel([e], depth=-1200, yscale="log")

fig, ax = plt.subplots()
plot_quantity(ax, panel)
fig.savefig("quantity.png")
```

## The data model: `Ensemble` and `DepthProfile`

- **`Ensemble`** — samples of one quantity vs. **time** at a fixed depth: `values` has shape
  `(n_samples, n_times)`, `time` is in years. Fields: `time`, `values`, `label`, `unit`, `color`,
  `linestyle`, `depth`.
- **`DepthProfile`** — samples of one quantity vs. **depth** at a fixed time: `values` has shape
  `(n_samples, n_depths)`, `depth` is in metres. Fields: `depth`, `values`, `time`, `label`,
  `unit`, `color`, `linestyle`.

You build these with the reader functions below, or construct them directly if your data comes
from somewhere else — `plot_quantity()`/`plot_depth_profile()` only care about the fields above.

## Building an `Ensemble` (quantity vs. time)

| Function | Reads from | Fixed by |
|---|---|---|
| `ensemble_from_surrogate(path, depth, site=None, factor=1.0, unit=None, label="Surrogate", color="#2ca02c", linestyle="-")` | A surrogate `propagation.h5` grid (`output/mean[sample, time, depth]`). | `depth`: a number (nearest `axes/depth_z`, within 1 m), or a name (e.g. `"Host_rock_bottom"`) matched against `axes/depth`. |
| `ensemble_from_high_fidelity(path, nuclide, quantity="concentration", z=None, site=None, floor=None, signed=None, factor=None, label="OGS", unit="", color="#1f4fd8", linestyle="--")` | A high-fidelity simulation file's sparse `(z, t, value)` triples (`simulation_results/<nuclide>[Flux]`). | `z`, by linear interpolation in depth per sample (NaN where a sample's mesh doesn't bracket its target). |
| `ensemble_from_leakage(path, zone="host rock", nuclide="total", kind="discharged", site=None, floor=None, signed=None, factor=None, label="OGS", unit="", color="#1f4fd8", linestyle="--")` | The same high-fidelity file's zone-integrated groups (`simulation_results/<nuclide>_<kind>`). | `zone` name (no depth — it's zone-integrated, not depth-resolved). `kind` ∈ `{"discharged", "frac_discharged", "release_rate"}`. |

Use `ensemble_from_surrogate`/`ensemble_from_high_fidelity` for the *same* depth from the two
model fidelities, to overlay and compare them (see `make_example_figures.py`'s clay-comparison
panel). Use `ensemble_from_leakage` for a zone-wide quantity that has no single depth, e.g. total
discharge out of the host rock over time.

### Naming a depth instead of giving a number: `depth`/`z` as a string

`depth` (`ensemble_from_surrogate`) and `z` (`ensemble_from_high_fidelity`) also accept a name
like `"Host_rock_top"`/`"Host_rock_bottom"` (a stratigraphy unit's top/bottom, see
`stratigraphy_plot.py`'s `resolve_depth()`/`boundary_key()`) instead of a metres value. This
matters whenever the *geometry itself* is uncertain (`Site.boundary_ranges` non-empty, see
`STRATIGRAPHY_PLOT.md`) — the boundary's real depth then varies per ensemble sample, so there is no
single numeric `depth`/`z` to ask for:

- `ensemble_from_surrogate`: the surrogate's `propagation.h5` names such a column directly in
  `axes/depth` (`axes/depth_z` is `NaN` for it, since its real depth isn't fixed); passing the same
  name matches it exactly. Pass `site=` as a fallback so the same name also works against a
  deterministic-geometry file, whose `axes/depth` entries are just the depths themselves,
  stringified — the name is resolved to a nominal number via `site` and matched as usual.
- `ensemble_from_high_fidelity`: pass `site=` (required for a name `z`). Each sample is
  interpolated to *its own* sampled boundary depth, read from this same ensemble file
  (`sampled_data/uncertain_parameters/geometry/rock_interface/<key>`) when that boundary is
  uncertain in it; otherwise every sample uses the nominal depth from `site`.

`Ensemble.depth` ends up `None` whenever the matched/interpolated-to depth has no single numeric
value (a zone-aggregated QoI, or a name whose real depth varies per sample) — `QuantityPanel.depth`
(below) is unaffected and can still be set to the same string, since a `Site` resolves it to one
nominal depth for the connector line and the title.

### Flooring and scaling raw high-fidelity values: `floor`/`signed`/`factor`

`ensemble_from_high_fidelity()`, `ensemble_from_leakage()` and `depth_profile_from_high_fidelity()`
(below) all take `site=None, floor=None, signed=None, factor=None`. A surrogate is trained on its
*floored and scaled* data (raw values this case's `surrogate.qoi.floor` calls insignificant are
zeroed, then multiplied by `surrogate.qoi.factor` -- see
`nuctrans_fastuq.surrogate.data_io.load_and_extract_data()`), so its predictions never show the tiny
near-zero noise a real high-fidelity run can, and are already in the scaled units `factor` converts
to. Comparing the two fairly means flooring *and* scaling the high-fidelity values the same way
before plotting them:

- `floor`/`signed` default to `site.surrogate_qoi["floor"]`/`["signed"]` (0.0/`True` -- no floor --
  without a `site`); pass them explicitly to override.
- `signed=True` (the default) floors `|value|`, zeroing small values of either sign; `signed=False`
  floors `value` directly, zeroing everything below it (including negative values).
- `factor` defaults to `site.surrogate_qoi["factor"] * site.analysis["qoi"]["factor"]` (1.0 for
  either/both when missing -- no `site`, no `analysis.qoi`, or a `qoi` dict with no `"factor"` key
  -- see `_resolve_factor`). Two factors multiply because some cases apply a *second*,
  display-only conversion on top of the surrogate's own training-time one -- e.g. `surrogate.qoi`
  converts raw flux into `mol/(m² year)` for training, then `analysis.qoi.factor` further converts
  that into a dimensionless "flux ratio" for display. Whatever the surrogate side ends up showing
  (its trained output already carries `surrogate.qoi.factor`, plus whatever `factor` you passed
  `ensemble_from_surrogate`/`depth_profile_from_surrogate` explicitly) is what the high-fidelity
  side needs to match -- if you instead pass your own `factor=` to the surrogate call (as in the
  "no analysis.qoi.factor" case, where there's nothing to apply beyond `1.0`), pass the matching
  value to the high-fidelity reader too rather than relying on this default.
- The floor is applied to the *raw* values read from the file first, then `factor` is applied to the
  result -- the same order `data_io.py` applies its own floor/factor in, and before any further
  processing (depth interpolation, zone selection, …).
- Pairing one of these readers with `ensemble_from_surrogate`/`depth_profile_from_surrogate` for the
  same case and passing `site=` to the high-fidelity side is normally all you need: the surrogate's
  own `output/mean` already carries its training-time `factor`, and the high-fidelity reader now
  lands in the same scaled units automatically.

## Building a `DepthProfile` (quantity vs. depth)

| Function | Reads from | Fixed by |
|---|---|---|
| `depth_profile_from_surrogate(path, time, factor=1.0, unit=None, label="Surrogate", color="#2ca02c", linestyle="-")` | The same surrogate grid file. | Nearest `time` in `axes/time`. |
| `depth_profile_from_high_fidelity(path, nuclide, quantity="concentration", time=0.0, site=None, floor=None, signed=None, factor=None, label="OGS", unit="", color="#1f4fd8", linestyle="--")` | The same high-fidelity file. | Nearest available time to `time` (years). |

These are the depth-axis counterparts of the two `ensemble_from_*` readers above — same files,
same `unit`/`label`/`color`/`linestyle` knobs (plus `site`/`floor`/`signed`/`factor` for the
high-fidelity one, same as `ensemble_from_high_fidelity()` above), just fixing a time instead of a
depth.

## Panel config: `QuantityPanel` (time-series) and `DepthPanel` (depth-profile)

`QuantityPanel` fields:

| Field | Default | Effect |
|---|---|---|
| `ensembles` | — | `list[Ensemble]` to overlay in this panel. |
| `depth` | `None` | The quantity's depth, used for the connector line to the stratigraphy and (via a `Site`'s `analysis["labels"]`) the auto title. A number, or a name like `"Host_rock_bottom"` — resolved to a number via the `site` passed to `plot_quantity()`/`compose_figure.make_figure()`. |
| `zone` | `None` | Unit name or `(z1, z2)` tuple, for a zone-integrated quantity (e.g. leakage) instead of a fixed depth. |
| `title` | `None` | Explicit title; `None` auto-generates one from `zone` or `depth`. |
| `ylabel` | `None` | Explicit y-axis label; `None` uses `"[unit]"` from the first ensemble that has one. |
| `xscale`, `yscale` | `"linear"`, `"linear"` | `"linear"`, `"log"`, or `"symlog"` (y only). |
| `threshold` | `None` | A value or list of values drawn as dashed horizontal lines. |
| `threshold_label` | `"Threshold"` | Legend label for the threshold line(s) (only the first gets a label). |
| `absolute` | `False` | Plot `\|values\|` — useful on a log y-scale when fluxes are negative (downward). |
| `quantiles` | `(5, 95)` | Percentiles bounding the shaded band. |
| `linthresh` | `None` | Linear-region half-width for `yscale="symlog"`; auto-picked from the data if `None`. |
| `xlim`, `ylim` | `None` | Explicit axis limits. |
| `legend` | `None` | `True`/`False` forces the legend on/off for this panel; `None` defers to `plot_quantity()`'s `legend` argument. |
| `extra` | `{}` | Free-form dict, not used by `plot_quantity()` itself (room for figure-building code to stash extra info per panel). |

`DepthPanel` fields: `profiles` (`list[DepthProfile]`), `z_limit`, `title`, `xlabel`, `xscale`,
`xlim`, `quantiles`, `legend` — the depth-axis analogues of the above. `z_limit` is the depth
window shown; when a `DepthPanel` is composed next to a stratigraphy column (via
`compose_figure.SiteBlock.depth_panels`), the stratigraphy's own `z_limit` wins instead, so the
two share one depth axis.

## `plot_quantity(ax, panel, site=None, show_xlabel=True, legend=True)`

Draws `panel`'s median line + quantile band (per ensemble) and threshold lines onto `ax`. `site`
(a `Site`) is only used to resolve `panel.depth` to a named label (`site.analysis["labels"]`) for
the auto-generated title, when `panel.title` is not set; without it, the title falls back to
`"z = ... m"` for a numeric `panel.depth`, or the name itself (underscores turned into spaces) for
a string one. `show_xlabel` toggles the `"Time [years]"` x-label (turn off for all but the bottom
panel in a stack). `legend` is the panel's default, overridden by `panel.legend` when that isn't
`None`.

## `plot_depth_profile(ax, panel, show_ylabel=True, legend=True)`

The depth-axis counterpart: median + quantile band per profile, x = quantity, y = depth.
`show_ylabel` toggles the `"z [m]"` y-label and its tick labels (turn off for all but the
leftmost panel in a row). `legend` behaves like `plot_quantity()`'s.

Both functions return the `ax` they drew into, for chaining.

## What you can do with it

- **Compare the surrogate and high-fidelity models at the same depth or time** — build one
  `Ensemble`/`DepthProfile` with each `*_from_surrogate`/`*_from_high_fidelity` reader and put
  both in the same panel's `ensembles`/`profiles` list. Pass `site=` to the high-fidelity reader
  so it floors and scales its raw values the same way the surrogate's training data was
  (`site.surrogate_qoi["floor"]`/`["signed"]`/`["factor"]`), rather than showing noise the
  surrogate never saw or comparing against it in the wrong units.
- **Plot zone-integrated leakage/discharge over time** with `ensemble_from_leakage`, choosing
  `kind` (`"discharged"`, `"frac_discharged"`, `"release_rate"`).
- **Overlay any number of series** in one panel, each with its own `color`/`linestyle`/`label` —
  the median + quantile band and legend entries are generated per series automatically.
- **Draw regulatory/threshold lines** with `threshold=value` or `threshold=[v1, v2, ...]`,
  labelled via `threshold_label`.
- **Switch axis scales**: `xscale`/`yscale` in `{"linear", "log", "symlog"}` — log masks
  non-positive values rather than erroring; symlog auto-picks a linear-region width you can
  override with `linthresh`.
- **Plot absolute values** (`absolute=True`) to show a signed, downward (negative) flux on a log
  scale.
- **Tighten or widen the uncertainty band** via `quantiles=(lo, hi)` (default 5–95%).
- **Fix axis limits** (`xlim`/`ylim` or `xlim`/`z_limit`) instead of relying on autoscaling.
- **Control the title**: set `panel.title` explicitly, or let it auto-derive from `zone`, from
  `depth` matched against a `Site`'s named depth labels, or from the profile's `time`.
- **Control legends per panel** (`panel.legend`) independently of the figure-wide default passed
  to `plot_quantity()`/`plot_depth_profile()` (and, when composing a whole figure, independently
  of `compose_figure.make_figure()`'s `quantity_legend="first"/"all"/"none"`).
- **Share a depth axis with a stratigraphy column** by using `DepthPanel`s inside a
  `compose_figure.SiteBlock`, rather than setting `z_limit` by hand.
- **Compose multiple time-series and/or depth panels into one figure**, with automatic
  depth/zone connector lines to the stratigraphy, via
  `sitefig.compose_figure.make_figure([SiteBlock(site, panels=[...])])` /
  `SiteBlock(site, depth_panels=[...])`, rather than calling `plot_quantity()`/
  `plot_depth_profile()` directly.
