# Using `map_plot.py`

`map_plot.py` draws one lon/lat category map (e.g. host-rock type) with an optional outline
layer (e.g. state borders) onto a matplotlib `Axes`, with the longitude/latitude aspect ratio
corrected automatically so the map isn't visually stretched. It only draws — loading the GeoJSON
data and styling rules is `category_map.py`'s job (see `MAP_STYLE.md` for how `map_data/` itself
was produced).

Most of the time you won't call `plot_map()` yourself: `sitefig.compose_figure.make_figure()`
calls it internally whenever you pass it `map_data=...`, placing the map next to stratigraphy and
quantity panels. `plot_map()` is exported separately (`from sitefig import plot_map`) for when you
want just the map, on its own `Axes`.

## Quickstart

```python
import matplotlib.pyplot as plt
from sitefig import load_map, plot_map

map_data = load_map("map_data")  # reads map_data/category_rules.csv, categories.geojson, outlines.geojson

fig, ax = plt.subplots()
plot_map(ax, map_data)
fig.savefig("map.png")
```

## Loading the data (`category_map.py`)

`plot_map()` takes a `MapData`, which bundles the category polygons, the outline polygons, and
the `MapStyle` (colours/labels) used to draw them. You normally get one from:

```python
from sitefig import load_map

map_data = load_map("map_data")
```

- `load_map(map_dir, style=None, rules_csv=None, **build_style_kwargs)` reads
  `map_dir/categories.geojson` and `map_dir/outlines.geojson` (both must be lon/lat, EPSG:4326 —
  it raises a clear error if it detects a projected CRS such as UTM instead), and either takes a
  `MapStyle` you already built, or builds one from `rules_csv` (default
  `map_dir/category_rules.csv`).
- The CSV has three columns — `label`, `filter`, `color` — and is the single source of truth for
  category names/colours: a blank `filter` is the catch-all rule, and `filter` supports
  `"Field" = 'value'` and `"Field" [NOT] LIKE 'pattern'` conditions (optionally `AND`-joined)
  against the GeoJSON features' properties. Edit the CSV to change what's drawn; no code changes
  or regeneration step needed.
- `load_category_rules(path)` / `build_style(rules, ...)` are the two steps `load_map()` runs
  internally — call them yourself only if you need a `MapStyle` without loading any GeoJSON (e.g.
  to pass a pre-built style into `load_map(map_dir, style=...)`), or want to change
  `category_layer`/`outline_layer`/`edge_color`/`linewidth_mm` from their defaults.
- `MapData.category_at(lon, lat)` returns the category label covering a point (or `None`), used
  e.g. by `compose_figure.make_figure()` to colour a site's stratigraphy by the map category at
  its location.
- `MapData.bounds` is the `(xmin, ymin, xmax, ymax)` of the outline layer (or the category
  features, if there's no outline layer) — the default `extent` for `plot_map()`.

## `plot_map(ax, map_data, ...)` parameters

| Parameter | Default | Effect |
|---|---|---|
| `ax` | — | Matplotlib `Axes` to draw into. |
| `map_data` | — | A `MapData`, e.g. from `load_map()`. |
| `categories` | `None` | Subset of `map_data.style.labels` to draw; `None` draws all of them. Raises `ValueError` on an unknown label. |
| `extent` | `None` | `(xmin, ymin, xmax, ymax)` in lon/lat to crop/pad the view to; `None` uses `map_data.bounds`. |
| `outline_scale` | `1.0` | Scales the outline/edge linewidths (set in mm by `MapStyle.linewidth_mm`) — turn this down for small multi-panel figures so borders don't look too heavy. |
| `legend` | `True` | Whether to draw a legend of the categories actually drawn. |
| `legend_loc` | `"below"` | `"below"` centers a multi-column legend under the map; any other value is passed straight through as a matplotlib `loc` (e.g. `"upper right"`). |
| `pad` | `0.01` | Extra margin around the map, as a fraction of its width/height. |
| `legend_fontsize` | `6` | Legend text size, in points. |
| `legend_ncol` | `2` | Legend columns, only used when `legend_loc="below"`. |

It returns the same `ax`, for chaining.

## What you can do with it

- **Draw the whole map** with default styling: `plot_map(ax, map_data)`.
- **Draw only some categories** (the rest of the map is simply left blank, outlines still drawn):
  `plot_map(ax, map_data, categories=["Host rock salt"])`.
- **Crop to a region of interest**: `plot_map(ax, map_data, extent=(xmin, ymin, xmax, ymax))`.
- **Turn the legend off**, or move/reflow it: `legend=False`, or
  `legend_loc="upper right"`/`legend_ncol=3`.
- **Shrink outline linewidths** for a map panel that will be small in the final figure:
  `outline_scale=0.6` (this is what `compose_figure.make_figure()`'s `map_outline_scale` controls).
- **Change what counts as a category, or its colour**, without touching any Python — edit
  `category_rules.csv` and reload with `load_map()`.
- **Look up a category at a specific location** (for custom colouring logic, outside
  `compose_figure`): `map_data.category_at(lon, lat)`.
- **Combine the map with stratigraphy and quantity panels** in one composed figure — pass
  `map_data=` to `sitefig.compose_figure.make_figure()` rather than calling `plot_map()` yourself;
  it also handles connector lines from map locations to their stratigraphy/panels and the shared
  figure sizing (see `compose_figure.py`).
- **Reuse `map_aspect(bounds)`** directly if you're doing your own axes layout around a map (as
  `make_figure()` does, to reserve the right height for the map panel before drawing into it).
