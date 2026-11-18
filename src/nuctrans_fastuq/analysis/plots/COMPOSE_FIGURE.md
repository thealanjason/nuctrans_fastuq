# Using `compose_figure.py`

`compose_figure.py` is the top-level assembler: it's the one module in `sitefig` that doesn't draw
a subplot of its own. Instead, it takes what `map_plot.py`, `stratigraphy_plot.py` and
`quantities_plot.py` already know how to draw — a map, one or more stratigraphy columns, and
time-series/depth-profile panels — and lays them out together into a single matplotlib `Figure`:
sizing every axes, merging shared stratigraphy columns, and drawing the dashed connector lines
between a map location/stratigraphy depth and its panels. Its one real entry point is
`make_figure()`.

## Quickstart

```python
from sitefig import SiteBlock, load_map, load_site, make_figure

map_data = load_map("map_data")
site = load_site("Case3_Rocksalt")

fig = make_figure([SiteBlock(site, location=(11.30, 52.80))], map_data=map_data)
fig.savefig("figure.png")
```

## The inputs: `Location`, `SiteBlock`, `Layout`

- **`Location`** — a map point: `lon`, `lat`, optional `label`, and `label_offset` (points; the
  label sits left of the dot when the x-offset is negative). Tags a `SiteBlock` so it gets a
  marker on the map and a connector line to its stratigraphy/panels. You can also just pass a
  bare `(lon, lat)` tuple as `SiteBlock.location` — it's auto-wrapped into a `Location` with the
  label `"location<i>"`.
- **`SiteBlock`** — one row of the figure: `site` (a `Site`, or `None` for a map-only/no-strat
  block) plus *either* `panels` (time-series `QuantityPanel`s, stacked below one another) *or*
  `depth_panels` (depth-profile `DepthPanel`s, placed side by side, sharing the stratigraphy's
  depth axis) — not both. Also carries `location`, `z_limit` (depth window for the stratigraphy),
  `host_category` (force the host-rock's map-colour category), `unit_labels` (display names for
  units), and `title` (column title override).
- **`Layout`** — pure sizing, in inches: `width`, `map_width`, `strat_width`, `panel_height`,
  `panel_gap`, `block_gap`, `strat_height`, `gap_after_map`, `gap_after_strat`,
  `strat_column_gap`, `margin` (left, top, right, bottom), `legend_height`, `quantity_legend_height`
  (for `quantity_legend="bottom"`/`"top"`), `quantity_legend_width` (for `"left"`/`"right"`),
  `map_legend_row`. Pass a customized `Layout()` to `make_figure(layout=...)` only when the
  defaults' auto-sizing isn't giving you what you want — most figures don't need one.

## `make_figure(blocks, ...)` parameters

| Parameter | Default | Effect |
|---|---|---|
| `blocks` | — | `list[SiteBlock]`, one per site/location. With quantity or depth panels, blocks stack vertically; without, their stratigraphy columns are placed side by side. |
| `map_data` | `None` | A `MapData` (e.g. from `load_map()`); also supplies host-rock colours even when `show_map=False`. |
| `show_map` | `True` | Whether to draw the map panel (ignored if `map_data` is `None`). |
| `show_stratigraphy` | `True` | Whether to draw stratigraphy columns for blocks that have a `site`. |
| `categories` | `None` | Subset of map categories to draw/size the map legend for; `None` draws all of them. |
| `location_lines` | `True` | Line from each tagged map location to its stratigraphy (or panels, if it has none). |
| `depth_lines` | `True` | Lines from the stratigraphy (at a panel's depth/zone) to each time-series panel — depth panels need none, they share the depth axis directly. |
| `lithology_legend` | `True` | Whether to draw the shared lithology legend below the stratigraphy columns. |
| `lithology_legend_ncol` | `None` | Columns in that legend; `None` uses the default (4 columns with quantity panels, else 3 with a map or 2 without). A smaller number wraps it onto more rows (`ceil(n_entries / lithology_legend_ncol)`) -- bump `Layout.legend_height` to match. |
| `lithology_mode` | `"dominant"` | `"dominant"` or `"split"`, passed through to `plot_stratigraphy()`/`lithology_handles()`. |
| `color_mode` | `None` | `"full"`/`"host_only"` forces that mode on every stratigraphy column; `None` auto-picks `"host_only"` when both a map and stratigraphy are drawn (so the host rock's map-category colour stands out against pattern-only non-host units), else `"full"`. |
| `merge_stratigraphy` | `True` | Blocks sharing the same site, `z_limit`, and host colour share one stratigraphy column instead of drawing a duplicate. |
| `quantity_legend` | `"first"` | `"first"` (only the first panel drawn gets a legend), `"all"`, `"none"`, or one shared, de-duplicated legend placed at `"bottom"`, `"top"`, `"left"` or `"right"` of the quantity panels. `"bottom"` auto-anchors just above `lithology_legend`'s *actual* rendered position (measured, not guessed) when one is shown -- `Layout.quantity_legend_height` then only needs to cover the gap to the panels above, usually small; without a lithology legend it anchors near the figure's own bottom margin instead, so `quantity_legend_height` needs to cover its full height (default `0.6` covers one row). `"top"` reserves the same `quantity_legend_height` above the panels (clearing their titles automatically). `"left"`/`"right"` reserve `Layout.quantity_legend_width` (default `1.6`) beside the panels instead, with the legend vertically centered and stacked in one column by default. |
| `quantity_legend_ncol` | `None` | Columns in the shared legend; `None` puts every entry in one row for `"bottom"`/`"top"`, or stacks them in one column for `"left"`/`"right"`. A smaller number (for `"bottom"`/`"top"`) wraps it onto more rows (`ceil(n_entries / quantity_legend_ncol)`); increase `Layout.quantity_legend_height` to match only when there's no lithology legend to anchor `"bottom"` against. |
| `map_extent` | `None` | `(xmin, ymin, xmax, ymax)` to crop the map to; `None` uses `map_data.bounds`. |
| `map_legend` | `True` | Whether to draw the map's own category legend. |
| `map_outline_scale` | `0.6` | Line-width scale for the map outlines (smaller than `plot_map()`'s own default of `1.0`, since the map is usually a small panel here). |
| `layout` | `None` | A `Layout` overriding the default sizes; `None` uses `Layout()`. |
| `repository_label` | `"Host unit"` | Legend label for the repository/host-unit marker. |

Returns the composed `matplotlib.figure.Figure`. Raises `ValueError` if there's nothing to plot
(no map, no stratigraphy, and no panels among the blocks), or if the figure is too narrow to fit
the quantity panels (widen it via `layout=Layout(width=...)`).

## What you can do with it

- **Draw just the map**: `make_figure([], map_data=map_data)` (or with `[SiteBlock(None, location=...)]`
  for a marker and no stratigraphy).
- **Draw just a stratigraphy column** (or several, side by side): `show_map=False`,
  `blocks=[SiteBlock(site), ...]`.
- **Draw a site's time-series panels stacked under its own stratigraphy column**:
  `SiteBlock(site, panels=[QuantityPanel(...), ...])`. `QuantityPanel.depth` accepts a name like
  `"Host_rock_bottom"` as well as a number — handy when `site`'s geometry is uncertain (see
  `STRATIGRAPHY_PLOT.md`), since the connector line and title then still land in the right place
  even though the boundary's real depth isn't fixed. When `site.boundary_ranges` isn't empty, its
  stratigraphy column also gets a hatched band at each uncertain boundary automatically, and the
  shared lithology legend gains a "boundary range" entry for it.
- **Draw depth profiles side by side, sharing the stratigraphy's depth axis**:
  `SiteBlock(site, depth_panels=[DepthPanel(...), ...])` — the panel's own `z_limit` is ignored in
  favour of the stratigraphy's, so they always agree.
- **Stack multiple sites** in one figure: pass multiple `SiteBlock`s in `blocks`.
- **Tag a block with a map location** so it gets a marker, a label, and a connector line into its
  stratigraphy/panels: `SiteBlock(site, location=(lon, lat))` or `location=Location(lon, lat, "my label")`.
- **Share one stratigraphy column across blocks** that are really the same site/depth-window/host
  colour (e.g. the same site shown at two map locations): leave `merge_stratigraphy=True` (the
  default); set it `False` to force separate columns.
- **Force or override the host-rock fill colour's map category** with `SiteBlock.host_category`,
  instead of relying on the category at the block's location (or the lithology → category
  fallback in `DEFAULT_HOST_CATEGORY`).
- **Crop a block's stratigraphy to a depth window**: `SiteBlock(site, z_limit=(-1000, -1600))` — the
  shared lithology legend automatically drops any lithology/boundary-range entry entirely outside
  that crop (each block's own `z_limit` is passed through to `lithology_handles()`), so a cropped
  column never ends up with legend entries for units you can't see.
- **Rename units for display** per block: `SiteBlock(site, unit_labels={...})`.
- **Draw only some map categories**, e.g. to focus on one host-rock type:
  `make_figure(blocks, map_data=map_data, categories=["Host rock salt"])`.
- **Crop the map to a region**: `map_extent=(xmin, ymin, xmax, ymax)`.
- **Control legends independently**: `map_legend`, `lithology_legend`, and `quantity_legend`
  (`"first"`/`"all"`/`"none"`/`"bottom"`/`"top"`/`"left"`/`"right"`) each toggle their own legend.
- **Collapse every panel's legend into one shared legend around the figure**: `quantity_legend` set
  to `"bottom"`, `"top"`, `"left"` or `"right"` -- instead of repeating (or picking just one of) the
  per-panel legends. `"bottom"`/`"top"` default to one row (set `quantity_legend_ncol=` to wrap it
  onto more); `"left"`/`"right"` default to one column instead. `"bottom"` stacks directly above the
  lithology legend when both are shown (measuring that legend's real height rather than assuming
  `Layout.legend_height`'s nominal size), so the two never end up with a large gap between them
  regardless of how many rows either one needs.
- **Turn off the connector lines** (map→stratigraphy or stratigraphy→panel) if they're too
  cluttered for a given layout: `location_lines=False`, `depth_lines=False`.
- **Show every lithology per unit instead of just the dominant one**: `lithology_mode="split"`
  (keep this consistent with any panel-level lithology display, there is none here but it affects
  both the columns and the shared legend together).
- **Override the figure's size or spacing** when the automatic sizing isn't right, via
  `layout=Layout(width=..., panel_height=..., margin=..., ...)`.
- **Relabel the repository/host-unit legend entry**: `repository_label="..."`.
- **Target a journal's print column width**: `NATURE_WIDTH_MM` (`{"1col": 88, "2col": 180}`, in mm)
  and `MM_PER_INCH` give the Nature-family "guide to preparing final artwork" widths —
  `layout=Layout(width=NATURE_WIDTH_MM["2col"] / MM_PER_INCH)` for a 2-column-wide figure. `FONT_RC`
  (the `font.size`/`axes.titlesize`/etc. dict applied via `plt.rc_context` inside `make_figure()`)
  is already within that guide's 5–7pt text-size window and sets `font.family` to
  `["Arial", "Helvetica", "DejaVu Sans"]` (first one installed wins); mutate `FONT_RC` before calling
  `make_figure()` to change it further. Save as PDF/SVG (`pdf.fonttype`/`svg.fonttype` are already
  set for embedded, editable text) rather than PNG, to satisfy the guide's vector-format
  requirement for line art.
