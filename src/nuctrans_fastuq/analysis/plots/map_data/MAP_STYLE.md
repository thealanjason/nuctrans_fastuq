# Map styling

`map_data/categories.geojson` and `map_data/outlines.geojson` are an unmodified export of the
GIS data provided at <https://www.bge.de/de/endlagersuche/unterlagen/zwischenbericht-teilgebiete/>
(Teilgebiete, Stand 2026) and of the German state boundaries (BKG VG250), exported to GeoJSON in
lon/lat (EPSG:4326, all attribute fields). sitefig only supports lon/lat map data -- `load_map()`
checks each file's coordinate range and raises a clear error if it falls outside lon/lat (e.g. a
projected CRS such as UTM) instead of silently drawing a wrong map. Displaying lon/lat degrees
without distortion (1° of longitude covers less ground distance than 1° of latitude away from the
equator) is handled by `sitefig.map_plot.map_aspect()`, applied automatically wherever the map is
drawn.

The three map categories group features by their `Wirtsgestein` (host-rock type) attribute — an
attribute of the source dataset above, matching BGE's own top-level host-rock types (flat and
steeply-bedded rock salt are merged into one "Steinsalz" category; Tertiary and pre-Tertiary
claystone into one "Tongestein" category). The fill colours were chosen manually for visual
distinction and carry no information from the source data.

**`map_data/category_rules.csv`** is the single source of truth for the map's categories and
colours — a 3-column table (`label`, `filter`, `color`; a blank `filter` is the catch-all "Other"
rule), readable and editable directly, no Python needed. `label` is the English text drawn in the
legend; `filter` matches against the source data's (German) `Wirtsgestein` attribute values:

| Category (legend label) | Filter (on the source attributes) | Fill colour |
|---|---|---|
| Crystalline host rock | `Wirtsgestein = 'Kristallin'` | `#F5A184` |
| Host rock salt | `Wirtsgestein` starts with `Steinsalz` | `#CBE7F5` |
| Host rock clay | `Wirtsgestein = 'Tongestein'` | `#BCB2D4` |
| Other (unclassified) | any feature matching none of the above | `#E31A1C` |

Every outline (category boundaries and state borders) is drawn in `#232323` at 0.26 mm (set via
`build_style()`'s `edge_color`/`linewidth_mm`, not the CSV, since they're shared by every rule).

`load_map("map_data")` reads `category_rules.csv` directly (via `load_category_rules()` and
`build_style()`) each time it runs — there is no intermediate generated file to keep in sync, and
no QGIS project needed to reproduce the map. After editing the CSV, the next figure run picks it
up automatically.
