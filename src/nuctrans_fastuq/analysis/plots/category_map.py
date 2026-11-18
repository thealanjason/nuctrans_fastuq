from __future__ import annotations

import csv
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.strtree import STRtree

# Category rules (styling): filters


def compile_filter(expression):
    """Compile a filter (e.g. ``"Wirtsgestein" = 'Kristallin'``) into ``f(properties) -> bool``.

    Supports ``"Field" = 'value'`` and ``"Field" [NOT] LIKE 'pattern'`` (``%``/``_`` wildcards) conditions,
    optionally joined by ``AND`` -- the only syntax (modeled on QGIS's rule-filter expressions)
    category_rules.csv actually uses.
    """
    cond_re = re.compile(
        r"\"(?P<field>[^\"]+)\"\s*(?P<op>=|NOT\s+LIKE|LIKE)\s*'(?P<value>(?:[^']|'')*)'",
        re.IGNORECASE,
    )
    conditions = []
    for part in re.split(r"\bAND\b", expression, flags=re.IGNORECASE):
        m = cond_re.fullmatch(part.strip())
        if not m:
            raise ValueError(
                f"Unsupported filter condition {part.strip()!r} (in {expression!r})"
            )
        field, op, value = m["field"], m["op"].upper(), m["value"].replace("''", "'")
        if op == "=":
            conditions.append(lambda p, field=field, value=value: p.get(field) == value)
        else:
            rx = re.compile(
                "".join(
                    ".*" if c == "%" else "." if c == "_" else re.escape(c)
                    for c in value
                )
                + r"\Z",
                re.DOTALL,
            )
            negate = op.startswith("NOT")
            conditions.append(
                lambda p, field=field, rx=rx, negate=negate: (
                    p.get(field) is not None and bool(rx.match(p[field]))
                )
                != negate
            )
    return lambda p: all(c(p) for c in conditions)


# style


@dataclass
class MapRule:
    label: str
    filter: str | None
    color: str  # hex, e.g. "#F5A184"
    is_else: bool = False


@dataclass
class MapStyle:
    """Styling of the category (polygon) layer and the outline layer. Map geometry is always lon/lat
    (EPSG:4326).
    """

    category_layer: str
    category_field_rules: list[MapRule]
    outline_layer: str | None = None
    edge_color: str = "#232323"
    linewidth_mm: float = 0.26

    @property
    def labels(self):
        return [r.label for r in self.category_field_rules]

    def color_of(self, label):
        for r in self.category_field_rules:
            if r.label == label:
                return r.color
        raise KeyError(f"Unknown map category {label!r}; available: {self.labels}")


EDGE_COLOR_HEX = "#232323"
DEFAULT_LINEWIDTH_MM = 0.26


def load_category_rules(path):
    """Load a (label, filter, color) rule table from a CSV with columns label, filter, color."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            (row["label"], row["filter"] or None, row["color"])
            for row in csv.DictReader(f)
        ]


# map geometry from exported GeoJSON
def build_style(
    rules,
    category_layer="Geometriekorrigierte_Teilgebiete_2026",
    outline_layer="vg250:vg250_lan",
    edge_color=EDGE_COLOR_HEX,
    linewidth_mm=DEFAULT_LINEWIDTH_MM,
):
    """Build a MapStyle from a (label, filter, color) rule table (e.g. from load_category_rules()).

    Pass ``outline_layer=None`` to omit the state-outline styling.
    """
    cat_rules = [
        MapRule(label, filt, color, is_else=filt is None)
        for label, filt, color in rules
    ]
    return MapStyle(category_layer, cat_rules, outline_layer, edge_color, linewidth_mm)


# data
@dataclass
class MapData:
    """Category polygons (with their category label) and outline polygons, in lon/lat (EPSG:4326)."""

    style: MapStyle
    features: list  # (category label or None, shapely geometry, properties)
    outlines: list  # shapely geometries
    _tree: STRtree | None = field(default=None, repr=False)

    def category_at(self, lon, lat):
        """Map category label at a geographic location, or None if outside all polygons."""
        if self._tree is None:
            self._tree = STRtree([g for _, g, _ in self.features])
        pt = Point(lon, lat)
        hits = [i for i in self._tree.query(pt) if self.features[i][1].covers(pt)]
        return self.features[min(hits)][0] if hits else None

    @property
    def bounds(self):
        geoms = self.outlines or [g for _, g, _ in self.features]
        xs = [g.bounds for g in geoms]
        return (
            min(b[0] for b in xs),
            min(b[1] for b in xs),
            max(b[2] for b in xs),
            max(b[3] for b in xs),
        )


def _read_geojson(path):
    """(properties, geometry) pairs; raises if coordinates are outside the lon/lat range."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = [
        (f.get("properties") or {}, shape(f["geometry"]))
        for f in d["features"]
        if f.get("geometry")
    ]
    if feats:
        bounds = [g.bounds for _, g in feats]
        xmin, ymin = min(b[0] for b in bounds), min(b[1] for b in bounds)
        xmax, ymax = max(b[2] for b in bounds), max(b[3] for b in bounds)
        if not (-180 <= xmin and xmax <= 180 and -90 <= ymin and ymax <= 90):
            raise ValueError(
                f"{path} coordinates are outside the lon/lat range (x in [{xmin:.3g}, {xmax:.3g}], y in "
                f"[{ymin:.3g}, {ymax:.3g}]); we only support GeoJSON exported in lon/lat (EPSG:4326), not "
                "a projected CRS such as UTM."
            )
    return feats


def load_map(map_dir, style=None, rules_csv=None, **build_style_kwargs):
    """Load exported GeoJSON layers and styling from ``map_dir``.

    ``style`` may be an already-built ``MapStyle``; otherwise one is built with ``build_style()`` from a
    category-rules CSV (``rules_csv``, default ``map_dir/category_rules.csv``; see ``load_category_rules()``).
    """
    map_dir = Path(map_dir)
    if style is None:
        rule_rows = load_category_rules(rules_csv or map_dir / "category_rules.csv")
        style = build_style(rule_rows, **build_style_kwargs)
    rules = [
        (
            r,
            (
                None
                if r.is_else
                else compile_filter(r.filter) if r.filter else (lambda p: True)
            ),
        )
        for r in style.category_field_rules
    ]
    else_rule = next((r for r in style.category_field_rules if r.is_else), None)
    features, unmatched = [], 0
    for props, geom in _read_geojson(map_dir / "categories.geojson"):
        label = next((r.label for r, fn in rules if fn is not None and fn(props)), None)
        if label is None:
            label = else_rule.label if else_rule else None
            unmatched += label is None
        features.append((label, geom, props))
    if unmatched:
        warnings.warn(
            f"{unmatched} map features match no category rule and will not be drawn"
        )
    outline_path = map_dir / "outlines.geojson"
    outlines = (
        [g for _, g in _read_geojson(outline_path)] if outline_path.exists() else []
    )
    return MapData(style, features, outlines)
