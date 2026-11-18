from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

# points-per-millimetre for matplotlib's linewidth
MM_TO_PT = 72.0 / 25.4


def map_aspect(bounds):
    """Display ratio of 1 y-unit to 1 x-unit (matplotlib's ``set_aspect``) that keeps a lon/lat map undistorted.

    1 degree of longitude covers less ground distance than 1 degree of latitude away from the equator (by
    cos(latitude)), so x must be compressed relative to y by that same factor: exact at the map's mean
    latitude.
    """
    _, ymin, _, ymax = bounds
    return 1.0 / np.cos(np.radians((ymin + ymax) / 2))


def _polygons(geom):
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [p for g in geom.geoms for p in _polygons(g)]
    return []


def geoms_to_path(geoms):
    """One compound path (exteriors + holes) for a list of shapely polygons/multipolygons."""
    verts, codes = [], []
    for g in geoms:
        for poly in _polygons(g):
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.coords)[:, :2]
                if len(xy) < 3:
                    continue
                verts.append(xy)
                c = np.full(len(xy), MplPath.LINETO, dtype=np.uint8)
                c[0], c[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
                codes.append(c)
    if not verts:
        return None
    return MplPath(np.concatenate(verts), np.concatenate(codes))


def plot_map(
    ax,
    map_data,
    categories=None,
    extent=None,
    outline_scale=1.0,
    legend=True,
    legend_loc="below",
    pad=0.01,
    legend_fontsize=6,
    legend_ncol=2,
):
    """Draw selected categories and the outline layer on top.

    ax: matplotlib Axes to draw into.
    map_data: a MapData, e.g. from load_map().
    categories: category labels to draw, in map_data.style.labels; None draws all of them.
    extent: (xmin, ymin, xmax, ymax) in lon/lat; defaults to the outline layer bounds.
    outline_scale: scales the outline widths (mm) for small figure panels.
    legend: whether to draw a legend of the drawn categories.
    legend_loc: "below" centers the legend under the map; any other value is a matplotlib legend loc.
    pad: fraction of the extent's width/height added as margin around the map.
    legend_fontsize: legend text size, in points.
    legend_ncol: number of legend columns when legend_loc="below".
    """
    style = map_data.style
    available = style.labels
    selected = available if categories is None else list(categories)
    unknown = [c for c in selected if c not in available]
    if unknown:
        raise ValueError(f"Unknown map categories {unknown}; available: {available}")

    linewidth = style.linewidth_mm * MM_TO_PT * outline_scale
    handles = []
    for rule in style.category_field_rules:
        if rule.label not in selected:
            continue
        path = geoms_to_path(
            [g for lab, g, _ in map_data.features if lab == rule.label]
        )
        if path is None:
            continue
        ax.add_patch(
            PathPatch(
                path,
                facecolor=rule.color,
                edgecolor=style.edge_color,
                linewidth=linewidth,
                zorder=2,
            )
        )
        handles.append(
            Patch(
                facecolor=rule.color,
                edgecolor=style.edge_color,
                linewidth=0.5,
                label=rule.label,
            )
        )

    if map_data.outlines and style.outline_layer:
        ax.add_patch(
            PathPatch(
                geoms_to_path(map_data.outlines),
                facecolor="none",
                edgecolor=style.edge_color,
                linewidth=linewidth,
                zorder=3,
            )
        )

    bounds = extent or map_data.bounds
    xmin, ymin, xmax, ymax = bounds
    dx, dy = (xmax - xmin) * pad, (ymax - ymin) * pad
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)
    ax.set_aspect(map_aspect(bounds))
    ax.set_axis_off()
    if legend and handles:
        where = (
            dict(loc="upper center", bbox_to_anchor=(0.5, -0.01))
            if legend_loc == "below"
            else dict(loc=legend_loc)
        )
        ax.legend(
            handles=handles,
            fontsize=legend_fontsize,
            frameon=False,
            handlelength=1.4,
            handleheight=1.0,
            borderaxespad=0.0,
            ncol=legend_ncol if legend_loc == "below" else 1,
            columnspacing=1.0,
            **where,
        )
    return ax
