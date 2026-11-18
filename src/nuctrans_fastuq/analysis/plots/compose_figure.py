from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D

from .map_plot import map_aspect, plot_map
from .quantities_plot import (
    DepthPanel,
    QuantityPanel,
    plot_depth_profile,
    plot_quantity,
)
from .stratigraphy_plot import (
    DEFAULT_HOST_CATEGORY,
    Site,
    lithology_handles,
    plot_stratigraphy,
    resolve_depth,
)

FONT_RC = {
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}

MM_PER_INCH = 25.4

# Layout(width=NATURE_WIDTH_MM["2col"] / MM_PER_INCH) for a submission-ready figure width.
NATURE_WIDTH_MM = {"1col": 88, "2col": 180}


@dataclass
class Location:
    """A map location (lon/lat), with an optional label, tagging a SiteBlock."""

    lon: float
    lat: float
    label: str | None = None
    label_offset: tuple = (-4, 5)  # points; text sits left of the dot for negative x


@dataclass
class SiteBlock:
    """One site: its stratigraphy and the quantity panels belonging to it, optionally tagged with a map location.

    panels: time-series panels (plotted vs time, stacked below one another).
    depth_panels: depth-profile panels (plotted vs depth, placed side by side, sharing the stratigraphy's depth
        axis): e.g. one panel per time step. A block uses either panels or depth_panels, not both.
    """

    site: Site | None = None
    panels: list[QuantityPanel] = field(default_factory=list)
    depth_panels: list[DepthPanel] = field(default_factory=list)
    location: Location | tuple | None = None
    z_limit: tuple | None = None
    host_category: str | None = None
    unit_labels: dict | None = None
    title: str | None = None


@dataclass
class Layout:
    """Sizes in inches."""

    width: float | None = None
    map_width: float | None = None
    strat_width: float = 0.6
    panel_height: float = 1.25
    panel_gap: float = 0.42
    block_gap: float = 0.72
    strat_height: float = 3.6
    gap_after_map: float = 0.8
    gap_after_strat: float = 1.25
    strat_column_gap: float = 0.85
    margin: tuple = (0.1, 0.3, 0.12, 0.42)  # left, top, right, bottom
    legend_height: float = 0.45
    quantity_legend_height: float = 0.6
    quantity_legend_width: float = 1.6
    map_legend_row: float = 0.14


def _fig_xy(fig, ax, xy):
    """``xy`` (in ``ax`` data coordinates) as a point in ``fig`` figure coordinates.

    Args:
        fig: Figure whose coordinate system the point is converted into.
        ax: Axes whose data coordinates ``xy`` is given in.
        xy: ``(x, y)`` point in ``ax`` data coordinates.

    Returns:
        ``(x, y)`` point in ``fig`` figure coordinates (0-1 range).
    """
    return fig.transFigure.inverted().transform(ax.transData.transform(xy))


def _fig_bbox(fig, bbox):
    """``bbox`` (in display/pixel coordinates, e.g. from get_window_extent()) as a Bbox in figure coordinates.

    Args:
        fig: Figure whose coordinate system the bbox is converted into.
        bbox: A ``matplotlib.transforms.Bbox`` in display/pixel coordinates.

    Returns:
        The same bbox, transformed into ``fig`` figure coordinates (0-1 range).
    """
    return bbox.transformed(fig.transFigure.inverted())


def _line(fig, p, q, **kw):
    """Draw a straight line from ``p`` to ``q`` (both in figure coordinates) directly onto ``fig``.

    Args:
        fig: Figure to draw the line onto (as a figure-level artist, not tied to one Axes).
        p: ``(x, y)`` start point, in ``fig`` figure coordinates.
        q: ``(x, y)`` end point, in ``fig`` figure coordinates.
        **kw: Extra keyword arguments passed through to ``matplotlib.lines.Line2D``
            (e.g. ``color``, ``linewidth``, ``linestyle``).
    """
    fig.add_artist(Line2D([p[0], q[0]], [p[1], q[1]], transform=fig.transFigure, **kw))


def make_figure(
    blocks,
    map_data=None,
    show_map=True,
    show_stratigraphy=True,
    categories=None,
    location_lines=True,
    depth_lines=True,
    lithology_legend=True,
    lithology_legend_ncol=None,
    lithology_mode="dominant",
    color_mode=None,
    merge_stratigraphy=True,
    quantity_legend="first",
    quantity_legend_ncol=None,
    map_extent=None,
    map_legend=True,
    map_outline_scale=0.6,
    layout=None,
    repository_label="Host unit",
):
    """Compose map | stratigraphy | quantity panels.

    Args:
        blocks: list of SiteBlock (one per site/location). With quantity or depth panels the blocks are
            stacked vertically; without, their stratigraphic columns are placed side by side.
        map_data: MapData; also used for host-rock colours when show_map=False.
        show_map: Whether to draw the map panel (ignored if ``map_data`` is ``None``).
        show_stratigraphy: Whether to draw stratigraphy columns for blocks that have a ``site``.
        categories: Subset of map categories to draw (and to size the map legend for); ``None`` uses all
            categories in ``map_data.style``.
        location_lines: Whether to draw a line from each tagged map location to its stratigraphy (or its
            panels, if it has no stratigraphy column).
        depth_lines: Whether to draw lines from the stratigraphy (at the panel's depth / zone) to each
            time-series quantity panel (depth panels need no line -- they share the stratigraphy's depth
            axis directly).
        lithology_legend: Whether to draw the shared lithology legend below the stratigraphy columns.
        lithology_legend_ncol: Columns in that legend; ``None`` uses the default (4 columns when
            there are quantity panels, else 3 with a map or 2 without).
        lithology_mode: Lithology rendering mode passed through to ``plot_stratigraphy``/``lithology_handles``
            (e.g. ``"dominant"``).
        color_mode: ``"full"``/``"host_only"`` forces that mode (see ``plot_stratigraphy``); ``None``
            picks ``"host_only"`` when both a map and stratigraphy are drawn, else ``"full"``.
        merge_stratigraphy: Whether blocks sharing the same site, z_limit and host colour share one column.
        quantity_legend: Which panels get a legend: ``'first'``, ``'all'``, ``'none'``, or a shared,
            de-duplicated legend placed at ``'bottom'``, ``'top'``, ``'left'`` or ``'right'`` of the
            quantity panels.
        quantity_legend_ncol: Number of columns in the shared legend; ``None`` puts every entry in
            one row for ``'bottom'``/``'top'``, or stacks them in a single column for ``'left'``/
            ``'right'``.
        map_extent: ``(xmin, ymin, xmax, ymax)`` to crop the map to; ``None`` uses ``map_data.bounds``.
        map_legend: Whether to draw the map's own category legend.
        map_outline_scale: Line-width scale for the map's outline, passed through to ``plot_map``.
        layout: ``Layout`` overriding the default sizes; ``None`` uses ``Layout()``.
        repository_label: Legend label for the repository/host unit, passed through to ``lithology_handles``.

    Returns:
        matplotlib.figure.Figure: The composed figure.

    Raises:
        ValueError: If there is nothing to plot (no map, no stratigraphy, no panels), or if the figure is
            too narrow to fit the quantity panels.
    """
    lay = layout or Layout()
    blocks = list(blocks)
    has_map = show_map and map_data is not None
    has_strat = show_stratigraphy and any(b.site is not None for b in blocks)
    has_quant = any(b.panels or b.depth_panels for b in blocks)
    if not (has_map or has_strat or has_quant):
        raise ValueError(
            "Nothing to plot: give map_data and/or blocks with a site, panels or depth_panels"
        )

    # each block's location, defaulted to Location(lon, lat, "location<i>") if given as a bare (lon, lat) tuple
    locs = []
    for i, b in enumerate(blocks):
        loc = b.location
        if loc is not None:
            if not isinstance(loc, Location):
                loc = Location(*loc)
            if loc.label is None:
                loc = Location(loc.lon, loc.lat, f"location{i + 1}", loc.label_offset)
        locs.append(loc)
    if location_lines and has_map and (has_strat or has_quant) and not any(locs):
        warnings.warn(
            "location_lines=True but no block has a location (lon, lat); no location lines drawn"
        )

    # each block's host-rock (fill colour, map category): from the map category at its location, else
    # block.host_category, else the DEFAULT_HOST_CATEGORY for its dominant lithology
    hosts = []
    for b, loc in zip(blocks, locs):
        cat = color = None
        if map_data is not None and b.site is not None:
            style = map_data.style
            cat = b.host_category
            if cat is None and loc is not None:
                cat = map_data.category_at(loc.lon, loc.lat)
                if cat is None:
                    warnings.warn(
                        f"{loc.label} ({loc.lon}, {loc.lat}) lies outside all map categories"
                    )
            if cat is None:
                lith = (b.site.host.lithologies or [None])[0]
                cat = DEFAULT_HOST_CATEGORY.get(lith)
            if cat is None or cat not in style.labels:
                cat = None
            else:
                color = style.color_of(cat)
        hosts.append((color, cat))
    if color_mode is None:
        color_mode = "host_only" if (has_map and has_strat) else "full"

    # stratigraphy columns: list of (block indices sharing one column)
    columns = []
    if has_strat:
        for i, b in enumerate(blocks):
            if b.site is None:
                continue
            key = (b.site.name, tuple(b.z_limit) if b.z_limit else None, hosts[i][0])
            match = next(
                (
                    c
                    for c in columns
                    if merge_stratigraphy and has_quant and c[0] == key
                ),
                None,
            )
            if match:
                match[1].append(i)
            else:
                columns.append((key, [i]))
    merged_all = (
        has_quant
        and len(columns) == 1
        and len(columns[0][1]) == len(blocks)
        and len(blocks) > 1
    )

    # ---- vertical sizes
    ml, mt, mr, mb = lay.margin
    legend_h = lay.legend_height if (has_strat and lithology_legend) else 0.0
    quant_legend_h = (
        lay.quantity_legend_height
        if (has_quant and quantity_legend == "bottom")
        else 0.0
    )
    quant_legend_h_top = (
        lay.quantity_legend_height if (has_quant and quantity_legend == "top") else 0.0
    )
    quant_legend_w = (
        lay.quantity_legend_width
        if (has_quant and quantity_legend in ("left", "right"))
        else 0.0
    )
    block_h = []
    for b in blocks:
        n = len(b.panels)
        block_h.append(
            n * lay.panel_height + (n - 1) * lay.panel_gap if n else lay.strat_height
        )
    if has_quant:
        content_h = sum(block_h) + lay.block_gap * (len(blocks) - 1)
    else:
        content_h = lay.strat_height if has_strat else 0.0

    # ---- horizontal sizes
    n_side = len(columns) if (has_strat and not has_quant) else (1 if has_strat else 0)
    strat_area_w = n_side * lay.strat_width + max(n_side - 1, 0) * lay.strat_column_gap
    if has_map:
        bounds = map_extent or map_data.bounds
        xmin, ymin, xmax, ymax = bounds
        aspect = (ymax - ymin) / (xmax - xmin) * map_aspect(bounds)
    lead = (
        0.0 if has_map else (0.55 if has_strat else 0.7)
    )  # room for tick labels of the leftmost axes
    if lay.width is None:
        if has_quant:
            width = 7.2
        elif has_map:
            width = (
                ml
                + mr
                + 2.9
                + (lay.gap_after_map + 0.5 + strat_area_w if has_strat else 0)
            )
        else:
            width = max(ml + mr + lead + strat_area_w, 2.4)
    else:
        width = lay.width
    if has_map:
        if has_strat or has_quant:
            map_w = lay.map_width or (
                0.38 * width
                if has_quant
                else width - ml - mr - lay.gap_after_map - 0.5 - strat_area_w
            )
        else:
            map_w = width - ml - mr
        n_cat = (
            len(categories) if categories is not None else len(map_data.style.labels)
        )
        map_leg_h = (-(-n_cat // 2) * lay.map_legend_row + 0.08) if map_legend else 0.0
        map_y0 = 0.1 + map_leg_h
        content_h = max(
            content_h, map_w * aspect + map_y0 - mb - legend_h - quant_legend_h
        )
    content_h = max(content_h, 2.6 if has_strat else 0.5)

    # stretch panels so blocks fill the content height
    if has_quant:
        fixed = sum(
            (len(b.panels) - 1) * lay.panel_gap for b in blocks if b.panels
        ) + lay.block_gap * (len(blocks) - 1)
        flex = sum(
            len(b.panels) * lay.panel_height if b.panels else lay.strat_height
            for b in blocks
        )
        scale = (content_h - fixed) / flex
        block_h = [
            (
                (
                    len(b.panels) * lay.panel_height * scale
                    + (len(b.panels) - 1) * lay.panel_gap
                )
                if b.panels
                else lay.strat_height * scale
            )
            for b in blocks
        ]
        panel_h = lay.panel_height * scale
    height = mb + legend_h + quant_legend_h + content_h + quant_legend_h_top + mt

    with plt.rc_context(FONT_RC):
        fig = plt.figure(figsize=(width, height))
        fx = lambda x: x / width  # noqa: E731
        fy = lambda y: y / height  # noqa: E731
        bottom = mb + legend_h + quant_legend_h
        x = ml + lead
        map_ax = None
        if has_map:
            map_ax = fig.add_axes(
                [fx(x), fy(map_y0), fx(map_w), fy(bottom + content_h - map_y0)]
            )
            plot_map(
                map_ax,
                map_data,
                categories=categories,
                extent=map_extent,
                outline_scale=map_outline_scale,
                legend=map_legend,
            )
            x += map_w + (lay.gap_after_map if has_strat else lay.gap_after_map + 0.35)

        # block vertical spans (top-down)
        spans, top = [], bottom + content_h
        for h in block_h if has_quant else [content_h] * len(blocks):
            spans.append((top - h, top))
            if has_quant:
                top -= h + lay.block_gap

        strat_axes = {}
        if has_strat:
            for ci, (key, members) in enumerate(columns):
                if has_quant:
                    y0 = min(spans[i][0] for i in members)
                    y1 = max(spans[i][1] for i in members)
                    cx = x
                else:
                    y0, y1 = bottom, bottom + content_h
                    cx = x + ci * (lay.strat_width + lay.strat_column_gap)
                ax = fig.add_axes([fx(cx), fy(y0), fx(lay.strat_width), fy(y1 - y0)])
                b = blocks[members[0]]
                plot_stratigraphy(
                    ax,
                    b.site,
                    z_limit=b.z_limit,
                    host_color=hosts[members[0]][0],
                    color_mode=color_mode,
                    lithology_mode=lithology_mode,
                    unit_labels=b.unit_labels,
                    ylabel="z [m]" if (ci == 0 or has_quant) else "",
                )
                if not has_quant or not merged_all:
                    ax.set_title(b.title or b.site.name, fontsize=6.5)
                for i in members:
                    strat_axes[i] = ax
            x += strat_area_w + (lay.gap_after_strat if has_quant else 0)

        quant_x0 = x
        # extra clearance beyond the legend's own reserved width, so a panel's y-axis tick labels
        # (rendered outside its own axes box, e.g. the leftmost of several depth panels sharing one
        # y-axis) never land back on top of the legend strip right next to it.
        side_legend_gap = 0.5 if quantity_legend in ("left", "right") else 0.0
        panel_x0 = (
            quant_x0 + quant_legend_w + side_legend_gap
            if quantity_legend == "left"
            else quant_x0
        )
        panel_axes = {}
        if has_quant:
            quant_w = width - mr - quant_x0 - quant_legend_w - side_legend_gap
            if quant_w < 1.5:
                raise ValueError(
                    f"Figure too narrow for quantity panels ({quant_w:.2f} in); increase Layout.width"
                )
            first = True
            for i, b in enumerate(blocks):
                axes = []
                if b.panels:
                    y = spans[i][1]
                    for k, p in enumerate(b.panels):
                        y -= panel_h
                        share = axes[0] if axes else None
                        ax = fig.add_axes(
                            [fx(panel_x0), fy(y), fx(quant_w), fy(panel_h)],
                            sharex=share,
                        )
                        last = k == len(b.panels) - 1
                        show_leg = quantity_legend == "all" or (
                            quantity_legend == "first" and first
                        )
                        plot_quantity(
                            ax, p, site=b.site, show_xlabel=last, legend=show_leg
                        )
                        if k == 0 and locs[i] is not None and len(blocks) > 1:
                            ax.set_title(
                                f"{locs[i].label}\n{ax.get_title()}", linespacing=1.3
                            )
                        if not last:
                            ax.tick_params(labelbottom=False)
                        first = False
                        axes.append(ax)
                        y -= lay.panel_gap
                elif b.depth_panels:
                    y0, y1 = spans[i]
                    sax = strat_axes.get(i)
                    n = len(b.depth_panels)
                    dp_w = (quant_w - (n - 1) * lay.panel_gap) / n
                    xk = panel_x0
                    for k, p in enumerate(b.depth_panels):
                        ax = fig.add_axes([fx(xk), fy(y0), fx(dp_w), fy(y1 - y0)])
                        show_leg = quantity_legend == "all" or (
                            quantity_legend == "first" and first
                        )
                        plot_depth_profile(ax, p, show_ylabel=k == 0, legend=show_leg)
                        if sax is not None:
                            ax.set_ylim(sax.get_ylim())
                        if k == 0 and locs[i] is not None and len(blocks) > 1:
                            ax.set_title(
                                f"{locs[i].label}\n{ax.get_title()}", linespacing=1.3
                            )
                        first = False
                        axes.append(ax)
                        xk += dp_w + lay.panel_gap
                panel_axes[i] = axes

        lithology_leg = None
        if has_strat and lithology_legend:
            strat_blocks = [b for b in blocks if b.site is not None]
            handles = lithology_handles(
                [b.site for b in strat_blocks],
                colored=color_mode == "full",
                repository_label=repository_label,
                lithology_mode=lithology_mode,
                z_limits=[b.z_limit for b in strat_blocks],
            )
            left = (
                fx(quant_x0 - lay.gap_after_strat - strat_area_w - 0.45)
                if has_quant
                else fx(x - strat_area_w - 0.45)
            )
            if not has_map:
                left = fx(ml)
            ncol = lithology_legend_ncol or (4 if has_quant else (3 if has_map else 2))
            lithology_leg = fig.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(left, fy(0.04)),
                ncol=min(len(handles), ncol),
                frameon=False,
                handlelength=2.0,
                handleheight=1.4,
                columnspacing=1.2,
                fontsize=6,
            )

        if has_quant and quantity_legend in ("bottom", "top", "left", "right"):
            handles, seen = [], set()
            for axes in panel_axes.values():
                for pax in axes:
                    h, l = pax.get_legend_handles_labels()
                    for hi, li in zip(h, l):
                        if li not in seen:
                            seen.add(li)
                            handles.append(hi)
            if handles:
                labels = [h.get_label() for h in handles]
                default_ncol = (
                    1 if quantity_legend in ("left", "right") else len(handles)
                )
                ncol = quantity_legend_ncol or default_ncol
                mid_y = fy((bottom + bottom + content_h) / 2)
                if quantity_legend == "bottom":
                    # anchor just above the lithology legend's own real rendered top (not the
                    # nominal legend_h band, which is usually much taller than its actual content)
                    # -- otherwise leftover slack between the two legends looks like an unwanted gap.
                    quant_leg_y0 = mb + legend_h + 0.04
                    if lithology_leg is not None:
                        fig.canvas.draw()
                        lith_top_in = (
                            lithology_leg.get_window_extent()
                            .transformed(fig.dpi_scale_trans.inverted())
                            .y1
                        )
                        quant_leg_y0 = lith_top_in + 0.1
                    fig.legend(
                        handles=handles,
                        labels=labels,
                        loc="lower center",
                        bbox_to_anchor=(fx(panel_x0 + quant_w / 2), fy(quant_leg_y0)),
                        ncol=ncol,
                        frameon=False,
                        fontsize=6,
                    )
                elif quantity_legend == "top":
                    # anchor above the real rendered top of the panels (titles included), not the
                    # nominal content-area edge -- otherwise the legend sits on top of the titles.
                    fig.canvas.draw()
                    renderer0 = fig.canvas.get_renderer()
                    to_in0 = fig.dpi_scale_trans.inverted()
                    panels_top_in = max(
                        (
                            pax.get_tightbbox(renderer0).transformed(to_in0).y1
                            for axes in panel_axes.values()
                            for pax in axes
                        ),
                        default=bottom + content_h,
                    )
                    fig.legend(
                        handles=handles,
                        labels=labels,
                        loc="lower center",
                        bbox_to_anchor=(
                            fx(panel_x0 + quant_w / 2),
                            fy(panels_top_in + 0.06),
                        ),
                        ncol=ncol,
                        frameon=False,
                        fontsize=6,
                    )
                elif quantity_legend == "left":
                    fig.legend(
                        handles=handles,
                        labels=labels,
                        loc="center",
                        bbox_to_anchor=(fx(quant_x0 + quant_legend_w / 2), mid_y),
                        ncol=ncol,
                        frameon=False,
                        fontsize=6,
                    )
                elif quantity_legend == "right":
                    fig.legend(
                        handles=handles,
                        labels=labels,
                        loc="center",
                        bbox_to_anchor=(fx(width - mr - quant_legend_w / 2), mid_y),
                        ncol=ncol,
                        frameon=False,
                        fontsize=6,
                    )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        if depth_lines and has_strat and has_quant:
            _draw_depth_lines(fig, renderer, blocks, strat_axes, panel_axes)
        if has_map:
            _draw_locations(
                fig,
                renderer,
                map_ax,
                blocks,
                locs,
                strat_axes,
                panel_axes,
                columns,
                draw_lines=location_lines and (has_strat or has_quant),
            )
    return fig


def _draw_depth_lines(fig, renderer, blocks, strat_axes, panel_axes):
    """Dashed connector lines from each block's stratigraphy column to its time-series panels, at the panel's
    depth or zone.

    Args:
        fig: The Figure to draw into.
        renderer: ``fig.canvas.get_renderer()``, for measuring already-drawn axes' pixel extents.
        blocks: The SiteBlocks being composed.
        strat_axes: ``{block index: stratigraphy Axes}``, as built in ``make_figure()``.
        panel_axes: ``{block index: [panel Axes, ...]}`` (time-series panels, in the same order as
            ``block.panels``).
    """
    style = dict(color="#7a7a7a", linewidth=0.6, linestyle=(0, (3, 2)))
    for i, b in enumerate(blocks):
        sax = strat_axes.get(i)
        if sax is None:
            continue
        zlo, zhi = sax.get_ylim()
        for p, pax in zip(b.panels, panel_axes.get(i, [])):
            if p.zone is not None:
                ztop, zbot = b.site.zone_range(p.zone)
                ztop, zbot = min(ztop, zhi), max(zbot, zlo)
                if ztop <= zbot:
                    warnings.warn(
                        f"zone {p.zone} is outside the stratigraphy z_limit; no connection line"
                    )
                    continue
                sax.plot(
                    [1.06, 1.06],
                    [zbot, ztop],
                    color="#262626",
                    linewidth=0.9,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                for z in (ztop, zbot):
                    sax.plot(
                        [1.0, 1.06],
                        [z, z],
                        color="#262626",
                        linewidth=0.9,
                        clip_on=False,
                    )
                a_top, a_bot = _fig_xy(fig, sax, (1.06, ztop)), _fig_xy(
                    fig, sax, (1.06, zbot)
                )
            elif p.depth is not None:
                pz = (
                    resolve_depth(b.site, p.depth)
                    if isinstance(p.depth, str)
                    else p.depth
                )
                if not zlo <= pz <= zhi:
                    warnings.warn(
                        f"depth {p.depth} is outside the stratigraphy z_limit; no connection line"
                    )
                    continue
                sax.plot(
                    [0, 1],
                    [pz, pz],
                    color="#262626",
                    linewidth=0.8,
                    linestyle=(0, (2, 1)),
                    zorder=4,
                )
                a_top = a_bot = _fig_xy(fig, sax, (1.0, pz))
            else:
                continue
            tb = _fig_bbox(fig, pax.get_tightbbox(renderer))
            ab = _fig_bbox(fig, pax.get_window_extent(renderer))
            xl = tb.x0 - 0.004
            _line(fig, a_top, (xl, ab.y1), **style)
            _line(fig, a_bot, (xl, ab.y0), **style)


def _draw_locations(
    fig, renderer, map_ax, blocks, locs, strat_axes, panel_axes, columns, draw_lines
):
    """Marker and label for each block's tagged location, plus (if ``draw_lines``) a connector line into its
    stratigraphy column (at its host-rock depth) or, lacking one, its panels.

    Args:
        fig: The Figure to draw into.
        renderer: ``fig.canvas.get_renderer()``, for measuring already-drawn axes' pixel extents.
        map_ax: The map Axes.
        blocks: The SiteBlocks being composed.
        locs: Each block's Location (or ``None``), in the same order as ``blocks`` (see ``make_figure()``'s
            ``locs``).
        strat_axes: ``{block index: stratigraphy Axes}``.
        panel_axes: ``{block index: [panel Axes, ...]}``, used when a block has no stratigraphy column.
        columns: The ``(key, member indices)`` list of merged stratigraphy columns, to spread out a shared
            column's multiple locations across its host-rock depth.
        draw_lines: Whether to draw the connector line (markers and labels are always drawn).
    """
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    members_of = {i: m for _, m in columns for i in m}
    for i, (b, loc) in enumerate(zip(blocks, locs)):
        if loc is None:
            continue
        mx, my = loc.lon, loc.lat
        x0, x1 = map_ax.get_xlim()
        y0, y1 = map_ax.get_ylim()
        if not (x0 <= mx <= x1 and y0 <= my <= y1):
            warnings.warn(f"{loc.label} is outside the map extent")
            continue
        map_ax.plot(mx, my, "o", color="black", markersize=3.5, zorder=6)
        map_ax.annotate(
            loc.label,
            (mx, my),
            xytext=loc.label_offset,
            textcoords="offset points",
            ha="right" if loc.label_offset[0] < 0 else "left",
            va="bottom" if loc.label_offset[1] >= 0 else "top",
            fontsize=7,
            zorder=7,
            path_effects=halo,
        )
        if not draw_lines:
            continue
        p = _fig_xy(fig, map_ax, (mx, my))
        sax = strat_axes.get(i)
        if sax is not None:
            zlo, zhi = sax.get_ylim()
            members = [m for m in members_of[i] if locs[m] is not None]
            host = b.site.host
            ht, hb = min(host.top, zhi), max(host.bottom, zlo)
            if len(members) > 1 and ht > hb:
                z = ht - (members.index(i) + 1) / (len(members) + 1) * (ht - hb)
            else:
                z = (
                    b.site.repository_depth
                    if b.site.repository_depth is not None
                    else (host.top + host.bottom) / 2
                )
                if not zlo <= z <= zhi:
                    z = (ht + hb) / 2 if ht > hb else (zlo + zhi) / 2
            tb = _fig_bbox(fig, sax.get_tightbbox(renderer))
            q = (tb.x0 - 0.004, _fig_xy(fig, sax, (0, z))[1])
        elif panel_axes.get(i):
            axes = panel_axes[i]
            tb = _fig_bbox(fig, axes[0].get_tightbbox(renderer))
            top = _fig_bbox(fig, axes[0].get_window_extent(renderer)).y1
            bot = _fig_bbox(fig, axes[-1].get_window_extent(renderer)).y0
            q = (tb.x0 - 0.004, (top + bot) / 2)
        else:
            continue
        _line(fig, p, q, color="black", linewidth=0.9)
