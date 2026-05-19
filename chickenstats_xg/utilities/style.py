"""
Visualization style system — yellowbrick palette + chickenstats aesthetic.

Blends the yellowbrick color system (palettes, sequences, heatmap cmaps) with
the chickenstats matplotlib style (dimgray text, no top/right spines, bold
titles/labels) and adds NHL team color utilities.

Named colors
------------
YB_KEY / LINE_COLOR     Very dark grey (#111111) — structural marks only
                        (confusion matrix diagonals, reference lines).
                        NOT used for text/ticks/axes; those use dimgray.

Style functions
---------------
set_style(theme)        Apply the merged aesthetic globally.
                        theme="light" (default) or "dark".

Color functions
---------------
color_palette()         Named discrete palette → list[hex]
color_sequence()        Named sequential palette → ListedColormap
find_text_color()       WCAG-contrast text color for a background hex

NHL team utilities
------------------
team_palette()          [primary, secondary, neutral] hex list for a team
team_colormap()         Sequential ListedColormap: white → team primary
team_diverging_colormap() Diverging: secondary → neutral → primary
TEAM_COLORS             Raw dict of all team color triplets
"""

from __future__ import annotations

from typing import Literal

import matplotlib as mpl
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

# ---------------------------------------------------------------------------
# Named colors
# ---------------------------------------------------------------------------

YB_KEY = "#111111"  # structural emphasis only (diagonals, borders)
LINE_COLOR = YB_KEY  # reference / best-fit lines
DIMGRAY = "dimgray"  # axes chrome: text, ticks, labels, spines

# ---------------------------------------------------------------------------
# Discrete palettes
# ---------------------------------------------------------------------------

PALETTES: dict[str, list[str]] = {
    "yellowbrick": ["#0272a2", "#9fc377", "#ca0b03", "#a50258", "#d7c703", "#88cada"],
    "accent": ["#386cb0", "#7fc97f", "#f0027f", "#beaed4", "#ffff99", "#fdc086"],
    "dark": ["#7570b3", "#66a61e", "#d95f02", "#e7298a", "#e6ab02", "#1b9e77"],
    "pastel": ["#cbd5e8", "#b3e2cd", "#fdcdac", "#f4cae4", "#fff2ae", "#e6f5c9"],
    "bold": ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ffff33", "#ff7f00"],
    "muted": ["#80b1d3", "#8dd3c7", "#fb8072", "#bebada", "#ffffb3", "#fdb462"],
    "colorblind": ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"],
    "flatui": ["#34495e", "#2ecc71", "#e74c3c", "#9b59b6", "#f4d03f", "#3498db"],
    "sns_deep": ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"],
    "sns_muted": ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"],
    "sns_pastel": ["#92C6FF", "#97F0AA", "#FF9F9A", "#D0BBFF", "#FFFEA3", "#B0E0E6"],
    "sns_bright": ["#003FFF", "#03ED3A", "#E8000B", "#8A2BE2", "#FFC400", "#00D7FF"],
    "sns_dark": ["#001C7F", "#017517", "#8C0900", "#7600A1", "#B8860B", "#006374"],
    "neural_paint": ["#167192", "#6e7548", "#c5a2ab", "#00ccff", "#de78ae", "#ffcc99", "#3d3f42", "#ffffcc"],
    "set1": ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ffff33", "#ff7f00", "#a65628", "#f781bf", "#999999"],
    "paired": [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
        "#fdbf6f",
        "#ff7f00",
    ],
}

# ---------------------------------------------------------------------------
# Sequential / diverging sequences (ColorBrewer)
# ---------------------------------------------------------------------------

SEQUENCES: dict[str, dict[int, list[str]]] = {
    "YlOrRd": {
        3: ["#ffeda0", "#feb24c", "#f03b20"],
        4: ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
        5: ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        6: ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
        7: ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"],
        8: ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"],
        9: ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"],
    },
    "RdPu": {
        3: ["#fde0dd", "#fa9fb5", "#c51b8a"],
        4: ["#feebe2", "#fbb4b9", "#f768a1", "#ae017e"],
        5: ["#feebe2", "#fbb4b9", "#f768a1", "#c51b8a", "#7a0177"],
        6: ["#feebe2", "#fcc5c0", "#fa9fb5", "#f768a1", "#c51b8a", "#7a0177"],
        7: ["#feebe2", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177"],
        8: ["#fff7f3", "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177"],
        9: ["#fff7f3", "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177", "#49006a"],
    },
    "Blues": {
        3: ["#deebf7", "#9ecae1", "#3182bd"],
        5: ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
        9: ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
    },
    "RdBu": {
        9: ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"],
    },
    "Greys": {
        9: ["#ffffff", "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525", "#000000"],
    },
}

DEFAULT_SEQUENCE = "RdBu"

# ---------------------------------------------------------------------------
# rcParam aesthetics — merged YB palette + chickenstats chrome
# ---------------------------------------------------------------------------
#
# Design rationale:
#   - dimgray (#696969) for all text/ticks/labels/spines: softer than YB's
#     .15 (#262626), avoiding the heavy feel of near-black chrome.
#   - No top/right spines (chickenstats): removes visual noise, focuses eye
#     on data. Standard in modern publication-quality charts.
#   - Bold title and label weights (chickenstats): adds hierarchy for free.
#   - Light grid (.8, YB): kept for ROC/PR line charts; subtle enough on
#     heatmaps/bars that it doesn't distract.
#   - YB palette as default color cycle: distinctive and recognizable.

_LIGHT_STYLE: dict = {
    # Figure / axes background
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "axes.facecolor": "white",
    # Spines — remove top and right (chickenstats); keep left and bottom dimgray
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": DIMGRAY,
    "axes.linewidth": 1.25,
    # Grid — off by default (matches seaborn "white" / 0_1_4 look)
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": ".8",
    "grid.linestyle": "-",
    # Text and chrome — dimgray throughout (chickenstats)
    "text.color": DIMGRAY,
    "axes.labelcolor": DIMGRAY,
    "axes.titlecolor": DIMGRAY,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "lines.color": DIMGRAY,
    "xtick.color": DIMGRAY,
    "xtick.labelcolor": DIMGRAY,
    "ytick.color": DIMGRAY,
    "ytick.labelcolor": DIMGRAY,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    # Legend
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    # Misc
    "lines.solid_capstyle": "round",
    "image.cmap": "Greys",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Arial", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
}

_DARK_STYLE: dict = {
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "axes.facecolor": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "white",
    "axes.linewidth": 1.25,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": ".25",
    "grid.linestyle": "-",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "lines.color": "white",
    "xtick.color": "white",
    "xtick.labelcolor": "white",
    "ytick.color": "white",
    "ytick.labelcolor": "white",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    "patch.edgecolor": "white",
    "lines.solid_capstyle": "round",
    "image.cmap": "Greys",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Arial", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
}

# chickenstats dark palette (desaturated Brewer-ish set)
_DARK_PALETTE = [
    "#8dd3c7",
    "#feffb3",
    "#bfbbd9",
    "#fa8174",
    "#81b1d2",
    "#fdb462",
    "#b3de69",
    "#bc82bd",
    "#ccebc4",
    "#ffed6f",
]

_CONTEXT: dict = {
    "font.size": 12,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "grid.linewidth": 1,
    "lines.linewidth": 1.75,
    "patch.linewidth": 0.3,
    "lines.markersize": 7,
    "lines.markeredgewidth": 0,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
}


def set_style(
    theme: Literal["light", "dark"] = "light",
    palette: str | None = None,
    font_scale: float = 1.0,
    rc: dict | None = None,
) -> None:
    """Apply the merged chickenstats + yellowbrick aesthetic globally.

    Parameters
    ----------
    theme:
        ``"light"`` (default) — white background, dimgray chrome, YB palette.
        ``"dark"``            — black background, white chrome, desaturated palette.
    palette:
        Override the default color cycle. Named palette from PALETTES or a
        list of hex strings. Defaults to ``"yellowbrick"`` for light theme and
        the chickenstats dark palette for dark theme.
    font_scale:
        Multiplicative scale applied to all font sizes.
    rc:
        Additional rcParam overrides applied after all other settings.
    """
    from cycler import cycler

    base = _LIGHT_STYLE if theme == "light" else _DARK_STYLE
    params = {**base, **_CONTEXT}

    font_keys = [
        "axes.labelsize",
        "axes.titlesize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "font.size",
    ]
    for k in font_keys:
        params[k] = params[k] * font_scale

    if rc:
        params.update(rc)

    mpl.rcParams.update(params)

    # Color cycle
    if palette is not None:
        colors = color_palette(palette)
    elif theme == "dark":
        colors = _DARK_PALETTE
    else:
        colors = color_palette("yellowbrick")

    hex_colors = [c if isinstance(c, str) else mpl.colors.to_hex(c) for c in colors]
    mpl.rcParams["axes.prop_cycle"] = cycler("color", hex_colors)
    mpl.rcParams["patch.facecolor"] = hex_colors[0]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def color_palette(palette: str | list | None = "yellowbrick", n_colors: int | None = None) -> list[str]:
    """
    Return a list of hex color strings for the named palette.

    Parameters
    ----------
    palette:
        Name from PALETTES, or an explicit list of hex/RGB colors, or None
        to return the current matplotlib color cycle.
    n_colors:
        Number of colors to return. Cycles the palette if more are requested.
    """
    if palette is None:
        raw = [mpl.colors.to_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
        n = n_colors or len(raw)
        return [mpl.colors.to_hex(c) for c in [next(cycle(raw)) for _ in range(n)]]

    if isinstance(palette, (list, tuple)):
        raw = [mpl.colors.to_hex(c) for c in palette]
    else:
        key = palette.lower()
        match = next((k for k in PALETTES if k.lower() == key), None)
        if match is None:
            raise ValueError(f"'{palette}' is not a recognised palette. Choose from: {list(PALETTES)}")
        raw = PALETTES[match]

    n = n_colors or len(raw)
    pal_cycle = cycle(raw)
    return [mpl.colors.to_hex(mpl.colors.to_rgb(next(pal_cycle))) for _ in range(n)]


def color_sequence(palette: str | None = None, n_colors: int | None = None) -> mplcol.ListedColormap:
    """
    Return a matplotlib ListedColormap from a named sequential/diverging palette.

    Parameters
    ----------
    palette:
        Name from SEQUENCES, or None to use the default (RdBu).
    n_colors:
        Exact step count. If None, the largest available step count is used.
    """
    name = palette or DEFAULT_SEQUENCE
    key = next((k for k in SEQUENCES if k.lower() == name.lower()), None)
    if key is None:
        # Fall back to matplotlib built-in cmap
        return mpl.colormaps.get_cmap(name)
    steps = SEQUENCES[key]
    n = n_colors if n_colors in steps else max(steps.keys())
    return mplcol.ListedColormap(steps[n], name=key, N=n)


def find_text_color(bg_hex: str, light: str = "white", dark: str = DIMGRAY) -> str:
    """
    Return light or dark text color for readability against *bg_hex*.

    Uses the WCAG relative luminance formula. *dark* defaults to dimgray
    rather than pure black, consistent with the chickenstats chrome style.
    """
    r, g, b = mpl.colors.to_rgb(bg_hex)

    def linearise(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * linearise(r) + 0.7152 * linearise(g) + 0.0722 * linearise(b)
    return light if lum < 0.179 else dark


# ---------------------------------------------------------------------------
# NHL team color utilities
# ---------------------------------------------------------------------------
#
# Source: chickenstats.chicken_nhl.team.TEAM_COLORS
# Each entry is {"GOAL": hex, "SHOT": hex, "MISS": hex} where:
#   GOAL = primary accent (goals, primary data series)
#   SHOT = secondary color (shots, secondary data series)
#   MISS = neutral gray shared across all teams (#D3D3D3)
#
# Imported at call time to avoid a hard import-time dependency; callers that
# only use palettes/sequences/style need not have chickenstats installed.


def _load_team_colors() -> dict[str, dict[str, str]]:
    """Lazy-load TEAM_COLORS from chickenstats, fall back to empty dict."""
    try:
        from chickenstats.chicken_nhl.team import TEAM_COLORS as _TC

        return _TC  # ty: ignore[return-value]
    except ImportError:
        return {}


def team_palette(team_code: str) -> list[str]:
    """
    Return ``[primary, secondary, neutral]`` hex strings for an NHL team.

    Parameters
    ----------
    team_code:
        Three-letter NHL code, e.g. ``"TOR"``, ``"NSH"``, ``"VGK"``.

    Returns
    -------
    list[str]
        ``[GOAL color, SHOT color, MISS color]`` — primary, secondary, neutral.

    Examples
    --------
    >>> from chickenstats_xg.utilities.style import team_palette
    >>> team_palette("NSH")
    ['#FFB81C', '#041E42', '#D3D3D3']
    """
    tc = _load_team_colors()
    code = team_code.upper()
    if code not in tc:
        raise ValueError(f"'{team_code}' is not a recognised NHL team code.")
    colors = tc[code]
    return [colors["GOAL"], colors["SHOT"], colors["MISS"]]


def team_colormap(
    team_code: str,
    n: int = 256,
    from_color: str = "white",
) -> mplcol.LinearSegmentedColormap:
    """
    Sequential colormap: *from_color* → team primary color.

    Useful for single-team heatmaps (shot density, xG maps).

    Parameters
    ----------
    team_code:
        Three-letter NHL code.
    n:
        Number of colormap steps.
    from_color:
        Starting color (light end). Defaults to ``"white"``.

    Examples
    --------
    >>> cmap = team_colormap("TOR")
    >>> ax.imshow(data, cmap=cmap)
    """
    primary = team_palette(team_code)[0]
    return mplcol.LinearSegmentedColormap.from_list(
        f"{team_code.upper()}_seq",
        [from_color, primary],
        N=n,
    )


def team_diverging_colormap(
    team_code: str,
    n: int = 256,
    mid_color: str = "#f5f5f5",
) -> mplcol.LinearSegmentedColormap:
    """
    Diverging colormap: team secondary → *mid_color* → team primary.

    Useful for differential maps (e.g. xG for vs against, shot share above/below 50%).

    Parameters
    ----------
    team_code:
        Three-letter NHL code.
    n:
        Number of colormap steps.
    mid_color:
        Neutral midpoint color. Defaults to near-white ``"#f5f5f5"``.

    Examples
    --------
    >>> cmap = team_diverging_colormap("TOR")
    >>> ax.imshow(corsi_rel, cmap=cmap, vmin=-0.1, vmax=0.1)
    """
    palette = team_palette(team_code)
    primary, secondary = palette[0], palette[1]
    return mplcol.LinearSegmentedColormap.from_list(
        f"{team_code.upper()}_div",
        [secondary, mid_color, primary],
        N=n,
    )


@property
def TEAM_COLORS() -> dict[str, dict[str, str]]:  # noqa: N802
    """All NHL team color triplets (lazy-loaded from chickenstats)."""
    return _load_team_colors()


# Make TEAM_COLORS accessible as a module-level dict (not a property)
TEAM_COLORS = _load_team_colors()
