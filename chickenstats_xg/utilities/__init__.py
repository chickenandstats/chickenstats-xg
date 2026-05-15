"""
chickenstats-xg visualization utilities.

Blends the yellowbrick color system with the chickenstats matplotlib style
(dimgray text, no top/right spines, bold titles/labels) and adds NHL team
color utilities. All charts are pure functions that return Figure objects —
no model objects required, no yellowbrick dependency.

Quick start
-----------
    from chickenstats_xg.utilities import all_classifier_charts, set_style, team_colormap
    from chickenstats_xg.utilities.style import team_palette, team_diverging_colormap

Public API
----------
Charts (utilities.charts)
    classification_report           Precision/recall/F1 heatmap per class
    roc_auc                         ROC curves with macro/micro averages
    precision_recall_curve          PR curves with optional ISO-F1 contours
    class_prediction_error          Stacked bar of predicted vs actual counts
    confusion_matrix                Annotated heatmap (counts or normalised)
    feature_importances             Horizontal bar, absolute or relative
    all_classifier_charts           All seven charts → dict[name, Figure]

Style (utilities.style)
    set_style(theme, palette, ...)  Apply global aesthetic ("light" or "dark")
    color_palette                   Named discrete palette → list[hex]
    color_sequence                  Named sequential palette → ListedColormap
    find_text_color                 WCAG-contrast text color for a background
    PALETTES                        All named discrete palette dicts
    SEQUENCES                       All named sequential palette dicts
    YB_KEY / LINE_COLOR             Structural dark color (#111111)
    DIMGRAY                         Chrome color for text/axes ("dimgray")

NHL team colors (utilities.style)
    TEAM_COLORS                     All team color triplets (primary/secondary/neutral)
    team_palette(code)              [primary, secondary, neutral] hex list
    team_colormap(code)             Sequential cmap: white → team primary
    team_diverging_colormap(code)   Diverging cmap: secondary → mid → primary
"""

from .charts import (
    all_classifier_charts,
    class_prediction_error,
    classification_report,
    confusion_matrix,
    feature_importances,
    precision_recall_curve,
    roc_auc,
)
from .style import (
    DIMGRAY,
    LINE_COLOR,
    PALETTES,
    SEQUENCES,
    TEAM_COLORS,
    YB_KEY,
    color_palette,
    color_sequence,
    find_text_color,
    set_style,
    team_colormap,
    team_diverging_colormap,
    team_palette,
)

__all__ = [
    # charts
    "all_classifier_charts",
    "class_prediction_error",
    "classification_report",
    "confusion_matrix",
    "feature_importances",
    "precision_recall_curve",
    "roc_auc",
    # style
    "DIMGRAY",
    "LINE_COLOR",
    "PALETTES",
    "SEQUENCES",
    "TEAM_COLORS",
    "YB_KEY",
    "color_palette",
    "color_sequence",
    "find_text_color",
    "set_style",
    "team_colormap",
    "team_diverging_colormap",
    "team_palette",
]