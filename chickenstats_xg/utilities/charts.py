"""
Standalone classification chart functions — yellowbrick style, no yellowbrick dependency.

All functions:
  - Accept raw arrays/lists (y_true, y_pred, y_proba, importances, …)
  - Return a matplotlib Figure (or None on failure)
  - Apply the yellowbrick aesthetic automatically
  - Expose palette, colormap, title, label, figsize, and dpi overrides

Charts
------
classification_report   Precision / recall / F1 heatmap per class
roc_auc                 ROC curves with macro/micro averages
precision_recall_curve  PR curves with ISO-F1 contours
class_prediction_error  Stacked bar of predicted vs actual counts
confusion_matrix        Heatmap confusion matrix (counts or %)
feature_importances     Horizontal bar (absolute or relative)
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.ticker as mticker
import numpy as np
import sklearn.metrics

from .style import (
    YB_KEY,
    LINE_COLOR,
    color_palette,
    color_sequence,
    find_text_color,
    set_style,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE = "yellowbrick"
_DEFAULT_HEATMAP = "RdPu"
_HEATMAP_OVER = "#2a7d4f"   # dark green for out-of-range highs (matches yellowbrick)
_DPI = 100
_FIGSIZE = (6, 4)


def _apply_style(theme: str = "light") -> None:
    """Apply the merged aesthetic before drawing. Called at the top of each chart."""
    set_style(theme=theme)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Classification Report
# ---------------------------------------------------------------------------


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: Sequence[str] | None = None,
    support: bool = True,
    title: str = "Classification Report",
    colormap: str = _DEFAULT_HEATMAP,
    figsize: tuple[float, float] = _FIGSIZE,
    dpi: int = _DPI,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Heatmap of precision, recall, and F1-score per class.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels (hard predictions, not probabilities).
    classes:
        Display names for each class label (sorted order). Defaults to
        the sorted unique values in y_true.
    support:
        If True, append an (n=…) count to each class label row.
    title:
        Chart title.
    colormap:
        Named sequence palette from style.SEQUENCES or a matplotlib cmap name.
    figsize / dpi:
        Figure dimensions and resolution.
    ax:
        Existing axes to draw into. A new Figure is created when None.
    """
    try:
        _apply_style()
        labels_int = sorted(np.unique(y_true).tolist())
        if classes is None:
            classes = [str(l) for l in labels_int]

        report = sklearn.metrics.classification_report(
            y_true, y_pred, labels=labels_int, output_dict=True, zero_division=0
        )

        cols = ["precision", "recall", "f1-score"]
        data = np.array([[report[str(l)][c] for c in cols] for l in labels_int])
        supports = [int(report[str(l)]["support"]) for l in labels_int]

        row_labels = (
            [f"{cls} (n={sup:,})" for cls, sup in zip(classes, supports)]
            if support else list(classes)
        )

        cmap = color_sequence(colormap)
        cmap.set_under("white")
        cmap.set_over(_HEATMAP_OVER)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        im = ax.pcolormesh(data, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(cols)) + 0.5)
        ax.set_xticklabels(cols, fontsize=10)
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_xlim(0, len(cols))
        ax.set_ylim(0, len(row_labels))

        colors_hex = cmap(np.linspace(0, 1, 256))
        for i in range(len(row_labels)):
            for j, val in enumerate(data[i]):
                bg = mplcol.to_hex(cmap(val))
                txt_color = find_text_color(bg)
                ax.text(j + 0.5, i + 0.5, f"{val:.3f}", ha="center", va="center",
                        color=txt_color, fontsize=11, fontweight="bold")

        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close()
        return None


# ---------------------------------------------------------------------------
# ROC AUC
# ---------------------------------------------------------------------------


def roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes: Sequence[str] | None = None,
    title: str = "ROC AUC",
    palette: str = _DEFAULT_PALETTE,
    figsize: tuple[float, float] = _FIGSIZE,
    dpi: int = _DPI,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    ROC curves with per-class, macro-average, and micro-average lines.

    Parameters
    ----------
    y_true:
        Ground-truth labels (integer class indices).
    y_proba:
        Predicted probabilities. Shape (n_samples,) for binary or
        (n_samples, n_classes) for multiclass.
    classes:
        Display names for classes (sorted order).
    title / palette / figsize / dpi:
        Aesthetic controls.
    ax:
        Existing axes to draw into.
    """
    try:
        _apply_style()
        labels_int = sorted(np.unique(y_true).tolist())
        n_classes = len(labels_int)
        if classes is None:
            classes = [str(l) for l in labels_int]

        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])

        colors = color_palette(palette, n_colors=n_classes + 2)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        fpr_dict: dict[int, np.ndarray] = {}
        tpr_dict: dict[int, np.ndarray] = {}

        for i, (label, cls_name) in enumerate(zip(labels_int, classes)):
            y_bin = (y_true == label).astype(int)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_bin, y_proba[:, i])
            auc_val = sklearn.metrics.auc(fpr, tpr)
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
            ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                    label=f"{cls_name} (AUC={auc_val:.3f})")

        if n_classes > 1:
            # Macro average
            all_fpr = np.unique(np.concatenate(list(fpr_dict.values())))
            mean_tpr = np.mean(
                [np.interp(all_fpr, fpr_dict[i], tpr_dict[i]) for i in range(n_classes)], axis=0
            )
            macro_auc = sklearn.metrics.auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, color=colors[n_classes], lw=1.5, linestyle="--",
                    label=f"macro avg (AUC={macro_auc:.3f})")

            # Micro average
            y_true_bin = np.zeros((len(y_true), n_classes), dtype=int)
            for i, label in enumerate(labels_int):
                y_true_bin[:, i] = (y_true == label).astype(int)
            fpr_micro, tpr_micro, _ = sklearn.metrics.roc_curve(
                y_true_bin.ravel(), y_proba.ravel()
            )
            micro_auc = sklearn.metrics.auc(fpr_micro, tpr_micro)
            ax.plot(fpr_micro, tpr_micro, color=colors[n_classes + 1], lw=1.0, linestyle=":",
                    label=f"micro avg (AUC={micro_auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="lightgray", linewidth=0.8)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close()
        return None


# ---------------------------------------------------------------------------
# Precision-Recall Curve
# ---------------------------------------------------------------------------


def precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes: Sequence[str] | None = None,
    iso_f1: bool = True,
    fill_area: bool = True,
    fill_opacity: float = 0.2,
    line_opacity: float = 0.8,
    title: str = "Precision-Recall Curve",
    palette: str = _DEFAULT_PALETTE,
    figsize: tuple[float, float] = _FIGSIZE,
    dpi: int = _DPI,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Precision-recall curves with optional ISO-F1 contours.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_proba:
        Predicted probabilities. Shape (n_samples,) for binary or
        (n_samples, n_classes) for multiclass.
    classes:
        Display names for classes.
    iso_f1:
        Overlay dashed ISO-F1 score contours.
    fill_area:
        Fill the area under each PR curve.
    fill_opacity / line_opacity:
        Alpha values for fill and line respectively.
    title / palette / figsize / dpi:
        Aesthetic controls.
    ax:
        Existing axes to draw into.
    """
    try:
        _apply_style()
        labels_int = sorted(np.unique(y_true).tolist())
        n_classes = len(labels_int)
        if classes is None:
            classes = [str(l) for l in labels_int]

        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])

        colors = color_palette(palette, n_colors=n_classes)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        if iso_f1:
            recall_pts = np.linspace(0.01, 1.0, 200)
            for f1_score in [0.2, 0.4, 0.6, 0.8]:
                prec = f1_score * recall_pts / (2 * recall_pts - f1_score)
                mask = prec >= 0
                ax.plot(recall_pts[mask], prec[mask], color="#333333",
                        alpha=0.2, linestyle="--", linewidth=0.8)
                ax.annotate(f"F1={f1_score:.1f}",
                            xy=(recall_pts[mask][-1], prec[mask][-1]),
                            fontsize=7, color="#555555")

        for i, (label, cls_name) in enumerate(zip(labels_int, classes)):
            y_bin = (y_true == label).astype(int)
            prec, rec, _ = sklearn.metrics.precision_recall_curve(y_bin, y_proba[:, i])
            ap = sklearn.metrics.average_precision_score(y_bin, y_proba[:, i])
            color = colors[i]
            ax.plot(rec, prec, color=color, lw=1.5, alpha=line_opacity,
                    label=f"{cls_name} (AP={ap:.3f})")
            if fill_area:
                ax.fill_between(rec, prec, alpha=fill_opacity, color=color)

        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close()
        return None


# ---------------------------------------------------------------------------
# Class Prediction Error
# ---------------------------------------------------------------------------


def class_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: Sequence[str] | None = None,
    title: str = "Class Prediction Error",
    palette: str = _DEFAULT_PALETTE,
    figsize: tuple[float, float] = _FIGSIZE,
    dpi: int = _DPI,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Stacked bar chart: for each true class, how predictions were distributed.

    Rows = actual class, bar stacks = predicted class.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    classes:
        Display names for classes (sorted order).
    title / palette / figsize / dpi:
        Aesthetic controls.
    ax:
        Existing axes to draw into.
    """
    try:
        _apply_style()
        labels_int = sorted(np.unique(y_true).tolist())
        n_classes = len(labels_int)
        if classes is None:
            classes = [str(l) for l in labels_int]

        cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels_int)
        colors = color_palette(palette, n_colors=n_classes)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        x = np.arange(n_classes)
        bottom = np.zeros(n_classes)

        for j, pred_cls in enumerate(classes):
            vals = cm[:, j]
            bars = ax.bar(x, vals, bottom=bottom, label=f"Predicted: {pred_cls}",
                          color=colors[j], width=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:,}", ha="center", va="center",
                        fontsize=9, color="white" if j == n_classes - 1 else YB_KEY,
                    )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_xlabel("Actual Class", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close()
        return None


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: Sequence[str] | None = None,
    normalize: bool = False,
    colormap: str = _DEFAULT_HEATMAP,
    title: str = "Confusion Matrix",
    figsize: tuple[float, float] = _FIGSIZE,
    dpi: int = _DPI,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    classes:
        Display names for classes (sorted order).
    normalize:
        If True, show row-normalised proportions instead of raw counts.
    colormap:
        Named sequential palette or matplotlib cmap.
    title / figsize / dpi:
        Aesthetic controls.
    ax:
        Existing axes to draw into.
    """
    try:
        _apply_style()
        labels_int = sorted(np.unique(y_true).tolist())
        n_classes = len(labels_int)
        if classes is None:
            classes = [str(l) for l in labels_int]

        cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels_int)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.where(row_sums > 0, cm / row_sums, 0.0)
            fmt = ".2f"
        else:
            cm_display = cm.astype(float)
            fmt = ",d"

        cmap = color_sequence(colormap)
        cmap.set_under("white")
        cmap.set_over(_HEATMAP_OVER)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        vmax = cm_display.max() if cm_display.max() > 0 else 1
        im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax)

        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_display[i, j]
                bg = mplcol.to_hex(cmap(val / vmax if vmax > 0 else 0))
                txt = find_text_color(bg)
                label = (f"{val:.2f}" if normalize else f"{int(cm[i, j]):,}")
                edge_color = YB_KEY if i == j else "white"
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=edge_color, lw=1.5,
                ))
                ax.text(j, i, label, ha="center", va="center",
                        color=txt, fontsize=10, fontweight="bold")

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close()
        return None


# ---------------------------------------------------------------------------
# Feature Importances
# ---------------------------------------------------------------------------


def feature_importances(
    importances: np.ndarray | Sequence[float],
    feature_names: Sequence[str],
    *,
    relative: bool = False,
    topn: int = 10,
    title: str | None = None,
    palette: str = _DEFAULT_PALETTE,
    figsize: tuple[float, float] = _FIGSIZE,
    dpi: int = _DPI,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Horizontal bar chart of feature importances (absolute or relative).

    Parameters
    ----------
    importances:
        Raw feature importance values (e.g. model.feature_importances_).
    feature_names:
        Names corresponding to each importance value.
    relative:
        If True, normalise to sum=1 and display as relative importances.
    topn:
        Show only the top-n most important features.
    title:
        Chart title. Defaults to "Top N (Relative) Feature Importances".
    palette:
        Named discrete palette for bar colors.
    figsize / dpi:
        Aesthetic controls.
    ax:
        Existing axes to draw into.
    """
    try:
        _apply_style()
        importances = np.asarray(importances, dtype=float)

        if relative:
            total = importances.sum()
            importances = importances / total if total > 0 else importances

        n_show = min(topn, len(importances))
        indices = np.argsort(importances)[-n_show:]
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]

        colors = color_palette(palette, n_colors=n_show)
        # Gradient: least important → lightest color (first in palette cycle is fine)
        bar_colors = [colors[i % len(colors)] for i in range(n_show)]

        if title is None:
            title = f"Top {n_show} {'Relative ' if relative else ''}Feature Importances"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        y_pos = np.arange(n_show)
        ax.barh(y_pos, top_vals, color=bar_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Relative Importance" if relative else "Feature Importance", fontsize=10)
        if relative:
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close()
        return None


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------


def all_classifier_charts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    importances: np.ndarray,
    feature_names: Sequence[str],
    *,
    classes: Sequence[str] | None = None,
    title_prefix: str = "",
    palette: str = _DEFAULT_PALETTE,
    colormap: str = _DEFAULT_HEATMAP,
    figsize: tuple[float, float] = _FIGSIZE,
    dpi: int = _DPI,
    topn: int = 10,
) -> dict[str, plt.Figure | None]:
    """
    Generate all seven standard classifier charts in one call.

    Returns a dict keyed by chart name so callers can log selectively.

    Keys
    ----
    classification_report, roc_auc, precision_recall_curve,
    class_prediction_error, confusion_matrix,
    feature_importances, relative_feature_importances
    """
    pfx = f"{title_prefix} — " if title_prefix else ""
    shared = dict(classes=classes, palette=palette, figsize=figsize, dpi=dpi)
    hm_shared = dict(classes=classes, colormap=colormap, figsize=figsize, dpi=dpi)

    return {
        "classification_report": classification_report(
            y_true, y_pred,
            title=f"{pfx}Classification Report",
            colormap=colormap, **{k: v for k, v in shared.items() if k != "palette"},
        ),
        "roc_auc": roc_auc(
            y_true, y_proba,
            title=f"{pfx}ROC AUC",
            **shared,
        ),
        "precision_recall_curve": precision_recall_curve(
            y_true, y_proba,
            title=f"{pfx}Precision-Recall Curve",
            **shared,
        ),
        "class_prediction_error": class_prediction_error(
            y_true, y_pred,
            title=f"{pfx}Class Prediction Error",
            **shared,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred,
            title=f"{pfx}Confusion Matrix",
            **hm_shared,
        ),
        "feature_importances": feature_importances(
            importances, feature_names,
            title=f"{pfx}Feature Importances",
            relative=False, topn=topn, palette=palette, figsize=figsize, dpi=dpi,
        ),
        "relative_feature_importances": feature_importances(
            importances, feature_names,
            title=f"{pfx}Relative Feature Importances",
            relative=True, topn=topn, palette=palette, figsize=figsize, dpi=dpi,
        ),
    }