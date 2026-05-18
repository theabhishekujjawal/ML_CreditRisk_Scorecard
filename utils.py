# -*- coding: utf-8 -*-
"""
Credit Risk Scorecard — Utility Functions
==========================================
Author : Abhishek Ujjawal
Institution : UCD Michael Smurfit Graduate Business School

Reusable helper functions for:
 • WoE / IV computation
 • Scorecard conversion
 • Prior‑probability correction
 • Plotting helpers (theme boilerplate)
 • Data‑quality profiling
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import (
    BASE_ODDS,
    BASE_SCORE,
    IV_MIN_THRESHOLD,
    LGD,
    PDO,
    RANDOM_STATE,
    REAL_PRIOR,
    SCORE_MAX,
    SCORE_MIN,
    TRAIN_PRIOR,
    VIZ_BG_DARK,
    VIZ_BG_PANEL,
    VIZ_COLOR_ACCENT,
    VIZ_COLOR_BAD,
    VIZ_COLOR_GOOD,
    VIZ_COLOR_INFO,
    VIZ_COLOR_NEUTRAL,
    VIZ_DPI,
    VIZ_FONT_ANNOT,
    VIZ_FONT_LABEL,
    VIZ_FONT_SUBTITLE,
    VIZ_FONT_TICK,
    VIZ_FONT_TITLE,
    VIZ_GRID,
    VIZ_PALETTE_QUAL,
    VIZ_PALETTE_SEQ,
    VIZ_TEXT,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Weight of Evidence & Information Value
# ══════════════════════════════════════════════════════════════════════════════


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """Calculate Weight of Evidence (WoE) and Information Value (IV).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    feature : str
        Column name of the predictor variable.
    target : str
        Column name of the binary target (0/1).
    n_bins : int, optional
        Number of quantile bins for continuous features, by default 10.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        A tuple of (WoE detail table, total IV).
    """
    data = df[[feature, target]].copy()

    # Bin continuous features into quantiles
    try:
        data['bin'] = pd.qcut(data[feature], q=n_bins, duplicates='drop')
    except ValueError:
        data['bin'] = pd.cut(data[feature], bins=n_bins, duplicates='drop')

    grouped = data.groupby('bin', observed=False)[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']

    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    # Avoid division by zero with Laplace smoothing
    grouped['event_dist'] = (grouped['events'] + 0.5) / (total_events + 1)
    grouped['non_event_dist'] = (grouped['non_events'] + 0.5) / (total_non_events + 1)

    grouped['woe'] = np.log(grouped['non_event_dist'] / grouped['event_dist'])
    grouped['iv'] = (grouped['non_event_dist'] - grouped['event_dist']) * grouped['woe']

    total_iv = float(grouped['iv'].sum())
    grouped = grouped.reset_index()

    return grouped, total_iv


def classify_iv(iv_value: float) -> str:
    """Classify Information Value into descriptive strength categories.

    Parameters
    ----------
    iv_value : float
        The computed IV for a feature.

    Returns
    -------
    str
        Descriptive label (e.g., '💪 Strong').
    """
    if iv_value < 0.02:
        return '❌ Useless'
    elif iv_value < 0.10:
        return '⚠️  Weak'
    elif iv_value < 0.30:
        return '✅ Medium'
    elif iv_value < 0.50:
        return '💪 Strong'
    else:
        return '🏆 Very Strong'


# ══════════════════════════════════════════════════════════════════════════════
# 2. Prior‑Probability Correction (Post‑SMOTE)
# ══════════════════════════════════════════════════════════════════════════════


def prior_correction(
    p_model: float | np.ndarray,
    real_prior: float = REAL_PRIOR,
    train_prior: float = TRAIN_PRIOR,
) -> float | np.ndarray:
    """Correct model‑predicted probabilities to reflect real‑world priors.

    After SMOTE balancing, the model sees 50 / 50 default rates. This
    function re‑maps the predicted PD to the original population base rate.

    Parameters
    ----------
    p_model : float or np.ndarray
        Raw model‑predicted probability of default.
    real_prior : float
        Real‑world default rate (population prior).
    train_prior : float
        Training‑set default rate (after SMOTE: 0.50).

    Returns
    -------
    float or np.ndarray
        Corrected probability of default.
    """
    beta_0 = np.log(real_prior / (1 - real_prior)) - np.log(
        train_prior / (1 - train_prior)
    )
    logit = np.log(p_model / (1 - p_model + 1e-15)) + beta_0
    return 1 / (1 + np.exp(-logit))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Credit Scorecard Mapping
# ══════════════════════════════════════════════════════════════════════════════


def pd_to_credit_score(
    pd_value: float | np.ndarray,
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
) -> int | np.ndarray:
    """Map a probability of default to a credit score (300–850 scale).

    Uses a linear inverse mapping:  higher PD ⟹ lower score.

    Parameters
    ----------
    pd_value : float or np.ndarray
        Probability of default ∈ (0, 1).
    score_min : int
        Lowest possible credit score.
    score_max : int
        Highest possible credit score.

    Returns
    -------
    int or np.ndarray
        Credit score (clipped to [score_min, score_max]).
    """
    score = score_max - (pd_value * (score_max - score_min))
    return np.clip(np.round(score).astype(int), score_min, score_max)


def pd_to_credit_score_pdo(
    pd_value: float | np.ndarray,
    base_score: int = BASE_SCORE,
    base_odds: float = BASE_ODDS,
    pdo: int = PDO,
) -> float | np.ndarray:
    """Map PD to credit score using industry‑standard PDO methodology.

    score = base_score + (pdo / ln(2)) × ln(odds)

    Parameters
    ----------
    pd_value : float or np.ndarray
        Probability of default ∈ (0, 1).
    base_score : int
        Score at which odds = base_odds.
    base_odds : float
        Good‑to‑bad odds at the base score.
    pdo : int
        Points to Double the Odds.

    Returns
    -------
    float or np.ndarray
        Credit score.
    """
    odds = (1 - pd_value) / (pd_value + 1e-15)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds)


# ══════════════════════════════════════════════════════════════════════════════
# 4. IFRS‑9 Helpers
# ══════════════════════════════════════════════════════════════════════════════


def assign_ifrs9_stage(
    pd_value: float | np.ndarray,
    stage1_max: float = 0.10,
    stage2_max: float = 0.30,
) -> str | np.ndarray:
    """Assign IFRS‑9 impairment stage based on PD thresholds.

    Parameters
    ----------
    pd_value : float or np.ndarray
        Probability of default.
    stage1_max : float
        Maximum PD for Stage 1.
    stage2_max : float
        Maximum PD for Stage 2.

    Returns
    -------
    str or np.ndarray
        'Stage 1', 'Stage 2', or 'Stage 3'.
    """
    if isinstance(pd_value, (float, int)):
        if pd_value <= stage1_max:
            return 'Stage 1'
        elif pd_value <= stage2_max:
            return 'Stage 2'
        else:
            return 'Stage 3'
    # Vectorised
    conditions = [pd_value <= stage1_max, pd_value <= stage2_max]
    choices = ['Stage 1', 'Stage 2']
    return np.select(conditions, choices, default='Stage 3')


def compute_ecl(
    pd_value: float | np.ndarray,
    lgd: float = LGD,
    ead: float = 1.0,
) -> float | np.ndarray:
    """Compute Expected Credit Loss (simplified).

    ECL = PD × LGD × EAD

    Parameters
    ----------
    pd_value : float or np.ndarray
        Probability of default.
    lgd : float
        Loss Given Default.
    ead : float
        Exposure at Default.

    Returns
    -------
    float or np.ndarray
        Expected Credit Loss.
    """
    return pd_value * lgd * ead


def compute_lifetime_ecl(
    pd_annual: float,
    lgd: float = LGD,
    ead: float = 1.0,
    maturity: int = 3,
    discount_rate: float = 0.05,
) -> float:
    """Compute multi‑year lifetime ECL with discounting.

    Parameters
    ----------
    pd_annual : float
        Annual probability of default.
    lgd : float
        Loss Given Default.
    ead : float
        Exposure at Default.
    maturity : int
        Remaining maturity in years.
    discount_rate : float
        Annual discount rate.

    Returns
    -------
    float
        Discounted lifetime ECL.
    """
    ecl_total = 0.0
    survival = 1.0
    for t in range(1, maturity + 1):
        marginal_pd = survival * pd_annual
        discount = 1 / (1 + discount_rate) ** t
        ecl_total += marginal_pd * lgd * ead * discount
        survival *= (1 - pd_annual)
    return ecl_total


# ══════════════════════════════════════════════════════════════════════════════
# 5. Model Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════


def compute_ks_statistic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """Compute the Kolmogorov–Smirnov (KS) statistic.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1).
    y_prob : np.ndarray
        Predicted probabilities of default.

    Returns
    -------
    Tuple[float, float]
        (KS statistic, threshold at which max separation occurs).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    idx = np.argmax(ks_values)
    return float(ks_values[idx]), float(thresholds[idx])


def compute_gini(auc_score: float) -> float:
    """Compute the Gini coefficient from AUC.

    Gini = 2 × AUC − 1

    Parameters
    ----------
    auc_score : float
        Area Under ROC Curve.

    Returns
    -------
    float
        Gini coefficient ∈ [−1, 1].
    """
    return 2 * auc_score - 1


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI).

    Parameters
    ----------
    expected : np.ndarray
        Score distribution from development sample.
    actual : np.ndarray
        Score distribution from monitoring sample.
    n_bins : int
        Number of bins for the distribution comparison.

    Returns
    -------
    float
        PSI value.  PSI < 0.10: stable,  0.10–0.25: moderate shift,  >0.25: significant.
    """
    breakpoints = np.linspace(0, 1, n_bins + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for multicollinearity check.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no target column).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'VIF'].
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Data Quality Profiling
# ══════════════════════════════════════════════════════════════════════════════


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a comprehensive data quality profile.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Profile with completeness, uniqueness, dtype, min, max, mean.
    """
    report = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'count': df.count(),
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'unique_pct': (df.nunique() / len(df) * 100).round(2),
        'mean': df.select_dtypes(include='number').mean(),
        'std': df.select_dtypes(include='number').std(),
        'min': df.select_dtypes(include='number').min(),
        'max': df.select_dtypes(include='number').max(),
    })
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plotting Helpers (Theme)
# ══════════════════════════════════════════════════════════════════════════════


def apply_dark_theme() -> None:
    """Apply a consistent theme to all figures."""
    mpl.rcParams.update({
        'figure.facecolor': VIZ_BG_DARK,
        'axes.facecolor': VIZ_BG_PANEL,
        'axes.edgecolor': VIZ_GRID,
        'axes.labelcolor': VIZ_TEXT,
        'axes.titlesize': VIZ_FONT_TITLE,
        'axes.labelsize': VIZ_FONT_LABEL,
        'xtick.color': VIZ_TEXT,
        'ytick.color': VIZ_TEXT,
        'xtick.labelsize': VIZ_FONT_TICK,
        'ytick.labelsize': VIZ_FONT_TICK,
        'text.color': VIZ_TEXT,
        'grid.color': VIZ_GRID,
        'grid.alpha': 0.2,
        'legend.facecolor': VIZ_BG_PANEL,
        'legend.edgecolor': VIZ_GRID,
        'legend.fontsize': VIZ_FONT_TICK,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial'],
        'savefig.dpi': VIZ_DPI,
        'savefig.bbox': 'tight',
        'savefig.facecolor': VIZ_BG_DARK,
    })


def style_axis(
    ax: plt.Axes,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    subtitle: str = '',
    grid: bool = True,
) -> None:
    """Apply consistent styling to a matplotlib Axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes to style.
    title : str
        Main title text.
    xlabel, ylabel : str
        Axis labels.
    subtitle : str
        Subtitle displayed below the main title.
    grid : bool
        Whether to show grid lines.
    """
    if title:
        ax.set_title(
            title,
            fontsize=VIZ_FONT_TITLE,
            fontweight='bold',
            color=VIZ_TEXT,
            pad=10,
        )
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes,
            fontsize=VIZ_FONT_SUBTITLE - 2,
            color='#999999',
            ha='center',
            va='bottom',
        )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=VIZ_FONT_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=VIZ_FONT_LABEL)
    if grid:
        ax.grid(True, alpha=0.2, color=VIZ_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(VIZ_GRID)
    ax.spines['bottom'].set_color(VIZ_GRID)


def save_figure(
    fig: plt.Figure,
    filepath: Any,
    close: bool = True,
) -> None:
    """Save a figure with consistent settings.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to save.
    filepath : Path or str
        Destination path.
    close : bool
        Whether to close the figure after saving.
    """
    fig.savefig(str(filepath), dpi=VIZ_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    logger.info(f'Saved figure → {filepath}')
    if close:
        plt.close(fig)


def annotate_bars(
    ax: plt.Axes,
    fmt: str = '{:.1f}',
    fontsize: int = VIZ_FONT_ANNOT,
    color: str = VIZ_TEXT,
    offset: float = 0.5,
) -> None:
    """Add value labels above each bar in a bar chart.

    Parameters
    ----------
    ax : plt.Axes
        The axes containing bar containers.
    fmt : str
        Format string for the label text.
    fontsize : int
        Label font size.
    color : str
        Label colour.
    offset : float
        Vertical offset from bar top.
    """
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    fmt.format(height),
                    ha='center',
                    va='bottom',
                    fontsize=fontsize,
                    color=color,
                )


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[int, int] = (14, 8),
    squeeze: bool = True,
) -> Tuple[plt.Figure, Any]:
    """Create a figure with themed background.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid dimensions.
    figsize : Tuple[int, int]
        Figure size in inches.
    squeeze : bool
        If True, squeeze single‑element arrays.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray | plt.Axes]
        The figure and axes.
    """
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        facecolor=VIZ_BG_DARK,
        squeeze=squeeze,
    )
    return fig, axes


def format_thousands(ax: plt.Axes, axis: str = 'y') -> None:
    """Apply comma‑separated thousands formatting to an axis.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    axis : str
        'x', 'y', or 'both'.
    """
    formatter = mpl.ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)


def format_percent(ax: plt.Axes, axis: str = 'y') -> None:
    """Apply percentage formatting to an axis.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    axis : str
        'x', 'y', or 'both'.
    """
    formatter = mpl.ticker.FuncFormatter(lambda x, _: f'{x:.0%}')
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)
