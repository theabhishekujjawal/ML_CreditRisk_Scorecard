# -*- coding: utf-8 -*-
"""
Credit Risk Scorecard — Configuration
======================================
Author : Abhishek Ujjawal
Institution : UCD Michael Smurfit Graduate Business School

Central configuration for all constants, paths, and hyper-parameters
used across the Credit Risk Scorecard project.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# 1. Paths
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = PROJECT_ROOT / 'cs-training.csv'
MODEL_PATH: Path = PROJECT_ROOT / 'pd_model.pkl'
MODEL_FINAL_PATH: Path = PROJECT_ROOT / 'pd_model_final.pkl'

# Figure output paths
FIG_EDA: Path = PROJECT_ROOT / 'fig1_eda.png'
FIG_WOE: Path = PROJECT_ROOT / 'fig2_woe.png'
FIG_MODEL: Path = PROJECT_ROOT / 'fig3_model_performance.png'
FIG_SHAP: Path = PROJECT_ROOT / 'fig4_shap.png'
FIG_CALIBRATION: Path = PROJECT_ROOT / 'fig5_calibration.png'
FIG_ECL: Path = PROJECT_ROOT / 'fig6_ecl.png'

# ══════════════════════════════════════════════════════════════════════════════
# 2. Random State & Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE: int = 42

# ══════════════════════════════════════════════════════════════════════════════
# 3. Data Cleaning Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Age bounds (inclusive)
AGE_MIN: int = 18
AGE_MAX: int = 100

# Winsorization quantiles
WINSOR_QUANTILE: float = 0.99

# Late‑payment cap
LATE_PAYMENT_CAP: int = 10

# Age group bins for income imputation
AGE_GROUP_BINS: List[int] = [18, 35, 50, 65, 100]
AGE_GROUP_LABELS: List[str] = ['Young', 'Middle', 'Senior', 'Elder']

# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

# Delinquency weights  (30‑day = 1,  60‑day = 2,  90‑day = 3)
DELINQ_WEIGHT_30: int = 1
DELINQ_WEIGHT_60: int = 2
DELINQ_WEIGHT_90: int = 3

# Target column
TARGET_COL: str = 'SeriousDlqin2yrs'

# Features used for WoE / IV analysis
FEATURES_FOR_IV: List[str] = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    'DTI_Ratio',
    'Total_Delinquency',
    'Income_Per_Dependent',
    'Credit_Burden',
]

# Features selected for the PD model  (IV ≥ 0.02)
SELECTED_FEATURES: List[str] = [
    'Total_Delinquency',
    'RevolvingUtilizationOfUnsecuredLines',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'age',
    'Income_Per_Dependent',
    'MonthlyIncome',
    'DebtRatio',
    'DTI_Ratio',
    'NumberOfOpenCreditLinesAndLoans',
    'Credit_Burden',
    'NumberOfDependents',
]

# IV thresholds for feature‑selection labelling
IV_THRESHOLDS: Dict[str, Tuple[float, str]] = {
    'useless':     (0.02,  '❌ Useless'),
    'weak':        (0.10,  '⚠️  Weak'),
    'medium':      (0.30,  '✅ Medium'),
    'strong':      (0.50,  '💪 Strong'),
    'very_strong': (float('inf'), '🏆 Very Strong'),
}

# Minimum IV to select a feature
IV_MIN_THRESHOLD: float = 0.02

# ══════════════════════════════════════════════════════════════════════════════
# 5. Model Training
# ══════════════════════════════════════════════════════════════════════════════

TEST_SIZE: float = 0.20
SMOTE_K_NEIGHBORS: int = 5

# Logistic Regression
LR_C: float = 1.0
LR_MAX_ITER: int = 1000

# Classification threshold (favours recall for default detection)
CLASSIFICATION_THRESHOLD: float = 0.30

# Cross‑validation
CV_FOLDS: int = 5

# ══════════════════════════════════════════════════════════════════════════════
# 6. Credit Scorecard Conversion
# ══════════════════════════════════════════════════════════════════════════════

SCORE_MIN: int = 300
SCORE_MAX: int = 850
SCORE_RANGE: int = SCORE_MAX - SCORE_MIN

# Score thresholds for risk bands
SCORE_LOW_RISK: int = 720
SCORE_MEDIUM_RISK: int = 600

# Points to Double Odds (PDO)
PDO: int = 50
BASE_SCORE: int = 600
BASE_ODDS: float = 14.0   # ratio of good to bad in training set

# ══════════════════════════════════════════════════════════════════════════════
# 7. IFRS‑9 ECL Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Loss Given Default (assumed — simplified)
LGD: float = 0.45

# Exposure at Default multiplier
EAD_MULTIPLIER: float = 1.0

# Average loan amount for portfolio simulation
AVG_LOAN_AMOUNT: float = 10_000.0

# Discount rate for lifetime ECL
DISCOUNT_RATE: float = 0.05

# Maturity (years) for lifetime ECL
MATURITY_YEARS: int = 3

# IFRS‑9 stage thresholds (PD based)
STAGE_1_MAX_PD: float = 0.10   # PD ≤ 10 % → Stage 1
STAGE_2_MAX_PD: float = 0.30   # 10 % < PD ≤ 30 % → Stage 2
#                                  PD > 30 % → Stage 3

# Prior probability correction
REAL_PRIOR: float = 0.0668
TRAIN_PRIOR: float = 0.50

# ══════════════════════════════════════════════════════════════════════════════
# 8. Visualisation Palette (Light Theme)
# ══════════════════════════════════════════════════════════════════════════════

VIZ_BG_DARK: str = '#f8f9fa'      # light gray background
VIZ_BG_PANEL: str = '#ffffff'     # white panels
VIZ_TEXT: str = '#1f2937'         # dark gray text
VIZ_GRID: str = '#e5e7eb'         # subtle light gray grid

VIZ_COLOR_GOOD: str = '#16a34a'      # deep green (high contrast on white)
VIZ_COLOR_BAD: str = '#dc2626'       # deep red
VIZ_COLOR_NEUTRAL: str = '#4f46e5'   # deep indigo
VIZ_COLOR_ACCENT: str = '#d97706'    # deep amber
VIZ_COLOR_INFO: str = '#2563eb'      # deep blue

VIZ_DPI: int = 300
VIZ_FONT_TITLE: int = 14
VIZ_FONT_SUBTITLE: int = 12
VIZ_FONT_LABEL: int = 11
VIZ_FONT_TICK: int = 9
VIZ_FONT_ANNOT: int = 8

# Colour‑blind friendly sequential palette
VIZ_PALETTE_SEQ: List[str] = [
    '#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51',
]

# Colour‑blind friendly qualitative palette
VIZ_PALETTE_QUAL: List[str] = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
    '#59a14f', '#edc948', '#b07aa1', '#ff9da7',
]
