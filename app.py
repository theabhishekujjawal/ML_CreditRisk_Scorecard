# -*- coding: utf-8 -*-
"""
Credit Risk Scorecard — Streamlit Application
===============================================
Author      : Abhishek Ujjawal
Institution : UCD Michael Smurfit Graduate Business School
Framework   : Basel III + IFRS 9

Interactive demo for the credit-risk PD model.  Allows a user to
enter borrower details and receive a real-time credit score,
default probability, IFRS-9 ECL estimate, and loan decision with
an explanation of the top risk factors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-8s │ %(message)s',
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

REAL_PRIOR: float = 0.0668
TRAIN_PRIOR: float = 0.50
LGD_DEFAULT: float = 0.45
SCORE_MIN: int = 300
SCORE_MAX: int = 850
SCORE_RANGE: int = SCORE_MAX - SCORE_MIN
SCORE_LOW_RISK: int = 720
SCORE_MEDIUM_RISK: int = 600
MODEL_AUC: str = '≈ 0.86'
MODEL_KS: str = '≈ 0.54'
MODEL_PATH: Path = Path(__file__).resolve().parent / 'pd_model_final.pkl'

FEATURE_NAMES = [
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

DISPLAY_NAMES = [
    'Total Delinquency',
    'Credit Utilisation',
    'Late 30-59 Days',
    'Age',
    'Income / Dependent',
    'Monthly Income',
    'Debt Ratio',
    'DTI Ratio',
    'Open Loans',
    'Credit Burden',
    'Dependents',
]

LGD_TABLE: Dict[str, float] = {
    'Home Loan':      0.25,
    'Car Loan':       0.35,
    'Gold Loan':      0.15,
    'Personal Loan':  0.75,
    'Credit Card':    0.85,
    'Education Loan': 0.60,
    'Business Loan':  0.45,
}


# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════


def prior_correction(
    p_model: float | np.ndarray,
    real_prior: float = REAL_PRIOR,
    train_prior: float = TRAIN_PRIOR,
) -> float | np.ndarray:
    """Correct model-predicted PD to real-world prior.

    After SMOTE (50/50), probabilities over-estimate default.
    This applies log-odds correction to map back to the true
    population base rate.
    """
    beta_0 = np.log(real_prior / (1 - real_prior)) - np.log(
        train_prior / (1 - train_prior)
    )
    logit = np.log(p_model / (1 - p_model + 1e-15)) + beta_0
    return float(1 / (1 + np.exp(-logit)))


def pd_to_score(pd_val: float) -> int:
    """Map probability of default to a 300–850 credit score."""
    return int(np.clip(SCORE_MAX - pd_val * SCORE_RANGE, SCORE_MIN, SCORE_MAX))


def classify_risk(score: int) -> Tuple[str, str, str]:
    """Return (decision, risk_category, colour) based on score."""
    if score >= SCORE_LOW_RISK:
        return '✅ Approved', 'Low Risk', '#22c55e'
    elif score >= SCORE_MEDIUM_RISK:
        return '⚠️ Conditional Approval', 'Medium Risk', '#f59e0b'
    else:
        return '❌ Declined', 'High Risk', '#ef4444'


def assign_ifrs9_stage(pd_val: float) -> str:
    """Return IFRS-9 impairment stage."""
    if pd_val <= 0.10:
        return 'Stage 1 (12-month ECL)'
    elif pd_val <= 0.30:
        return 'Stage 2 (Lifetime ECL)'
    else:
        return 'Stage 3 (Credit-impaired)'


# ══════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title='Credit Risk Scorecard — IFRS 9',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #333355;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetric"] label {
        color: #a0a0c0;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Sidebar — Model info & settings
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.image('https://img.icons8.com/color/96/bank-building.png', width=64)
    st.markdown('## 🏦 Credit Risk Scorecard')
    st.caption('Basel III + IFRS 9 Compliant')
    st.divider()

    st.markdown('### ℹ️ Model Information')
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Algorithm | Logistic Regression |
    | AUC-ROC | {MODEL_AUC} |
    | KS Statistic | {MODEL_KS} |
    | Calibration | Prior correction |
    | Framework | IFRS 9 / Basel III |
    """)

    st.divider()
    st.markdown('### ⚙️ Settings')
    loan_type = st.selectbox(
        'Loan Type', list(LGD_TABLE.keys()), index=3,
        help='Determines the Loss Given Default (LGD)',
    )
    lgd = LGD_TABLE[loan_type]
    st.metric('LGD', f'{lgd:.0%}')
    st.metric('Recovery Rate', f'{(1 - lgd):.0%}')

    st.divider()
    st.caption('Author: Abhishek Ujjawal')
    st.caption('UCD Smurfit Graduate Business School')


# ══════════════════════════════════════════════════════════════
# Load model
# ══════════════════════════════════════════════════════════════


@st.cache_resource
def load_model() -> Tuple[Any, Any]:
    """Load the serialised PD model and scaler."""
    with open(str(MODEL_PATH), 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']


try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error(
        f'Model file not found at `{MODEL_PATH}`.  '
        'Please run `Credit_Risk_Scorecard.py` first to generate it.'
    )
    st.stop()

# ══════════════════════════════════════════════════════════════
# Main content — tabs
# ══════════════════════════════════════════════════════════════

st.title('🏦 Credit Risk Scorecard — IFRS 9 Demo')
st.markdown(
    'Enter borrower details to estimate **Probability of Default**, '
    '**Credit Score**, **IFRS-9 Stage**, and **Expected Credit Loss**.'
)

tab_predict, tab_portfolio, tab_about = st.tabs([
    '📋 Individual Prediction',
    '📊 Portfolio Simulator',
    'ℹ️ About',
])

# ──────────────────────────────────────────────────────────────
# Tab 1: Individual Prediction
# ──────────────────────────────────────────────────────────────

with tab_predict:
    st.subheader('Borrower Details')

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        age = st.slider('Age', AGE_MIN := 18, AGE_MAX := 80, 35,
                         help='Borrower age in years')
        credit_util = st.slider(
            'Revolving Credit Utilisation', 0.0, 1.0, 0.30, 0.01,
            help='Credit card balance ÷ credit limit',
        )
        debt_ratio = st.slider(
            'Debt Ratio', 0.0, 5.0, 0.50, 0.01,
            help='Monthly debt payments ÷ monthly income',
        )

    with col_b:
        monthly_income = st.number_input(
            'Monthly Income ($)', 500, 50_000, 5_000, step=500,
            help='Gross monthly income in USD',
        )
        open_loans = st.number_input(
            'Open Credit Lines', 0, 30, 5,
            help='Total open loans + credit lines',
        )
        dependents = st.number_input(
            'Number of Dependents', 0, 15, 1,
            help='Family members supported',
        )

    with col_c:
        late_30 = st.number_input(
            '30–59 Days Late Count', 0, 10, 0,
            help='Times 30-59 days past due in last 2 years',
        )
        late_60 = st.number_input(
            '60–89 Days Late Count', 0, 10, 0,
            help='Times 60-89 days past due in last 2 years',
        )
        late_90 = st.number_input(
            '90+ Days Late Count', 0, 10, 0,
            help='Times 90+ days past due — most serious',
        )

    loan_amount = st.slider(
        'Loan Amount ($)', 1_000, 500_000, 25_000, step=1_000,
        help='Requested loan amount (used for ECL calculation)',
    )

    # ── Predict button ───────────────────────────
    if st.button('🔍  Predict Credit Risk', type='primary', use_container_width=True):

        # Feature engineering (mirrors training pipeline)
        total_delinquency = late_30 + (2 * late_60) + (3 * late_90)
        income_per_dep = monthly_income / (dependents + 1)
        dti_ratio = debt_ratio * monthly_income / max(monthly_income, 1)
        credit_burden = open_loans / (monthly_income / 1000 + 1)

        input_array = np.array([[
            total_delinquency,
            credit_util,
            late_30,
            age,
            income_per_dep,
            monthly_income,
            debt_ratio,
            dti_ratio,
            open_loans,
            credit_burden,
            dependents,
        ]])

        input_scaled = scaler.transform(input_array)
        pd_raw = model.predict_proba(input_scaled)[0][1]
        pd_prob = prior_correction(pd_raw)
        score = pd_to_score(pd_prob)
        decision, risk, colour = classify_risk(score)
        stage = assign_ifrs9_stage(pd_prob)
        ecl = pd_prob * lgd * loan_amount

        # ── Results display ──────────────────────
        st.divider()
        st.subheader('Prediction Results')

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Probability of Default', f'{pd_prob:.2%}')
        m2.metric('Credit Score', f'{score}')
        m3.metric('Expected Credit Loss', f'${ecl:,.2f}')
        m4.metric('IFRS-9 Stage', stage.split('(')[0].strip())

        st.markdown(
            f'<p style="text-align:center">'
            f'<span class="risk-badge" style="background:{colour}22; '
            f'color:{colour}; border:1px solid {colour}">'
            f'{decision} — {risk}</span></p>',
            unsafe_allow_html=True,
        )

        # ── Risk gauge ───────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=pd_prob * 100,
            number={'suffix': '%', 'font': {'size': 36}},
            delta={'reference': REAL_PRIOR * 100, 'suffix': '%'},
            title={'text': 'Probability of Default', 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 50], 'ticksuffix': '%'},
                'bar': {'color': colour},
                'bgcolor': '#1a1a2e',
                'steps': [
                    {'range': [0, 5],   'color': '#22c55e22'},
                    {'range': [5, 15],  'color': '#f59e0b22'},
                    {'range': [15, 50], 'color': '#ef444422'},
                ],
                'threshold': {
                    'line': {'color': '#f59e0b', 'width': 3},
                    'thickness': 0.8,
                    'value': REAL_PRIOR * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e0e0e0'},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Feature importance ───────────────────
        st.subheader('Top Risk Factors')
        importance = model.coef_[0]
        imp_df = pd.DataFrame({
            'Feature': DISPLAY_NAMES,
            'Impact': importance,
            'Abs_Impact': np.abs(importance),
        }).sort_values('Abs_Impact', ascending=False)

        fig_imp = go.Figure()
        colours = [
            '#ef4444' if v > 0 else '#22c55e'
            for v in imp_df['Impact']
        ]
        fig_imp.add_trace(go.Bar(
            y=imp_df['Feature'],
            x=imp_df['Impact'],
            orientation='h',
            marker_color=colours,
        ))
        fig_imp.update_layout(
            title='Model Coefficients — Red = increases risk, Green = decreases risk',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#1a1a2e',
            font={'color': '#e0e0e0'},
            xaxis_title='Coefficient',
            yaxis={'autorange': 'reversed'},
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── ECL breakdown ────────────────────────
        st.subheader('ECL Breakdown')
        ecl_col1, ecl_col2, ecl_col3 = st.columns(3)
        ecl_col1.metric('PD', f'{pd_prob:.2%}')
        ecl_col2.metric('LGD', f'{lgd:.0%}')
        ecl_col3.metric(
            'EAD', f'${loan_amount:,.0f}',
            help='Exposure at Default = Loan Amount',
        )
        st.info(
            f'**ECL = PD × LGD × EAD** = '
            f'{pd_prob:.4f} × {lgd:.2f} × ${loan_amount:,.0f} = '
            f'**${ecl:,.2f}**'
        )

        # Store in session state
        st.session_state['last_prediction'] = {
            'pd': pd_prob, 'score': score, 'ecl': ecl,
            'decision': decision, 'risk': risk, 'stage': stage,
        }

# ──────────────────────────────────────────────────────────────
# Tab 2: Portfolio Simulator
# ──────────────────────────────────────────────────────────────

with tab_portfolio:
    st.subheader('📊 Portfolio Risk Simulation')
    st.markdown(
        'Simulate a loan portfolio and estimate aggregate ECL '
        'under different stress scenarios.'
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        n_loans = st.number_input('Number of Loans', 100, 100_000, 10_000, step=1_000)
        avg_loan = st.number_input('Average Loan Amount ($)', 1_000, 500_000, 25_000, step=5_000)
    with p_col2:
        avg_pd = st.slider('Average Portfolio PD', 0.01, 0.30, 0.07, 0.01)
        stress_factor = st.slider('Stress Multiplier', 1.0, 3.0, 1.0, 0.1,
                                  help='Multiplier for stressed ECL scenario')

    total_ead = n_loans * avg_loan
    base_ecl = avg_pd * lgd * total_ead
    stressed_ecl = avg_pd * stress_factor * lgd * total_ead

    st.divider()
    ps1, ps2, ps3, ps4 = st.columns(4)
    ps1.metric('Total EAD', f'${total_ead / 1e6:,.1f}M')
    ps2.metric('Base ECL', f'${base_ecl / 1e6:,.2f}M')
    ps3.metric('Stressed ECL', f'${stressed_ecl / 1e6:,.2f}M',
               delta=f'+${(stressed_ecl - base_ecl) / 1e6:,.2f}M')
    ps4.metric('ECL / EAD', f'{stressed_ecl / total_ead * 100:.2f}%')

    # Stage breakdown
    st.markdown('#### IFRS-9 Stage Breakdown (estimated)')
    s1_pct = max(0, 1 - avg_pd * 5)
    s2_pct = min(avg_pd * 3, 0.5)
    s3_pct = 1 - s1_pct - s2_pct

    stage_df = pd.DataFrame({
        'Stage': ['Stage 1', 'Stage 2', 'Stage 3'],
        'Loans': [int(n_loans * s1_pct), int(n_loans * s2_pct), int(n_loans * s3_pct)],
        'ECL ($M)': [
            base_ecl * s1_pct / 1e6 * 0.3,
            base_ecl * s2_pct / 1e6 * 1.5,
            base_ecl * s3_pct / 1e6 * 4.0,
        ],
    })
    st.dataframe(stage_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Tab 3: About
# ──────────────────────────────────────────────────────────────

with tab_about:
    st.subheader('ℹ️ About This Application')
    st.markdown("""
    ### Credit Risk Scorecard — IFRS 9 Demo

    This application demonstrates a **production-grade credit risk
    assessment system** built on industry-standard methodologies:

    | Component | Detail |
    |-----------|--------|
    | **PD Model** | Logistic Regression (Basel III standard) |
    | **Calibration** | Prior probability correction (SMOTE → real-world) |
    | **Scorecard** | 300–850 scale (CIBIL / FICO equivalent) |
    | **ECL** | PD × LGD × EAD (IFRS 9 compliant) |
    | **Staging** | Stage 1 / 2 / 3 per IFRS 9 thresholds |
    | **Explainability** | Model coefficients (SHAP in notebook) |

    ### Methodology

    1. **Data**: Kaggle "Give Me Some Credit" — 150K applicants
    2. **Cleaning**: Winsorisation, median imputation, feature engineering
    3. **Imbalance**: SMOTE (14:1 → 1:1)
    4. **Model**: L2-regularised Logistic Regression
    5. **Calibration**: Prior correction to match 6.68% base rate
    6. **Validation**: 5-fold stratified CV, AUC, Gini, KS, Brier

    ---
    **Author**: Abhishek Ujjawal  
    **Institution**: UCD Michael Smurfit Graduate Business School  
    **Framework**: Basel III + IFRS 9  
    **GitHub**: [ML-Credit-Risk-Scorecard](https://github.com/theabhishekujjawal/ML-Credit-Risk-Scorecard)
    """)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    f'Model: Logistic Regression │ AUC {MODEL_AUC} │ KS {MODEL_KS} │ '
    f'Loan Type: {loan_type} (LGD = {lgd:.0%}) │ '
    f'Author: Abhishek Ujjawal'
)
