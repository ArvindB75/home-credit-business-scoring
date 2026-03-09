# The Underbanked Opportunity: What Alternative Data Reveals About Hidden Good Risks in Credit Scoring

A credit scoring model for underbanked populations, built on the
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset. The goal is not to maximize AUC but to build a deployable,
interpretable model that a risk committee can actually use.

---

## Business Context

Home Credit serves clients with limited or no formal credit history.
Standard scoring models reject these clients by default. This project
demonstrates that alternative data signals can identify good risks
within this population, unlocking a quantifiable revenue opportunity
while improving portfolio quality.

**Three business questions driving the analysis :**
1. Which underbanked profiles are worth financing ?
2. What threshold matches different risk appetites ?
3. What is the revenue opportunity in the good risk / underbanked segment ?

---

## Key Findings

| Metric | Value |
|---|---|
| Model | LightGBM + Isotonic Calibration |
| AUC-ROC | 0.780 |
| KS Statistic | 0.42 (above 0.30 deployment threshold) |
| Optimal threshold | 0.12 |
| Portfolio default rate at threshold | 4.0% vs 8.1% baseline |
| Acceptance rate | 91.9% |

**Alternative data dominates :**
EXT_SOURCE variables alone account for 25.9% of total predictive power,
vs 7.7% for six traditional variables combined. Per variable,
alternative data is 3x more informative than declared income or employment.

**The underbanked opportunity :**
10% of the portfolio consists of good risk / underbanked clients
with an actual default rate of 3.6%. Net revenue opportunity : 771M.
Cost of refusing this segment : 890M in missed interest income.

---

## Risk Segmentation

| Tier | Default Probability | Actual Default Rate | Share of Portfolio |
|---|---|---|---|
| Tier 1 — Prime | < 5% | 2.9% | 52.5% |
| Tier 2 — Standard | 5–10% | 7.5% | 22.9% |
| Tier 3 — Elevated | 10–15% | 13.3% | 9.0% |
| Tier 4 — High Risk | > 15% | 23.5% | 15.6% |

---

## Project Structure
```
home-credit-business-scoring/
├── data/                   # Raw data (not versioned)
├── notebooks/
│   └── home_credit_scoring.ipynb
├── outputs/                # Charts and saved artifacts
│   ├── 01_target_distribution.png
│   ├── 02_benchmark_sectoriel.png
│   ├── 03_demographic_analysis.png
│   ├── 04_income_employment.png
│   ├── 05_credit_variables.png
│   ├── 06_ext_sources.png
│   ├── 07_distress_signals.png
│   ├── 08_calibration.png
│   ├── 09_model_evaluation.png
│   ├── 10_threshold_optimization.png
│   ├── 11_shap_global.png
│   └── 12_segmentation_shap_local.png
├── src/                    # Reusable modules (future)
└── README.md
```

---

## Methodology

**Section 0** — Setup and data loading  
**Section 1** — Business framing, cost matrix, stakeholder mapping  
**Section 2** — EDA : 6 key findings, sector benchmarks  
**Section 3** — Feature engineering : 116 features from 6 source tables  
**Section 4** — Modeling : Logistic Regression baseline, LightGBM,
hyperparameter optimization, probability calibration, threshold selection  
**Section 5** — SHAP interpretability : global and local analysis  
**Section 6** — Business recommendations : 4 actionable levers  

---

## Stack

- Python 3.10
- pandas, numpy
- scikit-learn
- lightgbm
- shap
- matplotlib, seaborn

---

## Regulatory Notes

Two blockers exist for EU production deployment :

1. **EXT_SOURCE opacity** — strongest predictors but unknown composition,
   incompatible with GDPR Article 22 right to explanation.
   Mitigation : replace with auditable open banking or telco data.

2. **Gender as predictor** — 5th most important SHAP feature,
   raises concerns under EU Equal Treatment Directive.
   Mitigation : remove from feature set, expected AUC impact ~0.003.

---

## Dashboard

Interactive dashboard available on Tableau Public :
[The Underbanked Opportunity — Credit Scoring Dashboard](https://public.tableau.com/app/profile/arvind.bajolah/viz/TheUnderbankedOpportunityCreditScoringDashboard/Tableaudebord1)

---

## Reproduce
```bash
git clone https://github.com/ArvindB75/home-credit-business-scoring
cd home-credit-business-scoring
conda create -n home-credit python=3.10
conda activate home-credit
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn imbalanced-learn
# Add data files to data/ directory from Kaggle competition
jupyter notebook notebooks/home_credit_scoring.ipynb
```
