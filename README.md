# Predicting Country-Level AML Risk Using Machine Learning

A cross-national analysis of governance, financial access and corruption indicators across 140 countries, using the Basel AML Index 2023 as the outcome variable.

**Author:** Haribalu V
**Program:** Walsh College, QM640 Data Analytics Capstone, Term 3, Spring 2026
**Mentor:** Vikas S
**License:** MIT

## Problem

Money laundering moves between USD 800 billion and USD 2 trillion through the global financial system each year, equal to 2 to 5 percent of world GDP (UNODC, 2023). Financial institutions and regulators need country-level risk signals to set correspondent banking limits, prioritise FATF mutual evaluations and direct supervisory attention. The Basel AML Index publishes such a score for 177 jurisdictions, but it does not expose which governance and financial factors drive the score, nor does it classify countries into actionable risk tiers with quantified accuracy.

This project closes that gap. It builds a reproducible machine learning pipeline that links the Basel AML Index 2023 score to eight publicly available governance, corruption and financial access indicators, and answers four research questions with statistical rigour.

## Research Questions

**RQ1.** Which governance and financial indicators most predict a country's Basel AML Index score?
Method: LASSO regression with 10-fold cross-validated regularisation parameter.

**RQ2.** Can machine learning models classify countries into High, Medium and Low AML risk tiers with acceptable performance?
Method: Logistic Regression, Random Forest and XGBoost on a tertile-based three-class target.

**RQ3.** Does financial account ownership moderate the effect of corruption on AML risk?
Method: Hierarchical moderation regression with an F-test for incremental R-squared.

**RQ4.** What distinct country typologies emerge from AML vulnerability profiles?
Method: K-Means clustering with silhouette-based k selection and Ward-linkage hierarchical validation.

## Key Results

The LASSO model retains 7 of 8 predictors at lambda = 0.0063 and explains 73.3 percent of Basel score variance. WGI Regulatory Quality carries the strongest standardised coefficient at -0.686, followed by Rule of Law at -0.400. The Transparency International Corruption Perceptions Index is shrunk to exactly zero because it shares 99 percent of its variance with WGI Control of Corruption.

Random Forest is the best classifier with test accuracy of 75.0 percent, macro F1 of 0.736 and AUC-ROC of 0.881. Logistic Regression and XGBoost both reach F1 of 0.631. The Medium-risk tier is the hardest to separate, with recall of 44.4 percent across all three models, because mid-range Basel scores sit on the boundary between adjacent governance profiles.

Moderation analysis does not reject the null. The Corruption x Financial Access interaction adds only 0.0006 to R-squared, F(1, 135) = 0.233, p = 0.630, so financial access does not measurably change the effect of corruption on AML risk. Financial inclusion expansion needs to be paired with governance reform to reduce country-level risk.

Clustering identifies two natural groups at silhouette = 0.456. Cluster 1 contains 63 countries with strong governance scores, 87 percent financial account ownership and average Basel score of 4.39. Cluster 2 contains 77 countries with weak governance, 55 percent account ownership and average Basel score of 6.05.

## Data

The analytical sample is 140 countries with complete data across nine variables for the year 2023. Five public data sources contribute:

| Variable | Source | Scale | Role |
|---|---|---|---|
| Basel AML Index Score | Basel Institute on Governance | 0 to 10 | Outcome |
| WGI Control of Corruption | World Bank WGI | -2.5 to +2.5 (standardised) | Predictor |
| WGI Rule of Law | World Bank WGI | -2.5 to +2.5 (standardised) | Predictor |
| WGI Government Effectiveness | World Bank WGI | -2.5 to +2.5 (standardised) | Predictor |
| WGI Regulatory Quality | World Bank WGI | -2.5 to +2.5 (standardised) | Predictor |
| Corruption Perceptions Index | Transparency International | 0 to 100 | Predictor |
| Bank Branches per 100,000 Adults | IMF FAS via World Bank | Count ratio | Predictor |
| Financial Account Ownership | World Bank Global Findex | 0 to 100 percent | Predictor |
| GDP per Capita (log) | World Bank | Log USD | Control |

The merge key is the ISO 3166-1 alpha-3 country code. Countries missing more than 20 percent of predictor values are excluded. Remaining gaps are imputed by column median. GDP per capita is log-transformed to reduce right skew before standardisation.

## Repository Structure

```
aml-risk-capstone/
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.py
├── .gitignore
├── data/
│   ├── raw/                Raw inputs from each public source
│   └── processed/
│       └── merged_dataset_2023.csv
├── src/
│   ├── 01_data_collection.py
│   ├── 02_eda.py
│   ├── 03_feature_preparation.py
│   ├── 04_model_rq1_lasso.py
│   ├── 05_model_rq2_classification.py
│   ├── 06_model_rq3_moderation.py
│   └── 07_model_rq4_clustering.py
├── outputs/
│   ├── figures/            11 publication-ready PNG figures
│   └── results/            9 result tables in CSV
└── docs/
    └── AML_Capstone_Final_Report.docx
```

## Reproducing the Analysis

The full pipeline runs end-to-end in roughly 15 seconds on a standard laptop CPU.

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` executes the seven scripts under `src/` in sequence, refreshes every figure under `outputs/figures/` and every result table under `outputs/results/`. Individual stages can run in isolation:

```
python src/04_model_rq1_lasso.py
python src/05_model_rq2_classification.py
python src/06_model_rq3_moderation.py
python src/07_model_rq4_clustering.py
```

## Reproducibility Notes

Random seed is fixed at 42 in every script. All file paths resolve relative to the script location, so the project runs without modification on macOS, Linux and Windows. The fitted StandardScaler in RQ2 is trained on the training split only and then applied to the test split, which prevents test-set statistics from leaking into the model. Python 3.9 or later is required.

Green's Rule for multiple regression sets the minimum sample size at 50 + 8 x 8 = 114 for an eight-predictor model. The analytical sample of 140 countries clears this threshold by 23 percent.

## Citation

Haribalu V. (2026). Predicting country-level AML risk using machine learning: A cross-national analysis of governance, financial access and corruption indicators. Walsh College QM640 Capstone.
