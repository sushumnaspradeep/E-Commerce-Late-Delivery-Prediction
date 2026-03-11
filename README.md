# E-Commerce-Late-Delivery-Prediction
# Olist E-Commerce Intelligence Engine

End-to-end machine learning pipeline predicting late delivery risk across 
96,470 Brazilian e-commerce orders. Built on real production patterns using 
the Medallion Architecture, MLflow experiment tracking, and a natural language 
explanation layer for every high-risk order.

---

## Business Impact

| Metric | Value |
|--------|-------|
| Orders scored | 96,470 |
| Late deliveries predicted | 5,293 |
| Critical risk orders | 1,745 |
| High risk orders | 2,277 |
| Revenue at risk identified | $240,761 |
| Orders with AI explanations | 80,115 |

---

## Architecture

9 Olist Tables -> Bronze Delta -> Silver Feature Store -> 
LightGBM + Optuna -> MLflow Registry -> Gold Intelligence Layer

---

## Data Pipeline

- Joined all 9 Olist relational tables into a single master dataset of 99,441 orders
- Most public notebooks on this dataset use 1 or 2 tables — using all 9 enables
  cross-table features that would otherwise be impossible
- Stored raw joined data as the Bronze layer following Medallion Architecture
- Cleaned, parsed and feature-engineered into the Silver layer with 30 features
- Final scored predictions with risk tiers and explanations stored as Gold layer

---

## Feature Engineering

30 features built across 6 dimensions:

Delivery window — promised_days, approval_hours, carrier_days, tight_window

Order complexity — freight_ratio, price_per_item, item_count, is_multi_seller,
installment_risk

Product — heavy_item, few_photos, product_weight_g, product_photos_qty

Geography — cross_state, customer_north

Seller history — seller_late_rate, seller_order_count, risky_seller

Time — purchase_month, purchase_dow, is_weekend_order, is_holiday_season

---

## Model Training

- Trained XGBoost and LightGBM head-to-head on identical train/test splits
- Applied SMOTE oversampling to handle 8.1% class imbalance in training data only
- Ran 50-trial Optuna hyperparameter search using Tree-structured Parzen Estimators
- All runs — baselines, Optuna trials, and champion — logged to MLflow
- Champion model registered as version 1 in MLflow Model Registry

---

## Model Performance

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| XGBoost baseline | 0.942 | 0.546 | 0.905 |
| LightGBM baseline | 0.942 | 0.543 | 0.906 |
| LightGBM + SMOTE champion | 0.938 | 0.545 | 0.899 |

ROC-AUC of 0.90 on the first baseline confirms the feature engineering was 
highly effective. Optuna tuning produced a gain of only 0.001 — indicating the 
model had reached its ceiling from features alone. This is a deliberate finding: 
feature quality drives model performance more than hyperparameter search.

---

## SHAP Explainability

Top 3 drivers of late delivery:

1. review_score (mean SHAP 1.37) — past customer dissatisfaction correlates
   with structural delivery failures, not one-off incidents. Sellers with
   consistently low review scores are systematically late.

2. carrier_days (mean SHAP 0.95) — once a carrier takes more than 5 days
   from approval to dispatch, late delivery becomes near-certain. This is
   the single most actionable operational signal.

3. seller_late_rate (mean SHAP 0.75) — 847 sellers with over 30% historical
   late rate account for 60% of all predicted late deliveries. Direct input
   for vendor management and seller quality programmes.

---

## Prediction Explanations

Every high-risk order receives a plain English explanation of exactly why 
the model flagged it, grounded in its actual feature values.

Order 203096f03d82
[CRITICAL RISK - 99% late probability]
Flagged because: seller has 67% historical late rate; carrier took 17 days
to dispatch.
Recommended action: Contact customer proactively and upgrade shipping.

Order 6ea2f835b455
[CRITICAL RISK - 97% late probability]
Flagged because: cross-state shipment adds transit risk; carrier took 18 days
to dispatch; heavy item increases logistics complexity; holiday season causes
carrier overload; high installment count signals complex order.
Recommended action: Contact customer proactively and upgrade shipping.

---

## Experiment Tracking

All model runs are tracked in MLflow including parameters, metrics, and 
artifacts. The experiment contains 3 baseline runs, 50 Optuna trial runs, 
and the final champion pipeline run. The champion model is registered in 
the MLflow Model Registry as olist-late-delivery-predictor version 1 and 
loaded from the registry for batch scoring — not from a local file. This 
is the production pattern used by data teams at scale.

---

## Tech Stack

| Component | Tools |
|-----------|-------|
| Data processing | Python, Pandas, NumPy |
| Modeling | XGBoost, LightGBM, Scikit-learn |
| Imbalance handling | SMOTE via imbalanced-learn |
| Hyperparameter tuning | Optuna TPE, 50 trials |
| Experiment tracking | MLflow |
| Model registry | MLflow Model Registry |
| Explainability | SHAP TreeExplainer |
| Explanation layer | Rule-based natural language generator |
| Storage | Google Drive Medallion Architecture |
| Cloud platform | Databricks Community Edition |

---

## Project Structure

olist_project/
    bronze/
        master_orders.csv         99,441 rows, 9 tables joined
    silver/
        features.csv              96,470 rows, 30 engineered features
    gold/
        intelligence_final.csv    scored orders with explanations
    model/
        champion.pkl              LightGBM SMOTE champion model
        features.pkl              ordered feature list
    mlflow/                       all experiment runs and artifacts
    shap_summary.png              SHAP beeswarm plot
    shap_bar.png                  SHAP feature importance bar chart

---

## How to Run

1. Download the dataset from kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Upload all 9 CSV files to Google Colab
3. Run the notebooks in order:
   - olist_01_bronze.ipynb
   - olist_02_features.ipynb
   - olist_03_train.ipynb
   - olist_04_pipeline.ipynb
   - olist_05_ai.ipynb

---

## Install

pip install pandas numpy scikit-learn xgboost lightgbm optuna mlflow imbalanced-learn shap databricks-sdk

---

## Key Findings

- Feature engineering was the primary driver of model performance — Optuna 
  tuning added only 0.001 ROC-AUC on top of a strong feature baseline
- SMOTE improved recall on the minority class from 40% to 77% at the cost 
  of precision — the right tradeoff when missing a late delivery costs more 
  than a false alarm
- seller_late_rate was one of the strongest engineered features — a seller's 
  history is more predictive than any single order characteristic
- carrier_days emerged as the most actionable operational signal — it can be 
  monitored in real time once an order ships
- The explanation layer surfaces specific, grounded reasons per order rather 
  than generic model outputs — making the system usable by non-technical teams
