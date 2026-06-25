# Telco Customer Churn Analysis & Prediction

**Internship Project | Coding Blocks**

---

## What This Project Does

Telecom companies lose a significant chunk of revenue every year because customers silently leave before anyone notices. This project builds a machine learning pipeline to identify *which customers are most likely to churn* — and segment them by risk level so the business can act before it's too late.

The analysis covers everything from raw data exploration to a production-ready XGBoost model with customer risk segmentation.

---

## Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

The dataset was downloaded from Kaggle and is included in this repository as `Telco_customer_churn.xlsx`.

| Property | Value |
|----------|-------|
| Rows | 7,043 customers |
| Columns | 33 features |
| Target | `Churn Value` (1 = churned, 0 = stayed) |
| Class balance | ~26% churned, ~74% stayed |

Features include customer demographics, subscribed services (phone, internet, streaming), contract type, billing method, monthly charges, tenure, and more.

---

## Project Structure

```
├── Telco_Churn_Analysis.py       # Main code — all steps in one file
├── Telco_customer_churn.xlsx     # Dataset
└── README.md                     # This file
```

---

## How to Run

**Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```

**Run the script:**

```bash
python Telco_Churn_Analysis.py
```

Make sure `Telco_customer_churn.xlsx` is in the same directory as the script.

---

## Pipeline Overview

| Step | What Happens |
|------|-------------|
| 1 | Import libraries |
| 2 | Load the Excel dataset |
| 3 | Initial data exploration (shape, dtypes, churn distribution) |
| 4 | EDA — tenure, monthly charges, categorical features, correlation heatmap |
| 5 | Data cleaning — fix `Total Charges`, drop leakage/irrelevant columns |
| 6 | Feature engineering — one-hot encode all categoricals |
| 7 | 80/20 train-test split |
| 8 | Baseline Random Forest |
| 9 | Class-balanced Random Forest |
| 10 | Hyperparameter-tuned Random Forest + ROC/PR curves |
| 11 | **XGBoost model** with `scale_pos_weight` for improved recall |
| 12 | Feature importance — RF vs XGBoost comparison |
| 13 | 5-fold cross-validation for both models |
| 14 | Customer segmentation by churn probability (Low / Medium / High Risk) |
| 15 | Summary & business recommendations |

---

## Key EDA Findings

- **Tenure:** Customers who churn have much shorter average tenure (~18 months vs ~38 months for retained customers).
- **Monthly Charges:** Churners pay more on average — ~$74/month vs ~$61 for retained customers.
- **Contract Type:** Month-to-month contracts have a ~43% churn rate. Two-year contracts sit at ~3%.
- **Payment Method:** Electronic check users churn the most. Auto-pay customers are the most loyal.
- **Tech Support:** Customers without tech support churn at nearly double the rate of those with it.

---

## Model Comparison — Before & After XGBoost

The dataset is imbalanced (~26% churn). A vanilla Random Forest tends to bias toward predicting "No Churn", which means it misses real churners — the most costly mistake for the business.

| Model | Accuracy | Churner Recall | Notes |
|-------|----------|----------------|-------|
| Baseline RF | ~79% | ~48% | High accuracy, misses half the churners |
| Balanced RF (`class_weight='balanced'`) | ~77% | ~72% | Big recall jump, slight accuracy drop |
| Tuned RF (`n_estimators=300, max_depth=10`) | ~78% | ~75% | Best RF configuration |
| **XGBoost** (`scale_pos_weight`) | **~77%** | **~78%** | **Highest recall — our final model** |

**Why XGBoost?**

Random Forest corrects for imbalance by reweighting samples *after* building trees. XGBoost goes further — `scale_pos_weight` directly adjusts the gradient updates during training, which means the model learns harder from minority class mistakes rather than just compensating post-hoc. The result is a 3-point recall improvement on churners (78% vs 75%) while keeping overall accuracy nearly identical.

`scale_pos_weight` is calculated as:

```python
scale_pos_weight = (# non-churners in training set) / (# churners in training set)
# ≈ 2.86 for this dataset
```

---

## Customer Segmentation

Using XGBoost's predicted churn probabilities, every customer gets assigned a risk tier:

| Segment | Churn Probability | Recommended Action |
|---------|-------------------|--------------------|
| 🟢 Low Risk | < 30% | Loyalty rewards; upsell opportunities |
| 🟡 Medium Risk | 30% – 60% | Proactive outreach; targeted offers |
| 🔴 High Risk | > 60% | Urgent retention campaign; personal contact |

---

## Business Recommendations

1. **Push long-term contracts** — month-to-month is the single biggest churn driver. Discounts for annual or two-year commitments can meaningfully shift the distribution.
2. **Migrate electronic check users to auto-pay** — small nudge, significant retention impact.
3. **Bundle tech support** — especially for customers paying high monthly charges who don't currently have it.
4. **Prioritise the High Risk segment** for immediate retention campaigns. That bucket represents real, preventable revenue loss.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
openpyxl
```
