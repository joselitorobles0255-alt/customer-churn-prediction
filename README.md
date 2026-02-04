# Customer Churn Prediction for E-Commerce

## Identifying At-Risk Customers Using Transaction Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Source](https://img.shields.io/badge/Data-UCI%20ML%20Repository-orange.svg)](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

**Author:** Jamiu Olamilekan Badmus  
**Email:** jamiubadmus001@gmail.com  
**LinkedIn:** [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)  
**GitHub:** [jamiubadmusng](https://github.com/jamiubadmusng)

---

## Executive Summary

This project develops a machine learning model to predict customer churn in e-commerce, enabling proactive retention strategies. Using transaction data from a UK-based online retailer, we engineer RFM (Recency, Frequency, Monetary) and behavioral features to identify customers at risk of churning.

**Key Results:**
- **Best Model**: LightGBM with **76.77% ROC-AUC** and **75.5% recall**
- Built and evaluated 6 classification models on 3,370 customers
- Engineered 24 predictive features from 397,884 transactions
- **43% churn rate** identified in the dataset
- Identified key churn indicators: Cancellation behavior, Frequency, and Purchase Diversity

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Data Source](#data-source)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Key Findings](#key-findings)
9. [Business Recommendations](#business-recommendations)
10. [Future Work](#future-work)

---

## Problem Statement

Customer churn is a critical challenge for e-commerce businesses:

- **Acquiring a new customer costs 5-25x more** than retaining an existing one
- **Increasing retention by 5%** can increase profits by 25-95%
- The probability of selling to an existing customer is 60-70% vs. 5-20% for new prospects

This project addresses: **How can we identify customers at risk of churning before they leave, enabling proactive retention interventions?**

### Churn Definition

In a non-contractual e-commerce setting, churn is not explicit. We define a customer as **churned** if they:
- Were active during the feature period (first 9 months)
- Did NOT make any purchase during the outcome period (last 3 months)

---

## Data Source

The dataset is the **Online Retail Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Source | UCI ML Repository / Spark Definitive Guide |
| Time Period | December 2010 - December 2011 |
| Total Transactions | 541,909 |
| Unique Customers | ~4,300 (with CustomerID) |
| Features | 8 original columns |
| Business Type | UK-based online gift retailer |

### Data Dictionary

| Column | Description |
|--------|-------------|
| InvoiceNo | Invoice number (C prefix = cancellation) |
| StockCode | Product code |
| Description | Product name |
| Quantity | Quantity per transaction |
| InvoiceDate | Transaction date and time |
| UnitPrice | Unit price in GBP |
| CustomerID | Customer identifier |
| Country | Customer country |

---

## Project Structure

```
e-commerce/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── online_retail.csv
│   └── processed/                    # Feature-engineered data
│       └── customer_features.csv
├── docs/
│   ├── analysis_report.md            # Detailed analysis write-up
│   └── figures/                      # Visualization outputs
│       ├── temporal_analysis.png
│       ├── customer_distributions.png
│       ├── rfm_distributions.png
│       ├── churn_distribution.png
│       ├── feature_comparison.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_pr_curves.png
│       ├── feature_importance.png
│       ├── shap_importance.png
│       └── shap_beeswarm.png
├── models/                           # Trained model artifacts
│   ├── churn_model.joblib
│   └── scaler.joblib
├── notebooks/
│   └── customer_churn_prediction.ipynb  # Main analysis notebook
├── src/
│   └── predict_churn.py              # Standalone Python module
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jamiubadmusng/ecommerce-churn-predictor.git
   cd ecommerce-churn-predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Jupyter Notebook

```bash
cd notebooks
jupyter notebook customer_churn_prediction.ipynb
```

### Running the Python Script

```bash
python src/predict_churn.py --input data/raw/online_retail.csv
```

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/churn_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Prepare customer features (see notebook for feature engineering)
customer_data = pd.read_csv('data/processed/customer_features.csv')

# Make predictions
predictions = model.predict(scaler.transform(customer_data[feature_cols]))
probabilities = model.predict_proba(scaler.transform(customer_data[feature_cols]))[:, 1]
```

---

## Methodology

### 1. Data Cleaning
- Removed transactions without CustomerID (required for customer-level analysis)
- Handled cancelled orders (InvoiceNo starting with 'C')
- Filtered invalid quantities and prices

### 2. Feature Engineering

**RFM Features:**
| Feature | Description |
|---------|-------------|
| Recency | Days since last purchase |
| Frequency | Number of orders |
| Monetary | Total spend |

**Behavioral Features:**
| Feature | Description |
|---------|-------------|
| AvgOrderValue | Average value per order |
| AvgDaysBetweenPurchases | Purchase interval |
| UniqueProducts | Product diversity |
| CancellationRate | Order cancellation frequency |
| ItemsPerOrder | Average items per order |
| OrdersPerMonth | Purchase frequency |

### 3. Churn Labeling
- Split data: Feature period (9 months) + Outcome period (3 months)
- Churned = Active in feature period, inactive in outcome period

### 4. Model Training
- Applied SMOTE for class imbalance
- Trained 6 models with 5-fold cross-validation
- Evaluated on held-out test set (20%)

---

## Results

### Model Performance Comparison

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|-------|------------|--------------|---------|
| **LightGBM** | **0.83** | **0.77** | **0.68** |
| Gradient Boosting | 0.83 | 0.76 | 0.68 |
| XGBoost | 0.83 | 0.75 | 0.65 |
| Random Forest | 0.80 | 0.74 | 0.65 |
| Logistic Regression | 0.77 | 0.74 | 0.68 |
| Decision Tree | 0.72 | 0.64 | 0.61 |

### Feature Importance

Top predictive features for customer churn (LightGBM):
1. **TotalCancellations** (170) - Number of cancelled orders
2. **CancellationRate** (165) - Proportion of cancelled orders
3. **UniqueProducts** (152) - Product diversity purchased
4. **Tenure** (101) - Customer relationship length
5. **StdOrderValue** (100) - Variability in order values
6. **AvgUnitPrice** (91) - Average price of items purchased
7. **ItemsPerOrder** (82) - Shopping basket size

---

## Key Findings

### 1. Cancellation Behavior is the Strongest Predictor
Contrary to traditional RFM focus, **order cancellations** emerged as the strongest predictor. Customers with higher cancellation rates are significantly more likely to churn, indicating dissatisfaction.

### 2. Product Diversity Indicates Engagement
Customers who purchase from a **narrow range of products** (low UniqueProducts) are more likely to churn. Those with broader product exploration show stronger engagement.

### 3. Low-Frequency Customers Are High Risk
One-time buyers have the highest churn probability (retained customers average 4.8 orders vs 1.9 for churned). Implementing post-purchase follow-ups and loyalty programs is critical.

### 4. Early Warning Signs
- Increasing order cancellations
- Low product diversity (fewer unique products)
- Long gaps between purchases (high Recency)
- Single or very few orders (low Frequency)
- Lower total spending

---

## Business Recommendations

### Retention Strategies by Risk Level

| Risk Level | Churn Probability | Actual Churn Rate | Recommended Action |
|------------|-------------------|-------------------|-------------------|
| High Risk | >60% | **64%** (274 customers) | Immediate outreach, special offers |
| Medium Risk | 30-60% | **50%** (167 customers) | Re-engagement emails, product recommendations |
| Low Risk | <30% | **13%** (233 customers) | Loyalty rewards, satisfaction surveys |

### Implementation Roadmap

1. **Automated Triggers**: Set up email campaigns triggered by recency thresholds
2. **Loyalty Program**: Reward repeat purchases to increase frequency
3. **Personalization**: Recommend products based on purchase history
4. **Proactive Support**: Reach out to high-risk customers before they churn

---

## Future Work

1. **Real-time Scoring**: Deploy model as API for real-time churn predictions
2. **Customer Lifetime Value**: Combine churn prediction with CLV estimation
3. **A/B Testing**: Test intervention effectiveness on high-risk segments
4. **Deep Learning**: Explore sequential models for purchase pattern analysis

---

## References

1. UCI Machine Learning Repository - Online Retail Dataset
2. Chambers, B. & Zaharia, M. (2018). Spark: The Definitive Guide. O'Reilly Media.
3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS 2017.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration opportunities:

- **Email**: jamiubadmus001@gmail.com
- **LinkedIn**: [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)
- **GitHub**: [jamiubadmusng](https://github.com/jamiubadmusng)
