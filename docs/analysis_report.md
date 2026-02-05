# Analysis Report: Customer Churn Prediction for E-Commerce

## Identifying At-Risk Customers Using Transaction Data

**Author:** Jamiu Olamilekan Badmus  


---

## 1. Introduction

Customer churn represents one of the most significant challenges facing e-commerce businesses. The cost of acquiring a new customer is estimated to be 5-25 times higher than retaining an existing one, making customer retention a critical business priority.

This analysis develops a machine learning model to predict which customers are at risk of churning, enabling proactive retention strategies. We use transaction data from a UK-based online retailer to engineer predictive features and build classification models.

### 1.1 Objectives

1. Define churn in a non-contractual e-commerce setting
2. Engineer predictive features using RFM analysis and behavioral metrics
3. Build and evaluate machine learning models for churn prediction
4. Provide actionable recommendations for customer retention

### 1.2 Churn Definition

In subscription-based businesses, churn is explicit (customer cancels). In e-commerce, we must infer churn from purchase behavior:

**A customer is considered churned if they:**
- Were active during the feature period (made at least one purchase)
- Did NOT make any purchase during the outcome period (last 90 days)

---

## 2. Data Overview

### 2.1 Dataset Description

The dataset contains all transactions for a UK-based online retailer between December 2010 and December 2011. The company primarily sells unique all-occasion gifts, with many customers being wholesalers.

| Attribute | Value |
|-----------|-------|
| Total Transactions | 541,909 |
| Valid Transactions (after cleaning) | 397,884 |
| Unique Customers | 4,338 |
| Customers in Analysis | 3,370 (with sufficient history) |
| Date Range | Dec 1, 2010 - Dec 9, 2011 (373 days) |
| Countries | 38 |
| Products | ~4,000 unique items |
| Churn Rate | 43.0% |

### 2.2 Data Quality Issues

Several data quality issues were identified and addressed:

1. **Missing CustomerID**: ~25% of transactions lack CustomerID (guest checkouts)
2. **Cancelled Orders**: ~2% of invoices are cancellations (prefix 'C')
3. **Negative Quantities**: Some returns recorded as negative values
4. **Invalid Prices**: Some items have zero or negative prices

---

## 3. Exploratory Analysis

### 3.1 Temporal Patterns

**Key Findings:**
- Strong seasonality with peak sales in November (pre-Christmas)
- Weekday sales significantly higher than weekends
- Peak shopping hours: 10 AM - 3 PM
- UK customers account for ~90% of revenue

### 3.2 Customer Behavior

**Purchase Patterns:**
- Highly skewed distribution: 20% of customers generate 80% of revenue
- Median customer places 2-3 orders over the period
- Average order value: £450-500 (influenced by wholesale customers)
- Repeat purchase rate varies significantly by customer segment

---

## 4. Feature Engineering

### 4.1 RFM Analysis

RFM (Recency, Frequency, Monetary) is a proven customer segmentation technique:

| Feature | Description | Churn Correlation |
|---------|-------------|-------------------|
| **Recency** | Days since last purchase | Strong positive |
| **Frequency** | Number of orders | Strong negative |
| **Monetary** | Total spend | Moderate negative |

**Key Insight:** Recency is the strongest single predictor of churn.

### 4.2 Behavioral Features

Additional features engineered from transaction data:

| Feature | Description |
|---------|-------------|
| AvgOrderValue | Average spending per order |
| AvgDaysBetweenPurchases | Mean purchase interval |
| StdDaysBetweenPurchases | Variability in purchase timing |
| UniqueProducts | Number of different products purchased |
| ItemsPerOrder | Average items per transaction |
| CancellationRate | Proportion of cancelled orders |
| ProductDiversityRatio | Unique products / Total items |
| OrdersPerMonth | Purchase frequency normalized by tenure |

### 4.3 Feature Analysis by Churn Status

| Feature | Retained (Mean) | Churned (Mean) | % Difference |
|---------|----------------|----------------|--------------|
| Recency | 74 days | 124 days | +66% (churned higher) |
| Frequency | 4.8 orders | 1.9 orders | -61% (churned lower) |
| TotalSpend | £2,301 | £714 | -69% (churned lower) |
| AvgOrderValue | £30 | £103 | +243% (churned higher*) |
| UniqueProducts | 66 products | 29 products | -56% (churned lower) |
| CancellationRate | 2% | 3% | +13% (churned higher) |

*Note: Higher avg order value for churned customers reflects fewer, larger one-time purchases vs. regular smaller purchases for retained customers.

---

## 5. Model Development

### 5.1 Class Imbalance

The dataset exhibits moderate class imbalance:
- Retained customers: 1,921 (57%)
- Churned customers: 1,449 (43%)
- Class ratio: 0.75:1

SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance training data, increasing training samples from 2,696 to 3,074.

### 5.2 Models Evaluated

Six classification algorithms were evaluated:

1. **Logistic Regression**: Linear baseline model
2. **Decision Tree**: Interpretable single tree
3. **Random Forest**: Bagged ensemble of trees
4. **Gradient Boosting**: Sequential boosting ensemble
5. **XGBoost**: Optimized gradient boosting
6. **LightGBM**: Efficient gradient boosting

### 5.3 Evaluation Framework

- 5-fold stratified cross-validation
- 80/20 train/test split with stratification
- Primary metric: ROC-AUC
- Secondary metrics: Precision, Recall, F1

---

## 6. Results

### 6.1 Model Performance

Six machine learning models were evaluated using 5-fold stratified cross-validation:

| Model | CV ROC-AUC | CV Std | Test ROC-AUC | Test F1 |
|-------|------------|--------|--------------|---------|
| **LightGBM** | **0.83** | **0.02** | **0.77** | **0.68** |
| Gradient Boosting | 0.83 | 0.02 | 0.76 | 0.68 |
| XGBoost | 0.83 | 0.02 | 0.75 | 0.65 |
| Random Forest | 0.80 | 0.03 | 0.74 | 0.65 |
| Logistic Regression | 0.77 | 0.03 | 0.74 | 0.68 |
| Decision Tree | 0.72 | 0.01 | 0.64 | 0.61 |

**Best Model: LightGBM** achieved the highest test ROC-AUC of **0.7677** with excellent recall of **75.5%** for identifying churned customers.

### 6.2 Feature Importance

Top predictive features identified from LightGBM model:

1. **TotalCancellations** (Importance: 170): Number of cancelled orders
2. **CancellationRate** (Importance: 165): Proportion of cancelled orders
3. **UniqueProducts** (Importance: 152): Product diversity purchased
4. **Tenure** (Importance: 101): Customer relationship length
5. **StdOrderValue** (Importance: 100): Variability in order values
6. **AvgUnitPrice** (Importance: 91): Average price of items purchased
7. **ItemsPerOrder** (Importance: 82): Shopping basket size
8. **AvgOrderValue** (Importance: 77): Average spending per order
9. **StdDaysBetweenPurchases** (Importance: 77): Purchase timing variability
10. **Recency** (Importance: 75): Days since last purchase

### 6.3 SHAP Analysis

SHAP values provide interpretable feature contributions:

- **High TotalCancellations** → Strong positive impact on churn probability (customers who cancel more tend to churn)
- **High CancellationRate** → Strong positive impact on churn probability
- **Low Frequency** (few orders) → Positive impact on churn probability
- **Low UniqueProducts** (narrow product interest) → Positive impact on churn probability
- **High RecencyFrequencyRatio** → Positive impact on churn probability

---

## 7. Business Recommendations

### 7.1 Customer Segmentation

Based on model predictions, customers are segmented by churn probability risk:

| Segment | Churn Probability | Size | Actual Churn Rate | Recommended Action |
|---------|-------------------|------|-------------------|-------------------|
| High Risk | >60% | 274 customers (41%) | 64% | Immediate intervention |
| Medium Risk | 30-60% | 167 customers (25%) | 50% | Proactive engagement |
| Low Risk | <30% | 233 customers (34%) | 13% | Loyalty maintenance |

**Model Performance by Segment:**
- Low Risk segment: Only 13% actually churned (high retention)
- High Risk segment: 64% actually churned (accurate identification)

### 7.2 Retention Strategies

**For High-Risk Customers:**
- Personalized win-back campaigns
- Special discount offers
- Direct outreach from customer success

**For Medium-Risk Customers:**
- Re-engagement email sequences
- Product recommendations based on history
- Loyalty program incentives

**For Low-Risk Customers:**
- Satisfaction surveys
- Referral programs
- Early access to new products

### 7.3 Implementation Roadmap

1. **Immediate (Week 1-2):**
   - Deploy model for scoring existing customers
   - Identify high-risk customers for immediate outreach

2. **Short-term (Month 1-2):**
   - Set up automated recency-based triggers
   - Implement personalized email campaigns

3. **Long-term (Quarter 1-2):**
   - A/B test intervention effectiveness
   - Refine model with new data
   - Integrate with CRM system

---

## 8. Limitations and Considerations

### 8.1 Data Limitations

- **Single Year**: Model trained on one year of data; may not capture long-term trends
- **Missing CustomerIDs**: ~25% of transactions excluded due to missing customer identification
- **B2B vs B2C**: Mix of wholesale and retail customers may have different churn patterns

### 8.2 Model Limitations

- **90-Day Window**: Churn definition uses fixed 90-day window; optimal window may vary
- **Binary Classification**: Churn is treated as binary; in reality, it's a continuum
- **Static Features**: Features computed at single point; time-series patterns not fully captured

### 8.3 Ethical Considerations

- **Privacy**: Customer data must be handled according to GDPR and other regulations
- **Discrimination**: Model should be audited for unfair treatment of customer segments
- **Transparency**: Customers should understand how their data is used

---

## 9. Conclusions

This analysis demonstrates that customer churn in e-commerce can be effectively predicted using transaction-derived features. The best model (LightGBM) achieved **ROC-AUC of 0.77** with **75.5% recall** in identifying churned customers.

**Key findings:**

1. **Cancellation behavior is highly predictive**: Contrary to traditional RFM focus, TotalCancellations and CancellationRate emerged as the strongest predictors, indicating that order cancellation patterns signal customer dissatisfaction.

2. **Product diversity matters**: Customers who purchase from a narrow range of products are more likely to churn, suggesting that broader product exploration indicates stronger engagement.

3. **RFM features remain valuable**: Recency, Frequency, and Monetary value continue to be important predictors, validating their enduring relevance in customer analytics.

4. **Early intervention is key**: With 64% accuracy on high-risk customers and only 13% churn in low-risk segment, the model effectively separates customers for targeted interventions.

5. **Actionable deployment**: The model enables automated customer scoring and risk-based segmentation for immediate business use.

---

## References

1. UCI Machine Learning Repository - Online Retail Dataset
2. Chambers, B. & Zaharia, M. (2018). Spark: The Definitive Guide. O'Reilly Media.
3. Fader, P. S., & Hardie, B. G. (2009). Probability models for customer-base analysis. Journal of Interactive Marketing.
4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS 2017.

---

*For questions or collaboration opportunities, contact jamiubadmus001@gmail.com.*
