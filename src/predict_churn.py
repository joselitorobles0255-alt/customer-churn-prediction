"""
Customer Churn Prediction for E-Commerce
=========================================

This module provides functions to predict customer churn using transaction data
from e-commerce platforms. It implements RFM analysis and machine learning models
to identify customers at risk of churning.

Author: Jamiu Olamilekan Badmus
Email: jamiubadmus001@gmail.com
GitHub: https://github.com/jamiubadmusng

Usage:
------
    python predict_churn.py --input data/raw/online_retail.csv

"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os
import joblib

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


# =============================================================================
# Data Loading and Cleaning
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the e-commerce transaction dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing transaction data.
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]:,} transactions with {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transaction data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataset.
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset with valid transactions.
    """
    df_clean = df.copy()
    
    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Remove rows without CustomerID
    df_clean = df_clean[df_clean['CustomerID'].notna()]
    
    # Identify cancelled orders
    df_clean['IsCancelled'] = df_clean['InvoiceNo'].astype(str).str.startswith('C')
    
    # Remove cancelled orders for main analysis
    df_orders = df_clean[~df_clean['IsCancelled']].copy()
    
    # Remove invalid quantities and prices
    df_orders = df_orders[(df_orders['Quantity'] > 0) & (df_orders['UnitPrice'] > 0)]
    
    # Calculate total amount
    df_orders['TotalAmount'] = df_orders['Quantity'] * df_orders['UnitPrice']
    
    print(f"Cleaned dataset: {len(df_orders):,} transactions")
    print(f"Unique customers: {df_orders['CustomerID'].nunique():,}")
    
    return df_orders, df_clean


# =============================================================================
# Feature Engineering
# =============================================================================

def create_churn_labels(df: pd.DataFrame, outcome_window: int = 90) -> tuple:
    """
    Create churn labels based on customer activity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.
    outcome_window : int
        Number of days for outcome period.
        
    Returns
    -------
    tuple
        (feature_period_data, churn_labels, split_date)
    """
    data_end = df['InvoiceDate'].max()
    split_date = data_end - timedelta(days=outcome_window)
    
    # Split data
    feature_period = df[df['InvoiceDate'] <= split_date]
    outcome_period = df[df['InvoiceDate'] > split_date]
    
    # Identify customers in each period
    customers_feature = set(feature_period['CustomerID'].unique())
    customers_outcome = set(outcome_period['CustomerID'].unique())
    
    # Churned = Active in feature period but NOT in outcome period
    churned_customers = customers_feature - customers_outcome
    
    # Create labels
    churn_labels = pd.DataFrame({'CustomerID': list(customers_feature)})
    churn_labels['Churned'] = churn_labels['CustomerID'].apply(
        lambda x: 1 if x in churned_customers else 0
    )
    
    print(f"Feature period: {feature_period['InvoiceDate'].min().date()} to {split_date.date()}")
    print(f"Outcome period: {split_date.date()} to {data_end.date()}")
    print(f"Churn rate: {churn_labels['Churned'].mean()*100:.2f}%")
    
    return feature_period, churn_labels, split_date


def engineer_features(df: pd.DataFrame, df_full: pd.DataFrame, 
                      ref_date: datetime, customers: set) -> pd.DataFrame:
    """
    Engineer customer-level features from transaction data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature period transaction data.
    df_full : pd.DataFrame
        Full cleaned data (including cancellations).
    ref_date : datetime
        Reference date for recency calculation.
    customers : set
        Set of customer IDs to include.
        
    Returns
    -------
    pd.DataFrame
        Customer features dataframe.
    """
    # Basic RFM features
    customer_features = df.groupby('CustomerID').agg({
        'InvoiceDate': [
            lambda x: (ref_date - x.max()).days,  # Recency
            lambda x: (ref_date - x.min()).days,  # Tenure
        ],
        'InvoiceNo': 'nunique',  # Frequency
        'TotalAmount': ['sum', 'mean', 'std', 'max'],  # Monetary
        'Quantity': ['sum', 'mean'],
        'UnitPrice': ['mean', 'std'],
        'StockCode': 'nunique',
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = [
        'CustomerID', 'Recency', 'Tenure', 'Frequency',
        'TotalSpend', 'AvgOrderValue', 'StdOrderValue', 'MaxOrderValue',
        'TotalQuantity', 'AvgQuantityPerOrder',
        'AvgUnitPrice', 'StdUnitPrice',
        'UniqueProducts'
    ]
    
    # Fill NaN standard deviations
    customer_features['StdOrderValue'] = customer_features['StdOrderValue'].fillna(0)
    customer_features['StdUnitPrice'] = customer_features['StdUnitPrice'].fillna(0)
    
    # Purchase intervals
    def calc_intervals(group):
        dates = group['InvoiceDate'].sort_values()
        if len(dates) < 2:
            return pd.Series({'AvgDaysBetweenPurchases': 0, 'StdDaysBetweenPurchases': 0})
        intervals = dates.diff().dt.days.dropna()
        return pd.Series({
            'AvgDaysBetweenPurchases': intervals.mean(),
            'StdDaysBetweenPurchases': intervals.std() if len(intervals) > 1 else 0
        })
    
    interval_features = df.groupby('CustomerID').apply(calc_intervals).reset_index()
    interval_features['StdDaysBetweenPurchases'] = interval_features['StdDaysBetweenPurchases'].fillna(0)
    customer_features = customer_features.merge(interval_features, on='CustomerID', how='left')
    
    # Time features
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour
    
    time_features = df.groupby('CustomerID').agg({
        'DayOfWeek': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
        'Hour': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12,
    }).reset_index()
    time_features.columns = ['CustomerID', 'PreferredDay', 'PreferredHour']
    customer_features = customer_features.merge(time_features, on='CustomerID', how='left')
    
    # Cancellation features
    cancel_features = df_full[df_full['CustomerID'].isin(customers)].groupby('CustomerID').agg({
        'IsCancelled': ['sum', 'mean']
    }).reset_index()
    cancel_features.columns = ['CustomerID', 'TotalCancellations', 'CancellationRate']
    customer_features = customer_features.merge(cancel_features, on='CustomerID', how='left')
    customer_features['TotalCancellations'] = customer_features['TotalCancellations'].fillna(0)
    customer_features['CancellationRate'] = customer_features['CancellationRate'].fillna(0)
    
    # Geographic feature
    country_features = df.groupby('CustomerID')['Country'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    ).reset_index()
    country_features.columns = ['CustomerID', 'Country']
    customer_features = customer_features.merge(country_features, on='CustomerID', how='left')
    customer_features['IsUK'] = (customer_features['Country'] == 'United Kingdom').astype(int)
    
    # Derived features
    customer_features['ItemsPerOrder'] = customer_features['TotalQuantity'] / customer_features['Frequency']
    customer_features['SpendPerItem'] = customer_features['TotalSpend'] / customer_features['TotalQuantity']
    customer_features['OrdersPerMonth'] = customer_features['Frequency'] / (customer_features['Tenure'] / 30 + 1)
    customer_features['ProductDiversityRatio'] = customer_features['UniqueProducts'] / customer_features['TotalQuantity']
    customer_features['RecencyFrequencyRatio'] = customer_features['Recency'] / (customer_features['Frequency'] + 1)
    
    return customer_features


# =============================================================================
# Model Training
# =============================================================================

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                use_smote: bool = True) -> tuple:
    """
    Train the churn prediction model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    use_smote : bool
        Whether to apply SMOTE for class imbalance.
        
    Returns
    -------
    tuple
        (trained_model, scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Handle class imbalance
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
    
    # Train model
    model = GradientBoostingClassifier(
        random_state=42, 
        n_estimators=100, 
        max_depth=5
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    return model, scaler


def evaluate_model(model, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the trained model on test data.
    
    Parameters
    ----------
    model : trained model
        The trained classifier.
    scaler : StandardScaler
        The fitted scaler.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))


# =============================================================================
# Main Pipeline
# =============================================================================

def main(input_path: str, output_dir: str = None):
    """
    Run the complete churn prediction pipeline.
    
    Parameters
    ----------
    input_path : str
        Path to the input transaction data.
    output_dir : str
        Directory to save model artifacts.
    """
    print("="*60)
    print("CUSTOMER CHURN PREDICTION PIPELINE")
    print("="*60)
    
    # Load and clean data
    print("\n1. Loading and cleaning data...")
    df_raw = load_data(input_path)
    df_orders, df_clean = clean_data(df_raw)
    
    # Create churn labels
    print("\n2. Creating churn labels...")
    feature_period, churn_labels, split_date = create_churn_labels(df_orders)
    
    # Engineer features
    print("\n3. Engineering features...")
    customers = set(churn_labels['CustomerID'])
    ref_date = split_date + timedelta(days=1)
    customer_features = engineer_features(feature_period, df_clean, ref_date, customers)
    
    # Merge with labels
    df_final = customer_features.merge(churn_labels, on='CustomerID', how='inner')
    
    # Prepare for modeling
    exclude_cols = ['CustomerID', 'Churned', 'Country']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    X = df_final[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_final['Churned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\n4. Training model...")
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate
    print("\n5. Evaluating model...")
    evaluate_model(model, scaler, X_test, y_test)
    
    # Save artifacts
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, 'churn_model.joblib'))
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        print(f"\nModel artifacts saved to {output_dir}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return model, scaler, df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customer Churn Prediction')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input transaction data')
    parser.add_argument('--output', type=str, default='../models',
                        help='Directory to save model artifacts')
    
    args = parser.parse_args()
    main(args.input, args.output)
