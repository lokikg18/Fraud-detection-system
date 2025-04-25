import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random

def generate_ecommerce_dataset(num_records=100000):
    """Generate synthetic e-commerce transaction dataset"""
    np.random.seed(42)
    
    # Generate base features
    time = np.random.uniform(0, 172800, num_records)  # 48 hours of transactions
    amount = np.random.lognormal(mean=4.5, sigma=0.8, size=num_records)
    
    # Generate PCA features (V1-V28)
    pca_features = np.random.normal(0, 1, (num_records, 28))
    
    # Create DataFrame
    df = pd.DataFrame(pca_features, columns=[f'V{i}' for i in range(1, 29)])
    df['Time'] = time
    df['Amount'] = amount
    
    # Generate fraud labels (1% fraud rate)
    fraud_rate = 0.01
    df['Class'] = np.random.binomial(1, fraud_rate, num_records)
    
    # Add e-commerce specific patterns
    # Higher fraud rate for high-value transactions
    high_value_mask = df['Amount'] > 1000
    df.loc[high_value_mask, 'Class'] = np.random.binomial(1, 0.05, high_value_mask.sum())
    
    # Higher fraud rate during off-hours
    off_hours_mask = (df['Time'] % 86400 < 21600) | (df['Time'] % 86400 > 75600)  # 6 PM - 6 AM
    df.loc[off_hours_mask, 'Class'] = np.random.binomial(1, 0.02, off_hours_mask.sum())
    
    return df

def generate_international_dataset(num_records=100000):
    """Generate synthetic international transaction dataset"""
    np.random.seed(43)
    
    # Generate base features
    time = np.random.uniform(0, 172800, num_records)  # 48 hours of transactions
    amount = np.random.lognormal(mean=5.0, sigma=1.0, size=num_records)
    
    # Generate PCA features (V1-V28)
    pca_features = np.random.normal(0, 1, (num_records, 28))
    
    # Create DataFrame
    df = pd.DataFrame(pca_features, columns=[f'V{i}' for i in range(1, 29)])
    df['Time'] = time
    df['Amount'] = amount
    
    # Generate fraud labels (1.5% fraud rate)
    fraud_rate = 0.015
    df['Class'] = np.random.binomial(1, fraud_rate, num_records)
    
    # Add international transaction patterns
    # Higher fraud rate for international transactions (simulated by V1)
    international_mask = df['V1'] > 2
    df.loc[international_mask, 'Class'] = np.random.binomial(1, 0.04, international_mask.sum())
    
    # Higher fraud rate for currency conversion (simulated by V2)
    currency_mask = abs(df['V2']) > 2
    df.loc[currency_mask, 'Class'] = np.random.binomial(1, 0.03, currency_mask.sum())
    
    return df

def save_datasets():
    """Generate and save both datasets"""
    # Generate datasets
    ecommerce_df = generate_ecommerce_dataset()
    international_df = generate_international_dataset()
    
    # Save to CSV
    ecommerce_df.to_csv('ecommerce_transactions.csv', index=False)
    international_df.to_csv('international_transactions.csv', index=False)
    
    # Print dataset statistics
    print("\nE-commerce Dataset Statistics:")
    print(f"Total records: {len(ecommerce_df)}")
    print(f"Fraud rate: {ecommerce_df['Class'].mean():.2%}")
    print(f"Average amount: ${ecommerce_df['Amount'].mean():.2f}")
    
    print("\nInternational Dataset Statistics:")
    print(f"Total records: {len(international_df)}")
    print(f"Fraud rate: {international_df['Class'].mean():.2%}")
    print(f"Average amount: ${international_df['Amount'].mean():.2f}")

if __name__ == "__main__":
    save_datasets() 