# Credit Card Fraud Detection Dataset Blueprint

## Dataset Structure

### Base Features
1. **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset
2. **Amount**: Transaction amount
3. **Class**: Target variable (0: legitimate, 1: fraudulent)

### Derived Features (V1-V28)
These are features obtained through PCA transformation of the original features. They represent:
- Transaction patterns
- Merchant information
- Cardholder behavior
- Geographic location
- Transaction timing patterns

## Synthetic Dataset Generation Guidelines

### Dataset 1: E-commerce Transactions
- Focus on online shopping patterns
- Higher frequency of small transactions
- More transactions during peak hours
- Higher fraud rate in specific categories
- Seasonal patterns (holidays, weekends)

### Dataset 2: International Transactions
- Mix of domestic and international transactions
- Currency conversion patterns
- Time zone differences
- Cross-border transaction patterns
- Higher fraud rate in international transactions

## Feature Engineering Guidelines

### Time-based Features
- Hour of day
- Day of week
- Weekend indicator
- Holiday indicator

### Amount-based Features
- Transaction amount categories
- Average transaction amount
- Transaction amount deviation
- Cumulative amount

### Behavioral Features
- Transaction frequency
- Merchant category patterns
- Location patterns
- Time between transactions

## Data Quality Requirements

### Missing Values
- No missing values allowed
- Use appropriate imputation if needed

### Outliers
- Keep legitimate outliers
- Flag suspicious outliers for review

### Data Types
- Time: float64
- Amount: float64
- V1-V28: float64
- Class: int64

## Data Distribution Guidelines

### Class Distribution
- Legitimate transactions: 95-99%
- Fraudulent transactions: 1-5%

### Amount Distribution
- Mean: $100-200
- Standard deviation: $50-100
- Skewed right distribution

### Time Distribution
- Peak hours: 9 AM - 9 PM
- Off-peak hours: 9 PM - 9 AM
- Weekend patterns different from weekdays 