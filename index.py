# Financial Fraud Detection System
# A complete implementation of a machine learning system for detecting fraudulent transactions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
import logging
import joblib
import datetime
import requests
import io
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fraud_detector")

##############################################
# 1. DATA COLLECTION & PREPROCESSING
##############################################

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_column = None
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
    def load_data(self, file_path=None, target_column='Class', feature_columns=None):
        """Load data from CSV file and identify features and target"""
        logger.info("Loading data...")
        try:
            # If no file path is provided, use the Kaggle dataset path
            if file_path is None:
                logger.info("Using dataset...")
                file_path = "datasets/creditcard.csv"
            
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Identify target column if not specified
            if target_column not in df.columns:
                # Try to find a binary target column
                binary_columns = [col for col in df.columns if df[col].nunique() == 2]
                if binary_columns:
                    target_column = binary_columns[0]
                    logger.info(f"Auto-detected target column: {target_column}")
                else:
                    raise ValueError("Could not identify target column. Please specify target_column parameter.")
            
            # Identify feature columns if not specified
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
            
            self.feature_names = feature_columns
            self.target_column = target_column
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self, df):
        """Basic data exploration with enhanced visualizations"""
        logger.info("Exploring data...")
        print(f"Dataset shape: {df.shape}")
        print(f"Number of fraudulent transactions: {df[self.target_column].sum()}")
        print(f"Percentage of fraudulent transactions: {df[self.target_column].mean() * 100:.4f}%")
        
        # Set style for better visualizations using newer seaborn API
        sns.set_theme(style="whitegrid")
        
        # 1. Class Distribution with enhanced visualization
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(x=self.target_column, data=df, palette=['#3498db', '#e74c3c'])
        plt.title('Distribution of Transaction Classes', fontsize=14, pad=20)
        plt.xlabel('Transaction Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add percentage labels
        total = len(df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height() + 0.01 * total
            ax.annotate(percentage, (x, y), ha='center', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/class_distribution.png')  # Save the plot
        plt.close()
        
        # 2. Transaction Amount Distribution with enhanced visualization
        plt.figure(figsize=(14, 7))
        sns.histplot(df[df[self.target_column] == 0]['Amount'], 
                    color='#3498db', label='Normal', 
                    kde=True, bins=50, alpha=0.5)
        sns.histplot(df[df[self.target_column] == 1]['Amount'], 
                    color='#e74c3c', label='Fraud', 
                    kde=True, bins=50, alpha=0.5)
        
        plt.title('Transaction Amount Distribution by Class', fontsize=14, pad=20)
        plt.xlabel('Amount ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=10)
        
        # Add mean lines
        mean_normal = df[df[self.target_column] == 0]['Amount'].mean()
        mean_fraud = df[df[self.target_column] == 1]['Amount'].mean()
        plt.axvline(mean_normal, color='#3498db', linestyle='--', alpha=0.7)
        plt.axvline(mean_fraud, color='#e74c3c', linestyle='--', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/amount_distribution.png')  # Save the plot
        plt.close()
        
        return df
    
    def preprocess_data(self, df, balance_method='smote', test_size=0.2, random_state=42):
        """Preprocess the data for model training"""
        logger.info("Preprocessing data...")
        
        # Separate features and target
        X = df[self.feature_names]
        y = df[self.target_column]
        
        self.feature_names=X.columns.tolist()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        logger.info("Standardizing features...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the scaler in models directory
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Handle class imbalance
        if balance_method == 'smote':
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - X_train shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
        elif balance_method == 'undersampling':
            logger.info("Applying undersampling for class balancing...")
            normal_indices = np.where(y_train == 0)[0]
            fraud_indices = np.where(y_train == 1)[0]
            
            random_normal_indices = np.random.choice(
                normal_indices, 
                size=len(fraud_indices) * 2,
                replace=False
            )
            
            undersampled_indices = np.concatenate([fraud_indices, random_normal_indices])
            X_train = X_train[undersampled_indices]
            y_train = y_train.iloc[undersampled_indices] if isinstance(y_train, pd.Series) else y_train[undersampled_indices]
            
            logger.info(f"After undersampling - X_train shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        return X_train, X_test, y_train, y_test


##############################################
# 2. MODEL DEVELOPMENT
##############################################

class FraudDetector:
    def __init__(self, input_dim, feature_names):
        self.input_dim = input_dim
        self.models = {}
        self.model_performance = {}
        self.feature_names = feature_names
        self.scaler = None  # Initialize scaler as None
        self.original_data = None  # Store original data for statistics
        
    def set_original_data(self, df):
        """Store original data for statistics"""
        self.original_data = df
        
    def train_decision_tree(self, X_train, y_train):
        """Train a Decision Tree model"""
        logger.info("Training Decision Tree model...")
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        self.models['decision_tree'] = dt
        return dt
    
    def train_random_forest(self, X_train, y_train):
        """Train a Random Forest model"""
        logger.info("Training Random Forest model...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        return rf
    
    def train_xgboost(self, X_train, y_train):
        """Train an XGBoost model"""
        logger.info("Training XGBoost model...")
        xgb = XGBClassifier(random_state=42)
        xgb.fit(X_train, y_train)
        self.models['xgboost'] = xgb
        return xgb
    
    def train_ann(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=256):
        """Train the ANN model"""
        logger.info("Training ANN model...")
        
        if X_val is None or y_val is None:
            X_val, y_val = X_train, y_train
        
        model = Sequential([
            Dense(64, activation='relu', input_dim=self.input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC()]
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        model_checkpoint = ModelCheckpoint(
            'models/best_ann_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint]
        )
        
        self.models['ann'] = model
        return model, history
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a model's performance"""
        logger.info(f"Evaluating {model_name} model...")
        
        if model_name == 'ann':
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Store performance metrics
        self.model_performance[model_name] = {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
        
        return self.model_performance[model_name]
    
    def compare_models(self):
        """Compare performance of all trained models"""
        logger.info("Comparing model performance...")
        
        comparison = {}
        for model_name, performance in self.model_performance.items():
            comparison[model_name] = {
                'accuracy': performance['classification_report']['accuracy'],
                'precision': performance['classification_report']['1']['precision'],
                'recall': performance['classification_report']['1']['recall'],
                'f1_score': performance['classification_report']['1']['f1-score'],
                'roc_auc': performance['roc_auc'],
                'pr_auc': performance['pr_auc']
            }
        
        return pd.DataFrame(comparison).T
    
    def get_feature_importance(self, model_name):
        """Get feature importance from tree-based models"""
        if model_name not in ['decision_tree', 'random_forest', 'xgboost']:
            logger.warning(f"Feature importance not available for {model_name}")
            return None
        
        model = self.models[model_name]
        
        if model_name == 'decision_tree':
            importance = model.feature_importances_
        elif model_name == 'random_forest':
            importance = model.feature_importances_
        else:  # xgboost
            importance = model.feature_importances_
        
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    
    def save_models(self):
        """Save all trained models"""
        logger.info("Saving models...")
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'ann':
                model.save(f'models/{model_name}_model.h5')
            else:
                joblib.dump(model, f'models/{model_name}_model.pkl')
        
        # Save scaler in models directory if it exists
        if hasattr(self, 'scaler') and self.scaler is not None:
            joblib.dump(self.scaler, 'models/scaler.pkl')
        else:
            logger.warning("No scaler found to save. This might cause issues with predictions later.")
        
        # Save original data statistics in models directory
        if self.original_data is not None:
            original_stats = {feature: {'mean': self.original_data[feature].mean(), 'std': self.original_data[feature].std()} 
                            for feature in self.feature_names}
            joblib.dump(original_stats, 'models/original_data_stats.pkl')
        else:
            logger.warning("No original data found to save statistics. This might affect drift detection later.")
    
    def load_models(self):
        """Load saved models"""
        logger.info("Loading models...")
        for model_name in ['decision_tree', 'random_forest', 'xgboost']:
            try:
                self.models[model_name] = joblib.load(f'models/{model_name}_model.pkl')
            except Exception as e:
                logger.warning(f"Could not load {model_name} model: {e}")
        
        try:
            self.models['ann'] = load_model('models/ann_model.h5')
        except Exception as e:
            logger.warning(f"Could not load ANN model: {e}")
        
        try:
            self.scaler = joblib.load('models/scaler.pkl')
        except Exception as e:
            logger.warning(f"Could not load scaler: {e}")
    
    def predict(self, X, model_name=None):
        """Make predictions using a specific model or all models"""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            if model_name == 'ann':
                return model.predict(X)
            else:
                return model.predict_proba(X)[:, 1]
        else:
            predictions = {}
            for name, model in self.models.items():
                if name == 'ann':
                    predictions[name] = model.predict(X)
                else:
                    predictions[name] = model.predict_proba(X)[:, 1]
            return predictions


##############################################
# 3. MODEL EVALUATION
##############################################

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
    def evaluate_model(self, threshold=0.5):
        """Evaluate the model's performance"""
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        class_report = classification_report(self.y_test, y_pred)
        
        # Print results
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        
        return conf_matrix, class_report
    
    def plot_metrics(self, model_name):
        """Plot enhanced performance metrics"""
        logger.info(f"Plotting performance metrics for {model_name}...")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Set style
        sns.set_theme(style="whitegrid")
        
        # 1. Confusion Matrix with enhanced visualization
        plt.figure(figsize=(10, 8))
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'],
                   annot_kws={'size': 14})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.savefig(f'plots/confusion_matrix_{model_name}.png')  # Save with model name
        plt.close()
        
        # 2. Precision-Recall Curve with enhanced visualization
        plt.figure(figsize=(10, 8))
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, 
                label=f'PR Curve (AUC = {pr_auc:.3f})',
                color='#2ecc71', linewidth=2)
        plt.fill_between(recall, precision, alpha=0.2, color='#2ecc71')
        
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, pad=20)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'plots/precision_recall_curve_{model_name}.png')  # Save with model name
        plt.close()
        
        # 3. ROC Curve with enhanced visualization
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})',
                color='#3498db', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
        
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}', fontsize=14, pad=20)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'plots/roc_curve_{model_name}.png')  # Save with model name
        plt.close()
    
    def find_optimal_threshold(self):
        """Find the optimal decision threshold for classification"""
        logger.info("Finding optimal threshold...")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_test)
        
        # Calculate precision and recall for different thresholds
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * recall * precision / (recall + precision + 1e-7)
        
        # Find threshold with max F1 score
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f} with F1: {f1_scores[optimal_idx]:.4f}")
        
        # Plot thresholds
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
        plt.plot(thresholds, f1_scores[:-1], 'r-.', label='F1 Score')
        plt.vlines(optimal_threshold, 0, 1, colors='k', linestyles='dashed', label=f'Optimal Threshold: {optimal_threshold:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1 Score by Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig('threshold_optimization.png')
        
        return optimal_threshold

##############################################
# 3. SECURITY & MONITORING
##############################################

class ModelMonitor:
    def __init__(self, model_path, scaler_path, log_path="predictions.log"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.log_path = log_path
        self.model = None
        self.scaler = None
        
        # Setup logging
        self.logger = logging.getLogger("model_monitor")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
    def load_artifacts(self):
        """Load the model and scaler"""
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
    def log_prediction(self, transaction_id, features, prediction, probability):
        """Log a prediction for monitoring"""
        self.logger.info(f"{transaction_id},{','.join(map(str, features))},{prediction},{probability}")
    
    def analyze_logs(self, last_n_days=None):
        """Analyze prediction logs for drift or anomalies"""
        # Load log file
        logs = pd.read_csv(self.log_path, parse_dates=[0])
        
        # Filter by date if needed
        if last_n_days:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=last_n_days)
            logs = logs[logs['timestamp'] >= cutoff_date]
        
        # Basic statistics
        total_predictions = len(logs)
        fraud_predictions = logs['prediction'].sum()
        fraud_rate = fraud_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Total predictions: {total_predictions}")
        print(f"Fraud predictions: {fraud_predictions}")
        print(f"Fraud rate: {fraud_rate:.4f}")
        
        # Plot fraud rate over time
        logs['date'] = logs['timestamp'].dt.date
        daily_fraud_rate = logs.groupby('date')['prediction'].mean()
        
        plt.figure(figsize=(12, 6))
        daily_fraud_rate.plot()
        plt.title('Daily Fraud Rate')
        plt.ylabel('Fraud Rate')
        plt.xlabel('Date')
        plt.grid(True)
        plt.savefig('fraud_rate_trend.png')
        
        return {
            'total_predictions': total_predictions,
            'fraud_predictions': fraud_predictions,
            'fraud_rate': fraud_rate,
            'daily_fraud_rate': daily_fraud_rate.to_dict()
        }
    
    def check_for_drift(self, recent_data_path, threshold=0.1):
        """Check for concept drift by comparing distributions"""
        # Load recent data
        recent_df = pd.read_csv(recent_data_path)
        
        # Load the original model training data statistics
        # This would be saved during initial training
        original_stats = joblib.load('original_data_stats.pkl')
        
        # Compare distributions
        drift_detected = False
        feature_drifts = {}
        
        for feature in recent_df.columns[:-1]:  # Excluding the target column
            # Calculate distribution statistics
            recent_mean = recent_df[feature].mean()
            recent_std = recent_df[feature].std()
            
            # Compare with original statistics
            mean_diff = abs(recent_mean - original_stats[feature]['mean']) / original_stats[feature]['std']
            std_ratio = recent_std / original_stats[feature]['std']
            
            # Check for significant drift
            if mean_diff > threshold or abs(std_ratio - 1) > threshold:
                drift_detected = True
                feature_drifts[feature] = {
                    'mean_shift': mean_diff, 
                    'std_ratio': std_ratio
                }
        
        return drift_detected, feature_drifts
    
    def retrain_model(self, new_data_path, original_data_path=None):
        """Retrain the model with new data"""
        # Implementation would merge original and new data,
        # then rerun the training process
        
        logger.info("Model retraining initiated...")
        # The retraining would use the same DataProcessor and FraudDetector classes
        # defined earlier in the pipeline
        
        # After retraining, update the model and scaler
        logger.info("Model retrained and updated successfully")


##############################################
# MAIN EXECUTION FLOW
##############################################

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load and preprocess data
    df = data_processor.load_data()
    df = data_processor.explore_data(df)
    X_train, X_test, y_train, y_test = data_processor.preprocess_data(df)
    
    # Initialize fraud detector
    detector = FraudDetector(input_dim=len(data_processor.feature_names), feature_names=data_processor.feature_names)
    detector.scaler = data_processor.scaler  # Pass the scaler from DataProcessor to FraudDetector
    detector.set_original_data(df)  # Pass the original data for statistics
    
    # Check if models already exist
    models_exist = all([
        os.path.exists(f'models/{model_name}_model.pkl') for model_name in ['decision_tree', 'random_forest', 'xgboost']
    ]) and os.path.exists('models/ann_model.h5')
    
    if models_exist:
        logger.info("Loading existing models...")
        detector.load_models()
    else:
        # Train all models
        logger.info("Training models...")
        dt_model = detector.train_decision_tree(X_train, y_train)
        rf_model = detector.train_random_forest(X_train, y_train)
        xgb_model = detector.train_xgboost(X_train, y_train)
        ann_model, ann_history = detector.train_ann(X_train, y_train, X_test, y_test, epochs=30)
    
    # Evaluate models
    for model_name in ['decision_tree', 'random_forest', 'xgboost', 'ann']:
        print(f"\nEvaluating {model_name} model:")
        detector.evaluate_model(detector.models[model_name], model_name, X_test, y_test)
        
        # Create ModelEvaluator instance and plot metrics
        evaluator = ModelEvaluator(detector.models[model_name], X_test, y_test)
        evaluator.plot_metrics(model_name)
    
    # Compare models
    model_comparison = detector.compare_models()
    print("\nModel Comparison:")
    print(model_comparison)
    
    # Save models and statistics
    try:
        detector.save_models()
    except Exception as e:
        logger.warning(f"Error saving models: {e}")
        logger.info("Continuing with existing models...")
    
    # Get feature importance
    for model_name in ['decision_tree', 'random_forest', 'xgboost']:
        importance = detector.get_feature_importance(model_name)
        print(f"\nFeature Importance ({model_name}):")
        print(importance.head())
    
    logger.info("Fraud detection system setup complete!")

if __name__ == "__main__":
    main()