from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import joblib
import tensorflow as tf
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import tempfile
import os
from index import DataProcessor, FraudDetector  # Updated import

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fastapi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Fraud Detection API",
    description="API for fraud detection with credit card transactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
data_processor = None
fraud_detector = None
models = {}
scaler = None
model_metrics = {}

class Transaction(BaseModel):
    features: List[float]
    model: str = "auto"

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool
    model_used: str
    timestamp: str

def load_models():
    """Load all models and scaler with detailed logging"""
    global models, scaler
    try:
        logger.info("Starting model loading process...")
        
        # Check if models directory exists
        if not os.path.exists('models'):
            logger.error("Models directory not found")
            return False
            
        # Load models
        model_files = {
            'decision_tree': 'models/decision_tree_model.pkl',
            'random_forest': 'models/random_forest_model.pkl',
            'xgboost': 'models/xgboost_model.pkl',
            'ann': 'models/ann_model.h5'
        }
        
        for model_name, model_path in model_files.items():
            try:
                if os.path.exists(model_path):
                    logger.info(f"Loading {model_name} model from {model_path}")
                    if model_name == 'ann':
                        models[model_name] = tf.keras.models.load_model(model_path)
                    else:
                        models[model_name] = joblib.load(model_path)
                    logger.info(f"Successfully loaded {model_name} model")
                else:
                    logger.error(f"Model file not found: {model_path}")
                    return False
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")
                return False
        
        # Load scaler from models directory
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
            logger.info("Successfully loaded scaler")
        else:
            logger.error(f"Scaler file not found: {scaler_path}")
            return False
            
        logger.info("All models and scaler loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error in load_models: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    logger.info("Starting API initialization...")
    if not load_models():
        logger.error("Failed to load models during startup")
    else:
        logger.info("API started successfully with all models loaded")

@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    balance_method: str = Form("smote"),
    test_size: float = Form(0.2)
):
    """Upload and process a new dataset"""
    try:
        if not data_processor:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and preprocess data
            df = data_processor.load_data(tmp_file_path)
            df = data_processor.explore_data(df)
            X_train, X_test, y_train, y_test = data_processor.preprocess_data(
                df, 
                balance_method=balance_method,
                test_size=test_size
            )
            
            # Initialize fraud detector
            global fraud_detector
            fraud_detector = FraudDetector(input_dim=X_train.shape[1], feature_names=data_processor.feature_names)
            
            # Train models
            fraud_detector.train_decision_tree(X_train, y_train)
            fraud_detector.train_random_forest(X_train, y_train)
            fraud_detector.train_xgboost(X_train, y_train)
            fraud_detector.train_ann(X_train, y_train, X_test, y_test)
            
            # Save models
            fraud_detector.save_models()
            
            return {
                "message": "Dataset uploaded and processed successfully",
                "dataset_info": {
                    "total_records": len(df),
                    "features": len(data_processor.feature_names),
                    "fraud_count": int(df[data_processor.target_column].sum()),
                    "fraud_percentage": float(df[data_processor.target_column].mean() * 100),
                    "balance_method": balance_method,
                    "test_size": test_size
                },
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(transaction: Transaction):
    """Make a prediction using the selected model"""
    try:
        if not models:
            raise HTTPException(status_code=404, detail="No models available")
        
        if not scaler:
            raise HTTPException(status_code=500, detail="Scaler not loaded")
        
        # Scale the transaction data
        try:
            transaction_data = np.array(transaction.features).reshape(1, -1)
            scaled_data = scaler.transform(transaction_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid transaction data format: {str(e)}")
        
        # Select model
        if transaction.model == "auto":
            # Get predictions from all models
            predictions = {}
            for model_name, model in models.items():
                try:
                    if model_name == 'ann':
                        pred = model.predict(scaled_data)[0][0]
                    else:
                        pred = model.predict_proba(scaled_data)[0][1]
                    predictions[model_name] = pred
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    continue
            
            # Select best model based on confidence
            best_model = max(predictions.items(), key=lambda x: x[1])[0] if predictions else "random_forest"
        else:
            if transaction.model not in models:
                raise HTTPException(status_code=400, detail=f"Model {transaction.model} not found")
            best_model = transaction.model
        
        # Make prediction
        try:
            model = models[best_model]
            if best_model == 'ann':
                probability = model.predict(scaled_data)[0][0]
            else:
                probability = model.predict_proba(scaled_data)[0][1]
            
            is_fraud = probability > 0.5
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
        
        return PredictionResponse(
            prediction=int(is_fraud),
            probability=float(probability),
            is_fraud=bool(is_fraud),
            model_used=best_model,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health status"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": bool(models),
            "scaler_loaded": bool(scaler),
            "data_processor_initialized": data_processor is not None,
            "fraud_detector_initialized": fraud_detector is not None
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def model_status():
    """Check model status with detailed response"""
    try:
        status = "ready" if models and scaler else "not_ready"
        missing_components = []
        
        if not models:
            missing_components.append("models")
        if not scaler:
            missing_components.append("scaler")
            
        response = {
            "model_status": status,
            "loaded_models": list(models.keys()) if models else [],
            "scaler_loaded": bool(scaler),
            "missing_components": missing_components
        }
        
        logger.info(f"Model status check: {response}")
        return response
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return {"model_status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
