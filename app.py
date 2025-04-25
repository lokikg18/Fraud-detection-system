from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from index import DataProcessor, FraudDetector

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("flask.log"),
                            logging.StreamHandler()])
logger = logging.getLogger("fraud_api")

# Global variables
data_processor = DataProcessor()
fraud_detector = None
models = {}
scaler = None

def load_models():
    """Load all models and scaler"""
    global models, scaler
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load models
        model_files = {
            'decision_tree': 'models/decision_tree_model.pkl',
            'random_forest': 'models/random_forest_model.pkl',
            'xgboost': 'models/xgboost_model.pkl',
            'ann': 'models/ann_model.h5'
        }
        
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                if model_name == 'ann':
                    models[model_name] = load_model(model_path)
                else:
                    models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model")
        
        # Load scaler from models directory
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Check API health status"""
    try:
        return jsonify({
            "status": "ok",
            "models_loaded": bool(models),
            "scaler_loaded": bool(scaler),
            "data_processor_initialized": data_processor is not None,
            "fraud_detector_initialized": fraud_detector is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction using the selected model"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        features = data.get('features', [])
        model_name = data.get('model', 'auto')  # Default to auto selection
        
        if not features:
            return jsonify({"error": "No features provided"}), 400
        
        if not models:
            return jsonify({"error": "Models not loaded"}), 500
        
        if not scaler:
            return jsonify({"error": "Scaler not loaded"}), 500
        
        # Convert to numpy array
        try:
            features_array = np.array(features).reshape(1, -1)
        except Exception as e:
            return jsonify({"error": f"Invalid features format: {str(e)}"}), 400
        
        # Scale the data
        try:
            scaled_features = scaler.transform(features_array)
        except Exception as e:
            return jsonify({"error": f"Error scaling features: {str(e)}"}), 500
        
        # Select model if auto mode
        selected_model = model_name
        if selected_model == "auto":
            # Get predictions from all models
            predictions = {}
            for name, model in models.items():
                try:
                    if name == 'ann':
                        pred = model.predict(scaled_features)[0][0]
                    else:
                        pred = model.predict_proba(scaled_features)[0][1]
                    predictions[name] = pred
                except Exception as e:
                    logger.error(f"Error getting prediction from {name}: {e}")
                    continue
            
            # Select best model based on confidence
            selected_model = max(predictions.items(), key=lambda x: x[1])[0] if predictions else "random_forest"
        elif selected_model not in models:
            return jsonify({"error": f"Model {selected_model} not found"}), 400
        
        # Make prediction
        try:
            model = models[selected_model]
            if selected_model == 'ann':
                probability = model.predict(scaled_features)[0][0]
            else:
                probability = model.predict_proba(scaled_features)[0][1]
            
            is_fraud = probability > 0.5
        except Exception as e:
            return jsonify({"error": f"Error making prediction: {str(e)}"}), 500
        
        # Log the prediction
        logger.info(f"Prediction made using {selected_model}: {is_fraud} with probability {probability:.4f}")
        
        return jsonify({
            "prediction": int(is_fraud),
            "probability": float(probability),
            "is_fraud": bool(is_fraud),
            "model_used": selected_model,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    try:
        if not models:
            return jsonify({"error": "Models not loaded"}), 500
        
        return jsonify({
            "available_models": list(models.keys()),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

# Load models on startup
load_models()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)