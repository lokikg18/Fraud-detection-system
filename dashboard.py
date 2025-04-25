import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import logging
import requests
import tempfile
import os
from index import DataProcessor, FraudDetector
from plotly.subplots import make_subplots
import time
import random
import concurrent.futures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fraud_dashboard")

# Initialize session state with proper error handling
def initialize_session_state():
    try:
        if 'data_processor' not in st.session_state:
            st.session_state.data_processor = DataProcessor()
        if 'fraud_detector' not in st.session_state:
            st.session_state.fraud_detector = None
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'api_status' not in st.session_state:
            st.session_state.api_status = {
                'flask': False,
                'fastapi': False
            }
        if 'recent_predictions' not in st.session_state:
            st.session_state.recent_predictions = []
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = True  # Default to dark mode for modern look
        if 'cache' not in st.session_state:
            st.session_state.cache = {}
        if 'last_api_check' not in st.session_state:
            st.session_state.last_api_check = 0
        
        # Check model status on initialization
        check_model_status()
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        st.error("Error initializing dashboard. Please refresh the page.")

def load_models():
    """Load all trained models with improved error handling"""
    try:
        models = {}
        model_files = {
            'decision_tree': 'models/decision_tree_model.pkl',
            'random_forest': 'models/random_forest_model.pkl',
            'xgboost': 'models/xgboost_model.pkl',
            'ann': 'models/ann_model.h5'
        }
        
        # Check if models directory exists
        if not os.path.exists('models'):
            logger.warning("Models directory not found")
            return None, None
        
        # Load models
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                if model_name == 'ann':
                    models[model_name] = load_model(model_path)
                else:
                    models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load scaler
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            logger.info("Loaded scaler")
            return models, scaler
        else:
            logger.warning("Scaler file not found")
            return None, None
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

def check_model_status():
    """Check if models are properly loaded with improved error handling"""
    try:
        # First check FastAPI model status
        try:
            response = requests.get("http://localhost:8000/model/status", timeout=5)
            if response.status_code == 200:
                status = response.json().get("model_status")
                if status == "ready":
                    st.session_state.models_trained = True
                    logger.info("Models are ready (FastAPI check)")
                    return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking model status from FastAPI: {e}")

        # Fallback to local model check
        models, scaler = load_models()
        if models and scaler:
            st.session_state.models_trained = True
            st.session_state.fraud_detector = FraudDetector(
                input_dim=scaler.n_features_in_,
                feature_names=[f"V{i+1}" for i in range(scaler.n_features_in_)]
            )
            st.session_state.fraud_detector.models = models
            logger.info("Models are ready (local check)")
            return True
        else:
            st.session_state.models_trained = False
            logger.warning("Models are not ready")
            return False
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        st.session_state.models_trained = False
        return False

def make_prediction(features, model_name="auto"):
    """Make prediction using the selected model with improved error handling and logging"""
    try:
        # Log prediction attempt
        logger.info(f"Attempting prediction with model: {model_name}")
        logger.info(f"Features: {features}")

        # Try FastAPI first
        try:
            url = "http://localhost:8000/predict"
            payload = {
                "features": features,
                "model": model_name
            }
            logger.info(f"Sending request to FastAPI: {payload}")
            
            response = requests.post(url, json=payload, timeout=10)
            logger.info(f"FastAPI Response Status: {response.status_code}")
            logger.info(f"FastAPI Response Content: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"FastAPI error: {response.text}"
                logger.error(error_msg)
                st.error(f"FastAPI Error: {error_msg}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"FastAPI prediction failed, trying Flask: {e}")
            st.warning(f"FastAPI unavailable: {str(e)}")

        # Fallback to Flask
        try:
            url = "http://localhost:5001/predict"
            payload = {
                "features": features,
                "model": model_name
            }
            logger.info(f"Sending request to Flask API: {payload}")
            
            response = requests.post(url, json=payload, timeout=10)
            logger.info(f"Flask Response Status: {response.status_code}")
            logger.info(f"Flask Response Content: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Flask API error: {response.text}"
                logger.error(error_msg)
                st.error(f"Flask API Error: {error_msg}")
        except requests.exceptions.RequestException as e:
            error_msg = f"Flask prediction failed: {e}"
            logger.error(error_msg)
            st.error(error_msg)

        logger.error("Both prediction APIs failed")
        st.error("Both APIs failed to respond. Please check if they are running.")
        return None
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return None

def add_alert(alert_type, severity, message):
    """Add a new alert to the system with error handling"""
    try:
        alert = {
            'time': datetime.now(),
            'type': alert_type,
            'severity': severity,
            'message': message
        }
        st.session_state.alerts.append(alert)
        if len(st.session_state.alerts) > 10:  # Keep only last 10 alerts
            st.session_state.alerts = st.session_state.alerts[-10:]
    except Exception as e:
        logger.error(f"Error adding alert: {e}")

def add_prediction(features, result):
    """Add a new prediction to the history with error handling"""
    try:
        prediction = {
            'time': datetime.now(),
            'features': features,
            'result': result
        }
        st.session_state.recent_predictions.append(prediction)
        if len(st.session_state.recent_predictions) > 10:  # Keep only last 10 predictions
            st.session_state.recent_predictions = st.session_state.recent_predictions[-10:]
    except Exception as e:
        logger.error(f"Error adding prediction: {e}")

def check_api_status():
    """Check status of both APIs with improved error handling and caching"""
    cache_key = "api_status"
    current_time = time.time()
    
    # Only check API every 30 seconds to improve performance
    if current_time - st.session_state.last_api_check < 30:
        if cache_key in st.session_state.cache:
            return st.session_state.cache[cache_key]["status"]
        else:
            # First time check
            pass
    
    st.session_state.last_api_check = current_time
    
    try:
        # Use concurrent requests for faster checking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Check Flask API
            flask_future = executor.submit(
                lambda: requests.get("http://localhost:5001/health", timeout=2)
            )
            # Check FastAPI
            fastapi_future = executor.submit(
                lambda: requests.get("http://localhost:8000/health", timeout=2)
            )
            
            # Get results
            try:
                flask_response = flask_future.result()
                flask_status = flask_response.status_code == 200
            except Exception as e:
                logger.error(f"Error checking Flask API status: {e}")
                flask_status = False
                
            try:
                fastapi_response = fastapi_future.result()
                fastapi_status = fastapi_response.status_code == 200
            except Exception as e:
                logger.error(f"Error checking FastAPI status: {e}")
                fastapi_status = False
                
        status = {
            'flask': flask_status,
            'fastapi': fastapi_status
        }
        st.session_state.api_status = status
        # Cache the status
        st.session_state.cache[cache_key] = {"status": status, "timestamp": current_time}
        return status
    except Exception as e:
        logger.error(f"Unexpected error checking API status: {e}")
        status = {
            'flask': False,
            'fastapi': False
        }
        st.session_state.api_status = status
        # Cache the status
        st.session_state.cache[cache_key] = {"status": status, "timestamp": current_time}
        return status

# Initialize session state
initialize_session_state()

# Page config with modern theme
st.set_page_config(
    page_title="Advanced Fraud Detection Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern color schemes
def get_colors():
    if st.session_state.dark_mode:
        return {
            # Primary Colors
            'bg_primary': '#051937',  # Deep Navy
            'bg_secondary': '#162B4D',  # Dark Blue
            'bg_tertiary': '#2E4369',  # Medium Blue
            
            # Text Colors
            'text_primary': '#FFFFFF',  # White
            'text_secondary': '#B0B9C2',  # Mid Gray
            'text_tertiary': '#E0E4E8',  # Light Gray
            
            # Accent Colors
            'accent_1': '#00B4D8',  # Bright Teal
            'accent_2': '#48CAE4',  # Light Teal
            'accent_3': '#FFB703',  # Gold
            'accent_4': '#FD9E02',  # Amber
            
            # Status Colors
            'success': '#48CAE4',  # Teal
            'warning': '#FFB703',  # Gold
            'error': '#FF5A5F',  # Red
            
            # Chart Colors
            'chart_normal_1': '#00B4D8',  # Bright Teal
            'chart_normal_2': '#48CAE4',  # Light Teal
            'chart_fraud_1': '#FFB703',  # Gold
            'chart_fraud_2': '#FF5A5F',  # Red
            
            # Gradient Colors
            'gradient_start': '#051937',  # Deep Navy
            'gradient_mid': '#0A2E5C',  # Medium Navy
            'gradient_end': '#162B4D',  # Dark Blue
            
            # Interactive Elements
            'button_bg': '#00B4D8',  # Bright Teal
            'button_hover': '#48CAE4',  # Light Teal
            'button_text': '#FFFFFF',  # White
            
            # Card Backgrounds
            'card_bg': 'rgba(5, 25, 55, 0.8)',  # Semi-transparent Navy
            'card_border': '#2E4369',  # Medium Blue
            
            # Alert Colors
            'alert_high': '#FF5A5F',  # Red
            'alert_medium': '#FFB703',  # Gold
            'alert_low': '#48CAE4',  # Teal
        }
    else:
        return {
            # Primary Colors
            'bg_primary': '#F8FAFC',  # Off White
            'bg_secondary': '#FFFFFF',  # White
            'bg_tertiary': '#F1F5F9',  # Light Gray
            
            # Text Colors
            'text_primary': '#1E1E1E',  # Dark Charcoal
            'text_secondary': '#4B5563',  # Dark Gray
            'text_tertiary': '#6B7280',  # Medium Gray
            
            # Accent Colors
            'accent_1': '#00B4D8',  # Bright Teal
            'accent_2': '#48CAE4',  # Light Teal
            'accent_3': '#FFB703',  # Gold
            'accent_4': '#FD9E02',  # Amber
            
            # Status Colors
            'success': '#48CAE4',  # Teal
            'warning': '#FFB703',  # Gold
            'error': '#FF5A5F',  # Red
            
            # Chart Colors
            'chart_normal_1': '#00B4D8',  # Bright Teal
            'chart_normal_2': '#48CAE4',  # Light Teal
            'chart_fraud_1': '#FFB703',  # Gold
            'chart_fraud_2': '#FF5A5F',  # Red
            
            # Gradient Colors
            'gradient_start': '#F8FAFC',  # Off White
            'gradient_mid': '#F1F5F9',  # Light Gray
            'gradient_end': '#E2E8F0',  # Medium Gray
            
            # Interactive Elements
            'button_bg': '#00B4D8',  # Bright Teal
            'button_hover': '#48CAE4',  # Light Teal
            'button_text': '#FFFFFF',  # White
            
            # Card Backgrounds
            'card_bg': 'rgba(255, 255, 255, 0.9)',  # Semi-transparent White
            'card_border': '#E2E8F0',  # Medium Gray
            
            # Alert Colors
            'alert_high': '#FF5A5F',  # Red
            'alert_medium': '#FFB703',  # Gold
            'alert_low': '#48CAE4',  # Teal
        }

# Get current color scheme
colors = get_colors()

# Add advanced CSS animations and transitions
st.markdown(f"""
    <style>
    /* Advanced Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0% {{
            transform: scale(1);
            box-shadow: 0 0 0 0 {colors['accent_1']}40;
        }}
        70% {{
            transform: scale(1.05);
            box-shadow: 0 0 0 10px {colors['accent_1']}00;
        }}
        100% {{
            transform: scale(1);
            box-shadow: 0 0 0 0 {colors['accent_1']}00;
        }}
    }}
    
    @keyframes gradientShift {{
        0% {{
            background-position: 0% 50%;
        }}
        50% {{
            background-position: 100% 50%;
        }}
        100% {{
            background-position: 0% 50%;
        }}
    }}
    
    @keyframes borderGlow {{
        0% {{
            border-color: {colors['accent_1']}40;
        }}
        50% {{
            border-color: {colors['accent_2']}40;
        }}
        100% {{
            border-color: {colors['accent_1']}40;
        }}
    }}
    
    /* Enhanced Card Styles */
    .card {{
        background: {colors['bg_secondary']};
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid {colors['accent_1']}20;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        border-color: {colors['accent_1']}40;
    }}
    
    /* Enhanced Metrics Card */
    .metrics-card {{
        background: linear-gradient(135deg, {colors['bg_secondary']}, {colors['bg_primary']});
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid {colors['accent_1']}20;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .metrics-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, {colors['accent_1']}10, {colors['accent_2']}10);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .metrics-card:hover::before {{
        opacity: 1;
    }}
    
    /* Enhanced Button Styles */
    .stButton>button {{
        background: linear-gradient(45deg, {colors['accent_1']}, {colors['accent_2']});
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px {colors['accent_1']}40;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px {colors['accent_1']}60;
        background: linear-gradient(45deg, {colors['accent_2']}, {colors['accent_1']});
    }}
    
    /* Enhanced Alert Box */
    .alert-box {{
        background: {colors['bg_secondary']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid {colors['accent_1']}20;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .alert-box:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }}
    
    /* Enhanced Sidebar */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, {colors['accent_1']}, {colors['accent_2']}) !important;
        color: {colors['text_primary']} !important;
    }}
    
    .sidebar .sidebar-content .stRadio > div > div > label {{
        color: {colors['text_primary']} !important;
    }}
    
    .sidebar .sidebar-content .stSelectbox > div > div > div {{
        color: {colors['text_primary']} !important;
        background-color: {colors['bg_secondary']} !important;
    }}
    
    .sidebar .sidebar-content .stButton > button {{
        background: {colors['bg_secondary']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['text_primary']} !important;
    }}
    
    .sidebar .sidebar-content .stButton > button:hover {{
        background: {colors['accent_3']} !important;
        color: {colors['text_primary']} !important;
    }}
    
    .sidebar .sidebar-content .stMarkdown {{
        color: {colors['text_primary']} !important;
    }}
    
    .sidebar .sidebar-content .stMetric {{
        color: {colors['text_primary']} !important;
    }}
    
    .sidebar .sidebar-content .stProgress > div > div > div {{
        background-color: {colors['text_primary']} !important;
    }}
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        border-radius: 12px;
        padding: 5px;
        background: {colors['bg_secondary']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 10px 20px;
        background: transparent;
        border: none;
        color: {colors['text_secondary']};
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {colors['accent_1']}20;
        color: {colors['accent_1']};
        font-weight: 600;
    }}
    
    /* Enhanced File Uploader */
    [data-testid="stFileUploader"] {{
        border-radius: 12px;
        border: 2px dashed {colors['accent_1']}40;
        padding: 15px;
        transition: all 0.3s ease;
        background: {colors['bg_secondary']};
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {colors['accent_1']};
        background: {colors['accent_1']}10;
        animation: borderGlow 2s infinite;
    }}
    
    /* Enhanced Dataframe */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        animation: fadeInUp 0.6s ease-out;
    }}
    
    /* Enhanced Header */
    .gradient-header {{
        background: linear-gradient(135deg, {colors['accent_1']}, {colors['accent_2']});
        color: white;
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px {colors['accent_1']}40;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }}
    
    .gradient-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 200%;
        height: 100%;
        background: linear-gradient(45deg, {colors['accent_1']}00, {colors['accent_2']}80, {colors['accent_3']}00, {colors['accent_1']}80);
        z-index: -1;
        background-size: 200% 100%;
        animation: gradientShift 8s ease infinite;
    }}
    
    /* Enhanced Slider */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, {colors['accent_1']}, {colors['accent_2']});
    }}
    
    /* Enhanced Progress Bar */
    .progress-container {{
        width: 100%;
        background: {colors['bg_primary']};
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin-top: 10px;
    }}
    
    .progress-bar {{
        height: 100%;
        background: linear-gradient(90deg, {colors['accent_1']}, {colors['accent_2']});
        transition: width 1s ease;
        border-radius: 10px;
    }}
    
    /* Enhanced Status Indicators */
    .status-indicator {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }}
    
    .status-active {{
        background: {colors['success']};
        box-shadow: 0 0 10px {colors['success']}80;
    }}
    
    .status-inactive {{
        background: {colors['error']};
        box-shadow: 0 0 10px {colors['error']}80;
    }}
    
    .status-warning {{
        background: {colors['warning']};
        box-shadow: 0 0 10px {colors['warning']}80;
    }}
    
    /* Glass Morphism Effect */
    .glass-effect {{
        background: {colors['bg_secondary']}80 !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid {colors['bg_secondary']} !important;
    }}
    
    /* Responsive Improvements */
    @media (max-width: 768px) {{
        .gradient-header {{
            padding: 15px;
        }}
        
        .card, .metrics-card {{
            padding: 15px;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# Sidebar with improved navigation
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
            <h1 style="color: RoyalBlue; font-size: 3em; font-weight: bold; font-family: 'Orbitron', sans-serif;"> 💰 Fraud Detection</h1>
            <p style='color: Black; font-size: 1.4em; font-weight: bold;'>Advanced Fraud Detection System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check API status
    check_api_status()
    
    st.markdown("### API Status")
    flask_status = "🟢 Running" if st.session_state.api_status['flask'] else "🔴 Not Running"
    fastapi_status = "🟢 Running" if st.session_state.api_status['fastapi'] else "🔴 Not Running"
    
    st.markdown(f"""
        <div style='background-color: rgba(65,105,225,1); padding: 10px; border-radius: 10px; margin-bottom: 15px;'>
            <p style='color: white; margin: 0;'>Flask API: {flask_status}</p>
            <p style='color: white; margin: 0;'>FastAPI: {fastapi_status}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    st.markdown("### Model Selection")
    model_options = ["auto", "decision_tree", "random_forest", "xgboost", "ann"]
    selected_model = st.selectbox("Select Prediction Model", model_options)
    
    # Navigation
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Dataset Analysis", "Real-time Prediction", "Model Information", "Monitoring"],
        key="nav_radio"
    )
    
    # Update current page in session state
    st.session_state.current_page = page

# Main content based on selected page
if page == "Dashboard":
    st.markdown("""
        <div class="gradient-header">
            <h1 style='margin: 0;'>📊 Fraud Detection Dashboard</h1>
            <p style='margin: 0; opacity: 0.8;'>Real-time monitoring and analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # System overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='stMetric'>
                <h3 style='color: #666666; margin-bottom: 10px;'>System Status</h3>
                <h2 style='color: #2ecc71; margin: 0;'>Active</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_status = check_model_status()
        status = "Ready" if model_status else "Not Ready"
        color = "#2ecc71" if model_status else "#e74c3c"
        st.markdown(f"""
            <div class='stMetric'>
                <h3 style='color: #666666; margin-bottom: 10px;'>Model Status</h3>
                <h2 style='color: {color}; margin: 0;'>{status}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        dataset_status = "Loaded" if st.session_state.current_dataset is not None else "No Dataset"
        st.markdown(f"""
            <div class='stMetric'>
                <h3 style='color: #666666; margin-bottom: 10px;'>Dataset Status</h3>
                <h2 style='color: #3498db; margin: 0;'>{dataset_status}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    # # Quick actions
    # st.markdown("### Quick Actions")
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     if st.button("📤 Upload Dataset", key="upload_btn"):
    #         st.session_state.nav_radio = "Dataset Analysis"
    #         st.rerun()

    # with col2:
    #     if st.button("🔍 Make Prediction", key="predict_btn"):
    #         if not st.session_state.models_trained:
    #             st.error("Models are not ready. Please ensure models are trained first.")
    #         else:
    #             st.session_state.nav_radio = "Real-time Prediction"
    #             st.rerun()

    # with col3:
    #     if st.button("🤖 View Models", key="models_btn"):
    #         if not st.session_state.models_trained:
    #             st.error("Models are not ready. Please ensure models are trained first.")
    #         else:
    #             st.session_state.nav_radio = "Model Information"
    #             st.rerun()

    # Performance metrics section
    st.markdown("### Performance Metrics")

    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["📈 Fraud Detection Rate", "🎯 Model Performance", "💻 System Health"])

    # Add CSS class to tab content
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    with tab1:
        # Interactive fraud detection rate chart
        if st.session_state.recent_predictions:
            try:
                predictions_df = pd.DataFrame([
                    {
                        'Time': pred['time'],
                        'Fraud': pred['result']['is_fraud'],
                        'Confidence': pred['result']['probability']
                    }
                    for pred in st.session_state.recent_predictions
                ])
                
                # Create interactive time series chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=predictions_df['Time'],
                    y=predictions_df['Confidence'],
                    mode='lines+markers',
                    name='Confidence Score',
                    line=dict(color='#2196F3'),
                    hovertemplate='<b>Time:</b> %{x}<br><b>Confidence:</b> %{y:.2%}<extra></extra>'
                ))
                
                # Add fraud markers
                fraud_points = predictions_df[predictions_df['Fraud']]
                fig.add_trace(go.Scatter(
                    x=fraud_points['Time'],
                    y=fraud_points['Confidence'],
                    mode='markers',
                    name='Fraud Detected',
                    marker=dict(color='#f44336', size=10),
                    hovertemplate='<b>Fraud Detected!</b><br>Time: %{x}<br>Confidence: %{y:.2%}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Fraud Detection Confidence Over Time',
                    xaxis_title='Time',
                    yaxis_title='Confidence Score',
                    hovermode='x unified',
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating fraud detection chart: {e}")
        else:
            st.info("No prediction data available")

    with tab2:
        # Interactive model performance comparison
        if st.session_state.recent_predictions:
            try:
                model_performance = {}
                for pred in st.session_state.recent_predictions:
                    model = pred['result']['model_used']
                    if model not in model_performance:
                        model_performance[model] = {'correct': 0, 'total': 0}
                    model_performance[model]['total'] += 1
                    if pred['result']['is_fraud'] == (pred['result']['probability'] > 0.5):
                        model_performance[model]['correct'] += 1
                
                # Create performance chart
                fig = go.Figure()
                for model, perf in model_performance.items():
                    accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
                    fig.add_trace(go.Bar(
                        x=[model],
                        y=[accuracy],
                        name=model,
                        hovertemplate='<b>Model:</b> %{x}<br><b>Accuracy:</b> %{y:.2%}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title='Model Performance Comparison',
                    xaxis_title='Model',
                    yaxis_title='Accuracy',
                    barmode='group',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating model performance chart: {e}")
        else:
            st.info("No model performance data available")

    with tab3:
        # Interactive system health metrics
        try:
            # Calculate base health metrics
            api_status = check_api_status()
            api_health = (int(api_status['flask']) + int(api_status['fastapi'])) / 2 * 100
            
            model_health = 100 if st.session_state.models_trained else 0
            
            data_health = 100 if st.session_state.current_dataset is not None else 0
            
            # Adjust metrics based on alerts if they exist
            if st.session_state.alerts:
                alerts_df = pd.DataFrame(st.session_state.alerts)
                alerts_df['Time'] = pd.to_datetime(alerts_df['time'])
                
                # Reduce health metrics based on recent alerts
                api_errors = sum(1 for a in st.session_state.alerts if a['type'] == 'API Error')
                model_errors = sum(1 for a in st.session_state.alerts if a['type'] == 'Model Error')
                data_errors = sum(1 for a in st.session_state.alerts if a['type'] == 'Data Error')
                
                total_alerts = len(st.session_state.alerts)
                if total_alerts > 0:
                    api_health *= (1 - api_errors / total_alerts)
                    model_health *= (1 - model_errors / total_alerts)
                    data_health *= (1 - data_errors / total_alerts)
            
            # Create health metrics display
            health_data = {
                'Metric': ['API Health', 'Model Health', 'Data Health'],
                'Value': [api_health, model_health, data_health]
            }
            
            # Create gauge charts for each metric
            col1, col2, col3 = st.columns(3)
            
            for metric, value, col in zip(health_data['Metric'], health_data['Value'], [col1, col2, col3]):
                with col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': metric},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#2196F3"},
                            'steps': [
                                {'range': [0, 30], 'color': "#f44336"},
                                {'range': [30, 70], 'color': "#ff9800"},
                                {'range': [70, 100], 'color': "#4caf50"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 30
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Add system status details
            st.markdown("### System Status Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### API Status")
                st.markdown(f"- Flask API: {'🟢 Online' if api_status['flask'] else '🔴 Offline'}")
                st.markdown(f"- FastAPI: {'🟢 Online' if api_status['fastapi'] else '🔴 Offline'}")
            
            with col2:
                st.markdown("#### Model Status")
                st.markdown(f"- Models Loaded: {'🟢 Yes'}")
            
            # Show recent alerts if any
            if st.session_state.alerts:
                st.markdown("### Recent System Alerts")
                for alert in list(reversed(st.session_state.alerts))[:5]:  # Show last 5 alerts
                    severity_color = {
                        'High': '🔴',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }.get(alert['severity'], '⚪')
                    st.markdown(f"{severity_color} **{alert['type']}** - {alert['message']} _(at {alert['time'].strftime('%Y-%m-%d %H:%M:%S')})_")
            
        except Exception as e:
            st.error(f"Error creating system health metrics: {e}")
            logger.error(f"Error in system health tab: {e}")
    
    # Close tab content div
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive recent activity
    st.markdown("### Recent Activity")
    
    # Add refresh button
    if st.button("🔄 Refresh Data", key="refresh_btn"):
        st.rerun()
    
    # Recent predictions with interactive table
    st.markdown("#### Recent Predictions")
    if st.session_state.recent_predictions:
        predictions_df = pd.DataFrame([
            {
                'Time': pred['time'],
                'Model': pred['result']['model_used'],
                'Prediction': 'Fraud' if pred['result']['is_fraud'] else 'Not Fraud',
                'Confidence': f"{pred['result']['probability']:.2%}"
            }
            for pred in st.session_state.recent_predictions
        ])
        
        # Add interactive features to the table
        st.dataframe(
            predictions_df,
            use_container_width=True,
            column_config={
                "Time": st.column_config.DatetimeColumn(
                    "Time",
                    format="YYYY-MM-DD HH:mm:ss"
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.2f%%",
                    min_value=0,
                    max_value=1
                )
            }
        )
    else:
        st.info("No recent predictions")
    
    # Interactive alerts with filtering
    st.markdown("#### Recent Alerts")
    if st.session_state.alerts:
        # Add alert filter
        alert_types = st.multiselect(
            "Filter Alerts",
            options=['All'] + list(set(alert['type'] for alert in st.session_state.alerts)),
            default=['All']
        )
        
        filtered_alerts = st.session_state.alerts
        if 'All' not in alert_types:
            filtered_alerts = [alert for alert in filtered_alerts if alert['type'] in alert_types]
        
        for alert in reversed(filtered_alerts):
            severity_color = '#f44336' if alert['severity'] == 'High' else '#ff9800' if alert['severity'] == 'Medium' else '#4caf50'
            st.markdown(f"""
                <div class='alert-box alert-{alert['severity'].lower()}'>
                    <h4 style='margin: 0; color: {severity_color};'>{alert['type']}</h4>
                    <p style='margin: 5px 0; color: #666666;'>{alert['time'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p style='margin: 0;'>{alert['message']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent alerts")

elif page == "Dataset Analysis":
    st.title("📈 Dataset Analysis")
    
    # Custom CSS for file uploader
    st.markdown("""
        <style>
        /* Custom color for file uploader text */
        .stFileUploader > div > div > div > div > div > div {
            color: #d1a515 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Custom label and file uploader
    st.markdown("#### Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your dataset in CSV format", label_visibility="visible")
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.current_dataset = df
            st.success("Dataset loaded successfully!")
            
            # Display dataset info
            st.subheader("Dataset Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Fraudulent Transactions", df['Class'].sum() if 'Class' in df.columns else "N/A")
            
            # Data preview with enhanced styling
            st.subheader("Data Preview")
            st.dataframe(df.head().style.background_gradient(cmap='Blues'))
            
            # Enhanced Visualizations
            st.subheader("Data Analysis")
            
            # 1. Transaction Amount Distribution
            st.markdown("#### Transaction Amount Distribution")
            fig_amount = px.histogram(df, x='Amount', 
                                    color='Class' if 'Class' in df.columns else None,
                                    nbins=50,
                                    title='Distribution of Transaction Amounts',
                                    labels={'Amount': 'Transaction Amount', 'count': 'Frequency'},
                                    template='plotly_white')
            fig_amount.update_layout(bargap=0.1)
            st.plotly_chart(fig_amount, use_container_width=True)
            
            # 2. Time-based Analysis
            st.markdown("#### Time-based Analysis")
            if 'Time' in df.columns:
                fig_time = px.line(df, x='Time', y='Amount',
                                 color='Class' if 'Class' in df.columns else None,
                                 title='Transaction Amount Over Time',
                                 template='plotly_white')
                st.plotly_chart(fig_time, use_container_width=True)
            
            # 3. Feature Correlation Heatmap
            st.markdown("#### Feature Correlation")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix,
                               title='Feature Correlation Matrix',
                               color_continuous_scale='RdBu',
                               template='plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 4. Box Plots for Key Features
            st.markdown("#### Feature Distribution by Class")
            if 'Class' in df.columns:
                key_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
                available_features = [f for f in key_features if f in df.columns]
                
                for i in range(0, len(available_features), 2):
                    cols = st.columns(2)
                    for j, feature in enumerate(available_features[i:i+2]):
                        with cols[j]:
                            fig_box = px.box(df, x='Class', y=feature,
                                           color='Class',
                                           title=f'{feature} Distribution by Class',
                                           template='plotly_white')
                            st.plotly_chart(fig_box, use_container_width=True)
            
            # 5. Fraud Rate by Transaction Amount
            st.markdown("#### Fraud Rate by Transaction Amount")
            if 'Class' in df.columns:
                df['Amount_Bin'] = pd.qcut(df['Amount'], q=10)
                fraud_rate = df.groupby('Amount_Bin')['Class'].mean().reset_index()
                # Convert intervals to strings for plotting
                fraud_rate['Amount_Bin'] = fraud_rate['Amount_Bin'].astype(str)
                fig_fraud_rate = px.bar(
                    fraud_rate,
                    x='Amount_Bin',
                    y='Class',
                    title='Fraud Rate by Transaction Amount',
                    labels={'Class': 'Fraud Rate', 'Amount_Bin': 'Transaction Amount Range'},
                    color_discrete_sequence=[colors['error']]
                )
                fig_fraud_rate.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor=colors['bg_secondary'],
                    paper_bgcolor=colors['bg_secondary'],
                    font_color=colors['text_primary']
                )
                st.plotly_chart(fig_fraud_rate, use_container_width=True)
            
            # 6. Feature Importance (if available)
            if hasattr(st.session_state, 'feature_importance'):
                st.markdown("#### Feature Importance")
                fig_importance = px.bar(x=st.session_state.feature_importance.index,
                                      y=st.session_state.feature_importance.values,
                                      title='Feature Importance',
                                      labels={'x': 'Feature', 'y': 'Importance Score'},
                                      template='plotly_white')
                fig_importance.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_importance, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            add_alert("Data Processing Error", "High", str(e))

elif page == "Real-time Prediction":
    st.title("🔍 Real-time Prediction")
    
    if st.session_state.current_dataset is None:
        st.warning("Please upload a dataset first in the Dataset Analysis section.")
    else:
        # Get feature names from the dataset, excluding 'Class' and 'Amount_Bin' columns
        feature_names = [col for col in st.session_state.current_dataset.columns 
                        if col not in ['Class', 'Amount_Bin']]
        
        # Debug information in a collapsible section
        with st.expander("🔍 Debug Information", expanded=False):
            st.write(f"Number of features: {len(feature_names)}")
            st.write("Feature names:", feature_names)
        
        if not feature_names:
            st.error("No valid features found in the dataset. Please check your data.")
        else:
            # Add Random Value Generation Button
            if st.button("🎲 Generate Random Values"):
                if 'random_values' not in st.session_state:
                    st.session_state.random_values = {}
                
                # Decide if we want to generate a potentially fraudulent transaction (20% chance)
                generate_fraud = np.random.random() < 0.2
                
                # Generate random values based on feature statistics
                for feature in feature_names:
                    feature_data = st.session_state.current_dataset[feature]
                    if pd.api.types.is_numeric_dtype(feature_data):
                        if generate_fraud and 'Class' in st.session_state.current_dataset.columns:
                            # Get statistics from fraudulent transactions
                            fraud_data = feature_data[st.session_state.current_dataset['Class'] == 1]
                            if not fraud_data.empty:
                                mean = fraud_data.mean()
                                std = fraud_data.std()
                            else:
                                # If no fraud data, use extreme values
                                mean = feature_data.mean()
                                std = feature_data.std()
                                # Generate more extreme values
                                mean = mean + (2 * std * np.random.choice([-1, 1]))
                        else:
                            # Normal transaction statistics
                            mean = feature_data.mean()
                            std = feature_data.std()
                        
                        # Generate random value using feature statistics
                        # For potentially fraudulent transactions, increase the variance
                        if generate_fraud:
                            std = std * 2  # Double the standard deviation for more extreme values
                        
                        # Special handling for Time feature
                        if feature == 'Time':
                            # Generate time within last 24 hours (86400 seconds)
                            random_value = float(np.random.uniform(0, 86400))
                        else:
                            random_value = float(np.random.normal(mean, std))
                            
                            # For Amount feature, ensure it's non-negative and potentially generate suspicious amounts
                            if feature == 'Amount':
                                if generate_fraud:
                                    # Generate suspicious amounts (very small or very large)
                                    if np.random.random() < 0.5:
                                        random_value = float(np.random.uniform(0.01, 1.0))  # Very small amount
                                    else:
                                        random_value = float(np.random.uniform(1000, 5000))  # Large amount
                                random_value = max(0.0, random_value)
                        
                        st.session_state.random_values[feature] = random_value
                    else:
                        # For categorical features, if generating fraud, bias towards unusual values
                        unique_values = feature_data.unique()
                        if generate_fraud:
                            # Get value frequencies
                            value_counts = feature_data.value_counts()
                            # Bias towards less common values
                            probabilities = 1 / (value_counts + 1)  # Add 1 to avoid division by zero
                            probabilities = probabilities / probabilities.sum()
                            random_value = np.random.choice(unique_values, p=probabilities)
                        else:
                            random_value = np.random.choice(unique_values)
                        st.session_state.random_values[feature] = random_value
                
                if generate_fraud:
                    st.warning("Generated values that might indicate a fraudulent transaction!")
                else:
                    st.success("Random values generated for a typical transaction!")
            
            # Create input form
            st.subheader("Enter Transaction Details")
            features = []
            cols = st.columns(3)
            
            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    # Get the feature data
                    feature_data = st.session_state.current_dataset[feature]
                    
                    # Check if the feature is numeric
                    if pd.api.types.is_numeric_dtype(feature_data):
                        # Handle numeric features
                        # Use random value if available, otherwise use mean
                        default_val = st.session_state.random_values.get(feature, float(feature_data.mean())) if hasattr(st.session_state, 'random_values') else float(feature_data.mean())
                        
                        # Special handling for Amount feature
                        if feature == 'Amount':
                            value = st.number_input(
                                feature,
                                min_value=0.0,  # Amount should be non-negative
                                value=max(0.0, default_val),  # Ensure non-negative default for Amount
                                key=f"input_{feature}"
                            )
                        else:
                            # For PCA features (V1-V28), allow any value without min/max constraints
                            value = st.number_input(
                                feature,
                                value=default_val,
                                step=0.1,  # Add small step for fine control
                                format="%.6f",  # Show more decimal places
                                key=f"input_{feature}"
                            )
                    else:
                        # Handle categorical features
                        unique_values = feature_data.unique()
                        if len(unique_values) <= 10:
                            # Use random value if available, otherwise use first value
                            default_val = st.session_state.random_values.get(feature, unique_values[0]) if hasattr(st.session_state, 'random_values') else unique_values[0]
                            value = st.selectbox(
                                feature,
                                options=unique_values,
                                index=list(unique_values).index(default_val) if default_val in unique_values else 0,
                                key=f"input_{feature}"
                            )
                        else:
                            # Use random value if available, otherwise use first value
                            default_val = st.session_state.random_values.get(feature, str(unique_values[0])) if hasattr(st.session_state, 'random_values') else str(unique_values[0])
                            value = st.text_input(
                                feature,
                                value=str(default_val),
                                key=f"input_{feature}"
                            )
                    
                    features.append(value)
            
            # Add a clear button
            if st.button("🔄 Clear Values"):
                if 'random_values' in st.session_state:
                    del st.session_state.random_values
                st.rerun()
            
            if st.button("Predict", key="predict_button"):
                if len(features) != len(feature_names):
                    st.error("Please fill in all feature values")
                    add_alert("Prediction Error", "Medium", "Incomplete feature values")
                else:
                    with st.spinner("Making prediction..."):
                        try:
                            # Show API Status
                            st.info("Checking API Status...")
                            api_status = check_api_status()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Flask API", "Running" if api_status['flask'] else "Not Running")
                            with col2:
                                st.metric("FastAPI", "Running" if api_status['fastapi'] else "Not Running")
                            
                            # Convert features to the appropriate format for prediction
                            processed_features = []
                            processing_error = False
                            missing_features = []
                            
                            # Define the exact feature order expected by the scaler
                            required_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                            
                            # Create a mapping of feature names to their values
                            feature_map = dict(zip(feature_names, features))
                            
                            # Process features in the exact order expected by the scaler
                            for feature in required_features:
                                try:
                                    value = feature_map.get(feature)
                                    if value is not None:
                                        # Handle binned values
                                        if isinstance(value, str) and value.startswith("(") and value.endswith("]"):
                                            try:
                                                bounds = value.strip("()[]").split(",")
                                                lower = float(bounds[0].strip())
                                                upper = float(bounds[1].strip())
                                                processed_value = (lower + upper) / 2
                                            except Exception as e:
                                                st.error(f"Error processing binned value {value} for feature {feature}: {e}")
                                                processing_error = True
                                                break
                                        else:
                                            try:
                                                processed_value = float(value)
                                            except ValueError:
                                                st.error(f"Could not convert value '{value}' to float for feature {feature}")
                                                processing_error = True
                                                break
                                    else:
                                        # If feature is missing, use 0 as default
                                        processed_value = 0.0
                                        missing_features.append(feature)
                                    
                                    processed_features.append(processed_value)
                                    
                                except Exception as e:
                                    st.error(f"Error processing feature {feature}: {e}")
                                    processing_error = True
                                    break
                            
                            # Show all technical details in a single collapsible section
                            with st.expander("🔍 Technical Details", expanded=False):
                                st.write("### Feature Information")
                                st.write(f"Number of features: {len(features)}")
                                st.write("Feature names:", feature_names)
                                
                                # st.write("\n### Processing Results")
                                # if missing_features:
                                #     st.warning(f"Missing features (using 0 as default): {', '.join(missing_features)}")
                                
                                if len(processed_features) != 30:
                                    st.error(f"Expected 30 features but got {len(processed_features)}")
                                    processing_error = True
                                
                                st.write("\n### Processed Features")
                                feature_dict = dict(zip(required_features, processed_features))
                                st.json(feature_dict)
                            
                            if not processing_error:
                                # Make prediction
                                result = make_prediction(processed_features, selected_model)
                                
                                if result:
                                    # Add to prediction history
                                    add_prediction(processed_features, result)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown(f"""
                                            <div class='stMetric'>
                                                <h3 style='color: #666666; margin-bottom: 10px;'>Prediction</h3>
                                                <h2 style='color: {'#e74c3c' if result["is_fraud"] else '#2ecc71'}; margin: 0;'>
                                                    {'Fraud' if result["is_fraud"] else 'Not Fraud'}
                                                </h2>
                                                <p style='color: #666666; margin: 5px 0;'>
                                                    Confidence: {result['probability']:.2%}
                                                </p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                                    
                                    with col2:
                                        st.markdown(f"""
                                            <div class='stMetric'>
                                                <h3 style='color: #666666; margin-bottom: 10px;'>Model Used</h3>
                                                <h2 style='color: #3498db; margin: 0;'>{result["model_used"]}</h2>
                                            </div>
                                        """, unsafe_allow_html=True)
                                                    
                                    # Visualize prediction
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=result["probability"] * 100,
                                        title={'text': "Fraud Probability"},
                                        gauge={'axis': {'range': [0, 100]},
                                              'bar': {'color': "red" if result["is_fraud"] else "green"},
                                              'steps': [
                                                  {'range': [0, 30], 'color': "lightgreen"},
                                                  {'range': [30, 70], 'color': "yellow"},
                                                  {'range': [70, 100], 'color': "red"}
                                              ]}
                                    ))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add alert if fraud detected
                                    if result["is_fraud"]:
                                        add_alert("Fraud Detected", "High", 
                                                f"Transaction predicted as fraud with {result['probability']:.2%} confidence")
                        except Exception as e:
                            st.error(f"Error during prediction process: {str(e)}")
                            logger.error(f"Error during prediction process: {e}")
                            add_alert("Prediction Error", "High", str(e))

elif page == "Model Information":
    st.title("🤖 Model Information")
    
    # Check model status from FastAPI
    try:
        response = requests.get("http://localhost:8000/model/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            if status.get("model_status") == "ready":
                st.success("Models are ready and loaded")
                
                # Display model information
                st.subheader("Available Models")
                for model_name in status.get("loaded_models", []):
                    st.markdown(f"""
                        <div class='alert-box'>
                            <h3 style='margin: 0; color: #deac09'>{model_name}</h3>
                            <p style='margin: 5px 0; color: #deac09;'>Status: Loaded</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Model comparison
                st.subheader("Model Comparison")
                try:
                    # Get predictions from all models for a sample transaction
                    sample_features = [0.0] * 30  # Assuming 30 features
                    predictions = {}
                    
                    for model_name in status.get("loaded_models", []):
                        result = make_prediction(sample_features, model_name)
                        if result:
                            predictions[model_name] = result['probability']
                    
                    if predictions:
                        fig = px.bar(x=list(predictions.keys()), y=list(predictions.values()),
                                    title="Model Confidence Comparison",
                                    labels={'x': 'Model', 'y': 'Confidence Score'})
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not generate model comparison")
                    add_alert("Model Comparison Error", "Medium", str(e))
            else:
                st.warning("Models are not ready")
                st.json(status)  # Show detailed status
        else:
            st.error("Failed to get model status from API")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")

elif page == "Monitoring":
    st.title("📊 System Monitoring")
    
    # System metrics
    st.subheader("System Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_predictions = len(st.session_state.recent_predictions)
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        fraud_count = sum(1 for pred in st.session_state.recent_predictions 
                         if pred.get('result', {}).get('is_fraud', False))
        st.metric("Fraudulent Predictions", fraud_count)
    
    with col3:
        st.metric("Active Alerts", len(st.session_state.alerts))
    
    # Prediction trends
    st.subheader("Prediction Trends")
    if st.session_state.recent_predictions:
        try:
            predictions_df = pd.DataFrame([
                {
                    'Time': pred['time'],
                    'Fraud': pred.get('result', {}).get('is_fraud', False),
                    'Confidence': pred.get('result', {}).get('probability', 0.0)
                }
                for pred in st.session_state.recent_predictions
            ])
            
            if not predictions_df.empty:
                # Fraud rate over time
                predictions_df['Time'] = pd.to_datetime(predictions_df['Time'])
                fraud_rate = predictions_df.resample('H', on='Time')['Fraud'].mean()
                
                fig = px.line(x=fraud_rate.index, y=fraud_rate.values,
                             title='Fraud Rate Over Time',
                             labels={'x': 'Time', 'y': 'Fraud Rate'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                fig = px.histogram(predictions_df, x='Confidence',
                                  title='Prediction Confidence Distribution',
                                  nbins=20)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing prediction data: {e}")
    else:
        st.info("No prediction data available")
    
    # Alert analysis
    st.subheader("Alert Analysis")
    if st.session_state.alerts:
        try:
            alerts_df = pd.DataFrame(st.session_state.alerts)
            alerts_df['Time'] = pd.to_datetime(alerts_df['time'])
            
            # Alert count by severity
            severity_counts = alerts_df['severity'].value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                        title='Alert Distribution by Severity')
            st.plotly_chart(fig, use_container_width=True)
            
            # Alert timeline
            alerts_df['count'] = 1
            alert_timeline = alerts_df.resample('H', on='Time')['count'].sum()
            fig = px.line(x=alert_timeline.index, y=alert_timeline.values,
                         title='Alert Frequency Over Time',
                         labels={'x': 'Time', 'y': 'Alert Count'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing alert data: {e}")
    else:
        st.info("No alert data available")

# Footer with error handling
try:
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666666;'>
            <p>Advanced Fraud Detection System Dashboard</p>
            <p>All rights reserved</p>
        </div>
    """, unsafe_allow_html=True)
except Exception as e:
    logger.error(f"Error in footer: {e}") 




