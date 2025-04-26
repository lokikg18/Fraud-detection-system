import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
# Remove local model loading if APIs are primary
# from tensorflow.keras.models import load_model
import logging
import requests
import tempfile
import os
# Remove local index import if relying solely on APIs
# from index import DataProcessor, FraudDetector
from plotly.subplots import make_subplots
import time
import random
import concurrent.futures

# --- Configuration: Replace with your deployed API URLs ---
# Best Practice: Use Streamlit Secrets for deployment
# Example using st.secrets (uncomment and configure secrets in Streamlit Cloud)
# FLASK_API_URL = st.secrets.get("FLASK_API_URL", "YOUR_FLASK_API_URL_HERE") # Provide default or placeholder
# FASTAPI_API_URL = st.secrets.get("FASTAPI_API_URL", "YOUR_FASTAPI_API_URL_HERE") # Provide default or placeholder

# --- OR --- (Less Recommended for deployment, okay for local testing)
# Hardcode your URLs here if not using secrets (Replace placeholders)
FLASK_API_URL = os.environ.get("FLASK_API_URL")  # Provide a default for local testing if needed
FASTAPI_API_URL = os.environ.get("FASTAPI_API_URL") # Provide a default

# Ensure URLs don't end with a slash
FLASK_API_URL = FLASK_API_URL.rstrip('/')
FASTAPI_API_URL = FASTAPI_API_URL.rstrip('/')
# --- End Configuration ---


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
    # Removed local DataProcessor/FraudDetector initialization if relying on APIs
    # try:
    #     if 'data_processor' not in st.session_state:
    #         st.session_state.data_processor = DataProcessor()
    #     if 'fraud_detector' not in st.session_state:
    #         st.session_state.fraud_detector = None
    # except Exception as e:
    #     logger.error(f"Error initializing local classes: {e}")
    #     # Decide if this is critical or if the app can run with APIs only

    try:
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        if 'models_trained' not in st.session_state: # This now reflects API status primarily
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

        # Check model status on initialization using APIs
        check_model_status() # This function now calls the API
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        st.error("Error initializing dashboard. Please refresh the page.")

# Removed local load_models function as we rely on APIs

def check_model_status():
    """Check if models are ready via the FastAPI API"""
    global FASTAPI_API_URL # Access the global URL variable
    try:
        # Check FastAPI model status endpoint
        url = f"{FASTAPI_API_URL}/model/status"
        logger.info(f"Checking model status at: {url}")
        response = requests.get(url, timeout=10) # Increased timeout for cold starts
        if response.status_code == 200:
            status_data = response.json()
            status = status_data.get("model_status")
            if status == "ready":
                st.session_state.models_trained = True
                logger.info("Models are ready (FastAPI check)")
                return True
            else:
                 logger.warning(f"Models not ready via API: {status_data}")
                 st.session_state.models_trained = False
                 return False
        else:
            logger.error(f"Error checking model status from FastAPI: {response.status_code} - {response.text}")
            st.session_state.models_trained = False
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to FastAPI for model status: {e}")
        st.session_state.models_trained = False
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking model status: {e}")
        st.session_state.models_trained = False
        return False

def make_prediction(features, model_name="auto"):
    """Make prediction using the deployed APIs"""
    global FASTAPI_API_URL, FLASK_API_URL # Access global URL variables
    try:
        # Log prediction attempt
        logger.info(f"Attempting prediction with model: {model_name}")
        # logger.info(f"Features: {features}") # Be careful logging potentially sensitive feature data

        # Try FastAPI first
        try:
            url = f"{FASTAPI_API_URL}/predict" # Use deployed URL
            payload = {
                "features": features,
                "model": model_name
            }
            logger.info(f"Sending request to FastAPI ({url}): {payload}")

            response = requests.post(url, json=payload, timeout=15) # Increased timeout
            logger.info(f"FastAPI Response Status: {response.status_code}")
            # logger.info(f"FastAPI Response Content: {response.text}") # Avoid logging full response if large

            if response.status_code == 200:
                logger.info("Prediction successful via FastAPI.")
                return response.json()
            else:
                error_msg = f"FastAPI prediction error ({response.status_code}): {response.text}"
                logger.error(error_msg)
                # Don't show full error text to user unless necessary
                st.warning(f"FastAPI Error ({response.status_code}). Trying Flask API...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"FastAPI prediction failed, trying Flask: {e}")
            st.warning(f"FastAPI unavailable: {str(e)}. Trying Flask API...")

        # Fallback to Flask
        try:
            url = f"{FLASK_API_URL}/predict" # Use deployed URL
            payload = {
                "features": features,
                "model": model_name
            }
            logger.info(f"Sending request to Flask API ({url}): {payload}")

            response = requests.post(url, json=payload, timeout=15) # Increased timeout
            logger.info(f"Flask Response Status: {response.status_code}")
            # logger.info(f"Flask Response Content: {response.text}")

            if response.status_code == 200:
                 logger.info("Prediction successful via Flask API.")
                 return response.json()
            else:
                error_msg = f"Flask API prediction error ({response.status_code}): {response.text}"
                logger.error(error_msg)
                st.error(f"Flask API Error ({response.status_code}). Prediction failed.")
                return None # Both failed
        except requests.exceptions.RequestException as e:
            error_msg = f"Flask prediction failed: {e}"
            logger.error(error_msg)
            st.error(f"Flask API unavailable: {str(e)}. Prediction failed.")
            return None # Both failed

        # This part should ideally not be reached if one API works
        # logger.error("Both prediction APIs failed")
        # st.error("Both APIs failed to respond. Please check API status.")
        # return None
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {e}"
        logger.exception(error_msg) # Log full traceback for unexpected errors
        st.error("An unexpected error occurred during prediction.")
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
            'features': features, # Be mindful of storing large feature lists
            'result': result
        }
        st.session_state.recent_predictions.append(prediction)
        if len(st.session_state.recent_predictions) > 10:  # Keep only last 10 predictions
            st.session_state.recent_predictions = st.session_state.recent_predictions[-10:]
    except Exception as e:
        logger.error(f"Error adding prediction: {e}")

def check_api_status():
    """Check status of both deployed APIs with improved error handling and caching"""
    global FLASK_API_URL, FASTAPI_API_URL # Access global URLs
    cache_key = "api_status"
    current_time = time.time()

    # Only check API every 30 seconds to improve performance
    if current_time - st.session_state.get('last_api_check', 0) < 30:
        cached_status = st.session_state.get('cache', {}).get(cache_key)
        if cached_status and current_time - cached_status.get("timestamp", 0) < 30:
             # logger.debug("Using cached API status")
             st.session_state.api_status = cached_status["status"] # Ensure state is updated
             return cached_status["status"]
        # else: cache expired or first time

    st.session_state.last_api_check = current_time
    logger.info("Checking API status...")

    flask_status = False
    fastapi_status = False

    def check_health(url):
        try:
            # Use the /health endpoint which should be lightweight
            response = requests.get(f"{url}/health", timeout=5) # Short timeout for health check
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health check failed for {url}: {e}")
            return False
        except Exception as e_inner:
             logger.error(f"Unexpected error during health check for {url}: {e_inner}")
             return False

    try:
        # Use concurrent requests for faster checking
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Check Flask API
            flask_future = executor.submit(check_health, FLASK_API_URL)
            # Check FastAPI
            fastapi_future = executor.submit(check_health, FASTAPI_API_URL)

            flask_status = flask_future.result()
            fastapi_status = fastapi_future.result()

        status = {
            'flask': flask_status,
            'fastapi': fastapi_status
        }
        st.session_state.api_status = status
        # Cache the status
        st.session_state.cache[cache_key] = {"status": status, "timestamp": current_time}
        logger.info(f"API Status updated: {status}")
        return status
    except Exception as e:
        logger.error(f"Unexpected error checking API status: {e}")
        status = { # Default to false on error
            'flask': False,
            'fastapi': False
        }
        st.session_state.api_status = status
        # Cache the status
        st.session_state.cache[cache_key] = {"status": status, "timestamp": current_time}
        return status

# Initialize session state
initialize_session_state()

# --- PAGE CONFIG AND STYLING (Keep as is) ---
# Page config with modern theme
st.set_page_config(
    page_title="Advanced Fraud Detection Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern color schemes (Keep the get_colors function as is)
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

# Add advanced CSS animations and transitions (Keep the CSS block as is)
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

# --- SIDEBAR (Keep most as is, API status check is already updated) ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
            <h1 style="color: RoyalBlue; font-size: 3em; font-weight: bold; font-family: 'Orbitron', sans-serif;"> 💰 Fraud Detection</h1>
            <p style='color: Black; font-size: 1.4em; font-weight: bold;'>Advanced Fraud Detection System</p>
        </div>
    """, unsafe_allow_html=True)

    # Check API status (called within the sidebar now)
    if st.button("🔄 Check API Status"):
        check_api_status()
        st.rerun() # Rerun to update the display immediately

    st.markdown("### API Status")
    # Use status from session state which is updated by check_api_status
    api_status = st.session_state.get('api_status', {'flask': False, 'fastapi': False})
    flask_status = "🟢 Running" if api_status['flask'] else "🔴 Not Running"
    fastapi_status = "🟢 Running" if api_status['fastapi'] else "🔴 Not Running"

    st.markdown(f"""
        <div style='background-color: rgba(65,105,225,1); padding: 10px; border-radius: 10px; margin-bottom: 15px;'>
            <p style='color: white; margin: 0;'>Flask API: {flask_status}</p>
            <p style='color: white; margin: 0;'>FastAPI: {fastapi_status}</p>
        </div>
    """, unsafe_allow_html=True)

    # Model selection
    st.markdown("### Model Selection")
    model_options = ["auto", "decision_tree", "random_forest", "xgboost", "ann"]
    selected_model = st.selectbox("Select Prediction Model", model_options, key="model_select")

    # Navigation
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Dataset Analysis", "Real-time Prediction", "Model Information", "Monitoring"],
        key="nav_radio"
    )

    # Update current page in session state
    st.session_state.current_page = page

# --- MAIN CONTENT PAGES (Keep structure, API calls are updated in functions) ---

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
        # Use API status for system status (consider both APIs)
        system_active = st.session_state.api_status.get('flask', False) or st.session_state.api_status.get('fastapi', False)
        status_text = "Active" if system_active else "Inactive"
        status_color = colors['success'] if system_active else colors['error']
        st.markdown(f"""
            <div class='metrics-card'>
                <h3 style='color: {colors['text_secondary']}; margin-bottom: 10px;'>System Status</h3>
                <h2 style='color: {status_color}; margin: 0;'>{status_text}</h2>
            </div>
        """, unsafe_allow_html=True)


    with col2:
        model_status_ready = check_model_status() # Call the updated function
        status = "Ready" if model_status_ready else "Not Ready"
        color = colors['success'] if model_status_ready else colors['error']
        st.markdown(f"""
            <div class='metrics-card'>
                <h3 style='color: {colors['text_secondary']}; margin-bottom: 10px;'>Model Status</h3>
                <h2 style='color: {color}; margin: 0;'>{status}</h2>
            </div>
        """, unsafe_allow_html=True)


    with col3:
        dataset_status = "Loaded" if st.session_state.current_dataset is not None else "No Dataset"
        color = colors['accent_1'] if st.session_state.current_dataset is not None else colors['warning']
        st.markdown(f"""
             <div class='metrics-card'>
                 <h3 style='color: {colors['text_secondary']}; margin-bottom: 10px;'>Dataset Status</h3>
                 <h2 style='color: {color}; margin: 0;'>{dataset_status}</h2>
             </div>
         """, unsafe_allow_html=True)

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
                        'Fraud': pred['result'].get('is_fraud', False), # Use .get for safety
                        'Confidence': pred['result'].get('probability', 0.0) # Use .get for safety
                    }
                    for pred in st.session_state.recent_predictions if 'result' in pred # Ensure result exists
                ])

                if not predictions_df.empty:
                    # Create interactive time series chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=predictions_df['Time'],
                        y=predictions_df['Confidence'],
                        mode='lines+markers',
                        name='Confidence Score',
                        line=dict(color=colors['chart_normal_1']),
                        hovertemplate='<b>Time:</b> %{x}<br><b>Confidence:</b> %{y:.2%}<extra></extra>'
                    ))

                    # Add fraud markers
                    fraud_points = predictions_df[predictions_df['Fraud']]
                    if not fraud_points.empty:
                        fig.add_trace(go.Scatter(
                            x=fraud_points['Time'],
                            y=fraud_points['Confidence'],
                            mode='markers',
                            name='Fraud Detected',
                            marker=dict(color=colors['chart_fraud_2'], size=10),
                            hovertemplate='<b>Fraud Detected!</b><br>Time: %{x}<br>Confidence: %{y:.2%}<extra></extra>'
                        ))

                    fig.update_layout(
                        title='Fraud Detection Confidence Over Time',
                        xaxis_title='Time',
                        yaxis_title='Confidence Score',
                        hovermode='x unified',
                        showlegend=True,
                        template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                        paper_bgcolor=colors['bg_secondary'],
                        plot_bgcolor=colors['bg_secondary'],
                        font_color=colors['text_primary']
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("No valid prediction results found.")
            except Exception as e:
                st.error(f"Error creating fraud detection chart: {e}")
                logger.exception("Error in fraud detection chart")
        else:
            st.info("No prediction data available")

    with tab2:
        # Interactive model performance comparison
        if st.session_state.recent_predictions:
            try:
                model_performance = {}
                for pred in st.session_state.recent_predictions:
                    if 'result' in pred and 'model_used' in pred['result']:
                        model = pred['result']['model_used']
                        if model not in model_performance:
                            model_performance[model] = {'total': 0, 'probability_sum': 0.0}
                        model_performance[model]['total'] += 1
                        model_performance[model]['probability_sum'] += pred['result'].get('probability', 0.0)

                if model_performance:
                    # Create performance chart (showing average confidence)
                    models = list(model_performance.keys())
                    avg_confidence = [
                        (perf['probability_sum'] / perf['total']) if perf['total'] > 0 else 0
                        for perf in model_performance.values()
                    ]

                    fig = go.Figure(data=[go.Bar(
                        x=models,
                        y=avg_confidence,
                        marker_color=colors['accent_1'],
                        hovertemplate='<b>Model:</b> %{x}<br><b>Avg Confidence:</b> %{y:.2%}<extra></extra>'
                    )])

                    fig.update_layout(
                        title='Model Average Prediction Confidence',
                        xaxis_title='Model',
                        yaxis_title='Average Confidence Score',
                        yaxis_tickformat='.0%',
                        template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                        paper_bgcolor=colors['bg_secondary'],
                        plot_bgcolor=colors['bg_secondary'],
                        font_color=colors['text_primary']
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid model performance data available.")
            except Exception as e:
                st.error(f"Error creating model performance chart: {e}")
                logger.exception("Error in model performance chart")
        else:
            st.info("No model performance data available")

    with tab3:
        # Interactive system health metrics
        try:
            # Calculate base health metrics
            api_status = check_api_status() # Use the updated function
            api_health = (int(api_status.get('flask', False)) + int(api_status.get('fastapi', False))) / 2 * 100

            model_health = 100 if st.session_state.get('models_trained', False) else 0

            data_health = 100 if st.session_state.current_dataset is not None else 0

            # Adjust metrics based on alerts if they exist
            if st.session_state.alerts:
                # Consider only recent alerts (e.g., last hour) for health impact
                now = datetime.now()
                recent_alerts = [a for a in st.session_state.alerts if now - a['time'] < timedelta(hours=1)]

                if recent_alerts:
                    api_errors = sum(1 for a in recent_alerts if a['type'] == 'API Error')
                    model_errors = sum(1 for a in recent_alerts if a['type'] == 'Model Error')
                    data_errors = sum(1 for a in recent_alerts if a['type'] == 'Data Error')

                    total_recent_alerts = len(recent_alerts)
                    # Apply a decay factor based on number of errors (max reduction 50%)
                    api_health *= max(0.5, 1 - (api_errors / 5)) # Max 5 errors impact
                    model_health *= max(0.5, 1 - (model_errors / 5))
                    data_health *= max(0.5, 1 - (data_errors / 5))


            # Create health metrics display
            health_data = {
                'Metric': ['API Health', 'Model Health', 'Data Health'],
                'Value': [api_health, model_health, data_health]
            }

            # Create gauge charts for each metric
            cols_health = st.columns(3)

            for metric, value, col in zip(health_data['Metric'], health_data['Value'], cols_health):
                with col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': metric, 'font': {'color': colors['text_primary']}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickfont': {'color': colors['text_secondary']}},
                            'bar': {'color': colors['accent_1']},
                            'bgcolor': colors['bg_tertiary'],
                            'borderwidth': 2,
                            'bordercolor': colors['card_border'],
                            'steps': [
                                {'range': [0, 30], 'color': colors['error']},
                                {'range': [30, 70], 'color': colors['warning']},
                                {'range': [70, 100], 'color': colors['success']}
                            ],
                            'threshold': {
                                'line': {'color': colors['error'], 'width': 4},
                                'thickness': 0.75,
                                'value': 30
                            }
                        }
                    ))
                    fig.update_layout(
                        height=250,
                        paper_bgcolor=colors['bg_secondary'],
                        plot_bgcolor=colors['bg_secondary'],
                        font_color=colors['text_primary']
                        )
                    st.plotly_chart(fig, use_container_width=True)

            # Add system status details
            st.markdown("### System Status Details")
            col1_details, col2_details = st.columns(2)

            with col1_details:
                 st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                 st.markdown("#### API Status")
                 # --- FIX START ---
                 # Construct HTML strings outside the f-string expression
                 online_html = "<span class='status-indicator status-active'></span> Online"
                 offline_html = "<span class='status-indicator status-inactive'></span> Offline"
                 flask_status_html = online_html if api_status.get('flask', False) else offline_html
                 fastapi_status_html = online_html if api_status.get('fastapi', False) else offline_html

                 st.markdown(f"- Flask API: {flask_status_html}", unsafe_allow_html=True)
                 st.markdown(f"- FastAPI: {fastapi_status_html}", unsafe_allow_html=True)
                 # --- FIX END ---
                 st.markdown(f"</div>", unsafe_allow_html=True)


            with col2_details:
                 st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                 st.markdown("#### Model Status")
                 # --- FIX START ---
                 # Construct HTML strings outside the f-string expression
                 yes_html = "<span class='status-indicator status-active'></span> Yes"
                 no_html = "<span class='status-indicator status-inactive'></span> No"
                 model_loaded_html = yes_html if st.session_state.get('models_trained', False) else no_html

                 st.markdown(f"- Models Loaded: {model_loaded_html}", unsafe_allow_html=True)
                 # --- FIX END ---
                 # Add more details if available from API status check
                 st.markdown(f"</div>", unsafe_allow_html=True)


            # Show recent alerts if any
            if st.session_state.alerts:
                st.markdown("### Recent System Alerts")
                for alert in list(reversed(st.session_state.alerts))[:5]:  # Show last 5 alerts
                    severity_color = colors['alert_high'] if alert['severity'] == 'High' else colors['alert_medium'] if alert['severity'] == 'Medium' else colors['alert_low']
                    st.markdown(f"""
                        <div class='alert-box' style='border-left: 5px solid {severity_color};'>
                            <h4 style='margin: 0; color: {colors['text_primary']};'>{alert['type']} ({alert['severity']})</h4>
                            <p style='margin: 5px 0; color: {colors['text_secondary']}; font-size: 0.9em;'>{alert['time'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p style='margin: 0; color: {colors['text_primary']};'>{alert['message']}</p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error creating system health metrics: {e}")
            logger.exception("Error in system health tab")

    # Close tab content div
    st.markdown('</div>', unsafe_allow_html=True)

    # Interactive recent activity
    st.markdown("### Recent Activity")

    # Add refresh button
    if st.button("🔄 Refresh Data", key="refresh_btn"):
        # Clear cache and rerun
        st.session_state.cache = {}
        st.session_state.last_api_check = 0
        st.rerun()

    # Recent predictions with interactive table
    st.markdown("#### Recent Predictions")
    if st.session_state.recent_predictions:
        try:
            predictions_data = []
            for pred in st.session_state.recent_predictions:
                 if 'result' in pred: # Check if result exists
                     predictions_data.append({
                         'Time': pred['time'],
                         'Model': pred['result'].get('model_used', 'N/A'),
                         'Prediction': 'Fraud' if pred['result'].get('is_fraud', False) else 'Not Fraud',
                         'Confidence': pred['result'].get('probability', 0.0) # Use .get for safety
                     })

            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                # Add interactive features to the table
                st.dataframe(
                    predictions_df.sort_values(by='Time', ascending=False), # Show newest first
                    use_container_width=True,
                    column_config={
                        "Time": st.column_config.DatetimeColumn(
                            "Time",
                            format="YYYY-MM-DD HH:mm:ss"
                        ),
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            help="The model's confidence score (probability of fraud)",
                            format="%.2f", # Display as decimal
                            min_value=0.0,
                            max_value=1.0,
                        )
                    },
                    hide_index=True # Hide the default index
                )
            else:
                st.info("No valid prediction results recorded yet.")
        except Exception as e:
            st.error("Error displaying recent predictions.")
            logger.exception("Error displaying recent predictions")
    else:
        st.info("No recent predictions")

    # Interactive alerts with filtering
    st.markdown("#### Recent Alerts")
    if st.session_state.alerts:
        try:
            # Add alert filter
            alert_types = ['All'] + sorted(list(set(alert['type'] for alert in st.session_state.alerts)))
            selected_alert_types = st.multiselect(
                "Filter Alerts by Type",
                options=alert_types,
                default=['All']
            )

            filtered_alerts = st.session_state.alerts
            if 'All' not in selected_alert_types:
                filtered_alerts = [alert for alert in filtered_alerts if alert['type'] in selected_alert_types]

            if filtered_alerts:
                 # Sort alerts newest first
                 sorted_alerts = sorted(filtered_alerts, key=lambda x: x['time'], reverse=True)
                 for alert in sorted_alerts:
                    severity_color = colors['alert_high'] if alert['severity'] == 'High' else colors['alert_medium'] if alert['severity'] == 'Medium' else colors['alert_low']
                    st.markdown(f"""
                         <div class='alert-box' style='border-left: 5px solid {severity_color};'>
                             <h4 style='margin: 0; color: {colors['text_primary']};'>{alert['type']} ({alert['severity']})</h4>
                             <p style='margin: 5px 0; color: {colors['text_secondary']}; font-size: 0.9em;'>{alert['time'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                             <p style='margin: 0; color: {colors['text_primary']};'>{alert['message']}</p>
                         </div>
                     """, unsafe_allow_html=True)
            else:
                 st.info("No alerts match the selected filter.")
        except Exception as e:
            st.error("Error displaying alerts.")
            logger.exception("Error displaying alerts")
    else:
        st.info("No recent alerts")


elif page == "Dataset Analysis":
    st.title("📈 Dataset Analysis")

    # Custom CSS for file uploader
    st.markdown(f"""
        <style>
        /* Custom color for file uploader text */
        [data-testid="stFileUploader"] label span {{
             color: {colors['accent_1']} !important;
        }}
         [data-testid="stFileUploader"] section button {{
             border-color: {colors['accent_1']} !important;
             color: {colors['accent_1']} !important;
         }}
        </style>
    """, unsafe_allow_html=True)

    # Custom label and file uploader
    st.markdown("#### Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your dataset in CSV format. Ensure it has features (like V1-V28, Time, Amount) and a 'Class' column (0 for normal, 1 for fraud).",
        label_visibility="collapsed" # Use markdown above for label
        )

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
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                if 'Class' in df.columns:
                    fraud_count = df['Class'].sum()
                    st.metric("Fraudulent Transactions", f"{fraud_count:,} ({fraud_count/len(df):.2%})")
                else:
                    st.metric("Fraudulent Transactions", "N/A ('Class' column missing)")

            # Data preview with enhanced styling
            st.subheader("Data Preview")
            st.dataframe(df.head().style.background_gradient(cmap='viridis' if st.session_state.dark_mode else 'Blues'))

            # Enhanced Visualizations
            st.subheader("Data Analysis")

            # Check if 'Class' column exists for relevant plots
            class_col_exists = 'Class' in df.columns

            # 1. Transaction Amount Distribution
            st.markdown("#### Transaction Amount Distribution")
            fig_amount = px.histogram(df, x='Amount',
                                    color='Class' if class_col_exists else None,
                                    color_discrete_map={0: colors['chart_normal_1'], 1: colors['chart_fraud_1']} if class_col_exists else {},
                                    nbins=50,
                                    title='Distribution of Transaction Amounts',
                                    labels={'Amount': 'Transaction Amount ($)', 'count': 'Frequency'},
                                    template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                                    opacity=0.7)
            fig_amount.update_layout(bargap=0.1, paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
            st.plotly_chart(fig_amount, use_container_width=True)

            # 2. Time-based Analysis (if 'Time' exists)
            if 'Time' in df.columns:
                st.markdown("#### Transactions Over Time")
                # Convert 'Time' (seconds) to a more interpretable format if needed (e.g., hour of day)
                df['Hour'] = (df['Time'] / 3600) % 24 # Assuming time is seconds from start
                time_analysis = df.groupby('Hour').size().reset_index(name='Count')
                fig_time = px.line(time_analysis, x='Hour', y='Count',
                                 title='Transaction Count by Hour of Day',
                                 template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                                 markers=True)
                fig_time.update_layout(paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
                st.plotly_chart(fig_time, use_container_width=True)

                if class_col_exists:
                     fraud_time_analysis = df[df['Class']==1].groupby('Hour').size().reset_index(name='Fraud Count')
                     fig_fraud_time = px.line(fraud_time_analysis, x='Hour', y='Fraud Count',
                                      title='Fraudulent Transaction Count by Hour of Day',
                                      template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                                      markers=True, color_discrete_sequence=[colors['chart_fraud_2']])
                     fig_fraud_time.update_layout(paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
                     st.plotly_chart(fig_fraud_time, use_container_width=True)


            # 3. Feature Correlation Heatmap
            st.markdown("#### Feature Correlation")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix,
                                   title='Feature Correlation Matrix',
                                   color_continuous_scale='RdBu_r' if st.session_state.dark_mode else 'RdBu',
                                   template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                                   zmin=-1, zmax=1) # Ensure consistent scale
                fig_corr.update_layout(paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough numeric columns for a correlation matrix.")

            # 4. Box Plots for Key Features (if Class exists)
            if class_col_exists:
                st.markdown("#### Feature Distribution by Class")
                # Select features with significant differences or importance (e.g., V1-V17 often show separation)
                key_features = [f'V{i}' for i in range(1, 18)] + ['Amount']
                available_features = [f for f in key_features if f in df.columns]

                if available_features:
                    feature_select = st.multiselect(
                        "Select features for Box Plots:",
                        options=available_features,
                        default=available_features[:6] # Default to first 6
                    )

                    if feature_select:
                         num_plots = len(feature_select)
                         cols_box = st.columns(min(num_plots, 3)) # Max 3 columns
                         for i, feature in enumerate(feature_select):
                             with cols_box[i % 3]:
                                 fig_box = px.box(df, x='Class', y=feature,
                                                color='Class',
                                                color_discrete_map={0: colors['chart_normal_1'], 1: colors['chart_fraud_1']},
                                                title=f'{feature} Distribution',
                                                template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                                                points=False) # Hide outliers for cleaner look
                                 fig_box.update_layout(paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
                                 st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("No key features (V1-V17, Amount) found for box plots.")

            # 5. Fraud Rate by Transaction Amount (if Class exists)
            if class_col_exists and 'Amount' in df.columns:
                st.markdown("#### Fraud Rate by Transaction Amount")
                # Create bins for Amount
                # Use pd.qcut for quantile-based bins, or pd.cut for equal-width bins
                try:
                    # Handle potential zero amounts causing issues with qcut
                    df_amount_filtered = df[df['Amount'] > 0]
                    if not df_amount_filtered.empty:
                         df_amount_filtered['Amount_Bin'] = pd.qcut(df_amount_filtered['Amount'], q=10, duplicates='drop')
                         fraud_rate = df_amount_filtered.groupby('Amount_Bin', observed=False)['Class'].mean().reset_index()
                         # Convert intervals to strings for plotting
                         fraud_rate['Amount_Bin'] = fraud_rate['Amount_Bin'].astype(str)
                         fig_fraud_rate = px.bar(
                             fraud_rate,
                             x='Amount_Bin',
                             y='Class',
                             title='Fraud Rate by Transaction Amount Quantile',
                             labels={'Class': 'Fraud Rate', 'Amount_Bin': 'Transaction Amount Range'},
                             color_discrete_sequence=[colors['chart_fraud_2']]
                         )
                         fig_fraud_rate.update_layout(
                             xaxis_tickangle=-45,
                             yaxis_tickformat='.1%',
                             template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                             paper_bgcolor=colors['bg_secondary'],
                             plot_bgcolor=colors['bg_secondary'],
                             font_color=colors['text_primary']
                         )
                         st.plotly_chart(fig_fraud_rate, use_container_width=True)
                    else:
                         st.info("No transactions with Amount > 0 found for quantile analysis.")
                except Exception as e:
                    st.warning(f"Could not generate Fraud Rate by Amount plot: {e}")
                    logger.warning(f"Error in fraud rate by amount plot: {e}")


            # 6. Feature Importance (if available from API or local state)
            # This part would ideally fetch importance from an API endpoint if models are trained via API
            # Placeholder:
            # if 'feature_importance' in st.session_state and st.session_state.feature_importance:
            #     st.markdown("#### Feature Importance")
            #     # Plot feature importance (assuming it's a dict or Series)
            #     pass


        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            logger.exception("Error processing uploaded dataset")
            add_alert("Data Processing Error", "High", str(e))
    else:
         st.info("Upload a CSV dataset to begin analysis.")


elif page == "Real-time Prediction":
    st.title("🔍 Real-time Prediction")

    # Check if models are ready via API before allowing prediction
    if not check_model_status():
         st.warning("⚠️ Models are not ready or API is unavailable. Please check system status.")
    else:
        # Determine feature names - Ideally fetch from API or use a fixed known list
        # Using a fixed list based on common credit card fraud datasets
        required_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        num_features = len(required_features)

        # Debug information in a collapsible section
        with st.expander("🔍 Feature Information", expanded=False):
            st.write(f"Number of features expected by model: {num_features}")
            st.write("Expected features:", required_features)
            st.caption("Input values for these features below.")

        # Add Random Value Generation Button
        if st.button("🎲 Generate Random Sample Values"):
            if 'random_values' not in st.session_state:
                st.session_state.random_values = {}

            # Generate random values (simple uniform/normal for now)
            for feature in required_features:
                if feature == 'Time':
                    random_value = float(np.random.uniform(0, 172800)) # Up to 2 days in seconds
                elif feature == 'Amount':
                    random_value = float(np.random.uniform(0.01, 500)) # Common amount range
                else: # V1-V28 features (PCA components, typically centered around 0)
                    random_value = float(np.random.normal(0, 1)) # Assume standard normal distribution
                st.session_state.random_values[feature] = random_value
            st.success("Random sample values generated!")
            # Use st.rerun() to immediately populate the fields with generated values
            time.sleep(0.5) # Short delay before rerun
            st.rerun()


        # Create input form
        st.subheader("Enter Transaction Details")
        input_features = {}
        # Use columns for better layout
        num_cols = 3
        cols_input = st.columns(num_cols)

        for i, feature in enumerate(required_features):
            with cols_input[i % num_cols]:
                # Use random value if available, otherwise use a sensible default (e.g., 0)
                default_val = st.session_state.get('random_values', {}).get(feature, 0.0)

                # Use number_input for all features
                value = st.number_input(
                    label=feature,
                    value=float(default_val), # Ensure default is float
                    step=0.1 if feature != 'Time' else 1.0,
                    format="%.6f" if feature.startswith('V') else "%.2f" if feature == 'Amount' else "%.0f",
                    key=f"input_{feature}"
                )
                input_features[feature] = value

        # Get the final list of features in the correct order
        features_list = [input_features[feature] for feature in required_features]

        # Add a clear button
        if st.button("🔄 Clear Values"):
            if 'random_values' in st.session_state:
                del st.session_state.random_values
            st.rerun()

        if st.button("Predict Fraud", key="predict_button", type="primary"):
            # Basic validation
            if len(features_list) != num_features:
                st.error(f"Incorrect number of features provided. Expected {num_features}, got {len(features_list)}.")
                add_alert("Prediction Error", "Medium", "Incorrect number of features")
            else:
                with st.spinner("Making prediction via API..."):
                    try:
                        # Show API Status during prediction attempt
                        with st.expander("API Status Check", expanded=False):
                            api_status = check_api_status()
                            st.metric("Flask API", "Running" if api_status.get('flask', False) else "Not Running")
                            st.metric("FastAPI", "Running" if api_status.get('fastapi', False) else "Not Running")

                        # Make prediction using the selected model from sidebar
                        selected_model_from_sidebar = st.session_state.get('model_select', 'auto')
                        result = make_prediction(features_list, selected_model_from_sidebar)

                        if result:
                            # Add to prediction history
                            add_prediction(features_list, result) # Store the input features too

                            # Display results prominently
                            is_fraud = result.get("is_fraud", False)
                            probability = result.get("probability", 0.0)
                            model_used = result.get("model_used", "N/A")

                            st.markdown("---")
                            st.subheader("Prediction Result")
                            cols_result = st.columns(2)

                            with cols_result[0]:
                                if is_fraud:
                                    st.error("🚨 Prediction: FRAUD DETECTED")
                                else:
                                    st.success("✅ Prediction: Not Fraud")
                                st.metric("Confidence (Fraud Probability)", f"{probability:.2%}")

                            with cols_result[1]:
                                st.info(f"Model Used: {model_used}")
                                # Optionally show timestamp from result if available
                                # st.caption(f"Timestamp: {result.get('timestamp', 'N/A')}")


                            # Visualize prediction confidence
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=probability * 100,
                                title={'text': "Fraud Probability Gauge", 'font': {'color': colors['text_primary']}},
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={'axis': {'range': [0, 100], 'tickfont': {'color': colors['text_secondary']}},
                                      'bar': {'color': colors['error'] if is_fraud else colors['success']},
                                      'bgcolor': colors['bg_tertiary'],
                                      'borderwidth': 2,
                                      'bordercolor': colors['card_border'],
                                      'steps': [
                                          {'range': [0, 50], 'color': colors['bg_secondary']}, # Threshold at 50%
                                          {'range': [50, 100], 'color': colors['bg_tertiary']}
                                      ],
                                      'threshold': {
                                          'line': {'color': colors['warning'], 'width': 4},
                                          'thickness': 0.75,
                                          'value': 50 # Standard 50% threshold line
                                      }}
                            ))
                            fig.update_layout(
                                height=250,
                                paper_bgcolor=colors['bg_secondary'],
                                font_color=colors['text_primary']
                                )
                            st.plotly_chart(fig, use_container_width=True)

                            # Add alert if fraud detected
                            if is_fraud:
                                add_alert("Fraud Detected", "High",
                                        f"Transaction predicted as fraud with {probability:.2%} confidence using {model_used} model.")
                        else:
                            # Error message already shown by make_prediction
                            add_alert("Prediction Error", "High", "API call failed or returned invalid result.")
                    except Exception as e:
                        st.error(f"Error during prediction process: {str(e)}")
                        logger.exception("Error during prediction process")
                        add_alert("Prediction Error", "High", str(e))

elif page == "Model Information":
    st.title("🤖 Model Information")

    # Check model status from FastAPI
    st.info("Fetching model status from API...")
    try:
        url = f"{FASTAPI_API_URL}/model/status"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            if status_data.get("model_status") == "ready":
                st.success("✅ Models are ready and loaded according to the API.")

                # Display model information in cards
                st.subheader("Available Models")
                loaded_models = status_data.get("loaded_models", [])
                if loaded_models:
                    cols_models = st.columns(min(len(loaded_models), 4)) # Max 4 columns
                    for i, model_name in enumerate(loaded_models):
                        with cols_models[i % 4]:
                             st.markdown(f"""
                                 <div class='card glass-effect' style='text-align: center;'>
                                     <h3 style='margin: 0; color: {colors['accent_1']}'>{model_name.replace('_', ' ').title()}</h3>
                                     <p style='margin: 5px 0; color: {colors['text_secondary']}; font-size: 0.9em;'>Status: Loaded</p>
                                 </div>
                             """, unsafe_allow_html=True)
                else:
                    st.warning("API reported status 'ready' but did not list any loaded models.")

                # Model comparison (using sample prediction confidence)
                st.subheader("Model Confidence Comparison (Sample)")
                st.caption("Confidence scores based on predicting a sample 'zero-feature' transaction.")
                try:
                    # Get predictions from all models for a sample transaction
                    # Use a simple, consistent input for comparison
                    sample_features = [0.0] * 30 # Assuming 30 features based on common datasets
                    predictions = {}
                    confidences = {}

                    with st.spinner("Querying models for comparison..."):
                        for model_name in loaded_models:
                            # Use the make_prediction function which handles API calls
                            result = make_prediction(sample_features, model_name)
                            if result and 'probability' in result:
                                predictions[model_name] = result['probability']
                                confidences[model_name] = f"{result['probability']:.2%}" # Store formatted confidence
                            else:
                                predictions[model_name] = 0.0 # Assign 0 if prediction failed
                                confidences[model_name] = "N/A"


                    if predictions:
                        # Prepare data for chart and table
                        comparison_df = pd.DataFrame({
                            'Model': list(predictions.keys()),
                            'Sample Confidence': list(predictions.values()),
                            'Formatted Confidence': list(confidences.values())
                        }).sort_values(by='Sample Confidence', ascending=False)

                        # Bar chart
                        fig = px.bar(comparison_df, x='Model', y='Sample Confidence',
                                    title="Model Confidence on Sample Transaction",
                                    labels={'Model': 'Model', 'Sample Confidence': 'Confidence Score'},
                                    color='Model', # Color bars by model
                                    text='Formatted Confidence', # Show formatted confidence on bars
                                    template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
                        fig.update_layout(
                            yaxis_tickformat='.0%',
                            paper_bgcolor=colors['bg_secondary'],
                            plot_bgcolor=colors['bg_secondary'],
                            font_color=colors['text_primary']
                            )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)

                        # Optional: Display as table
                        # st.dataframe(comparison_df[['Model', 'Formatted Confidence']], hide_index=True)
                    else:
                        st.warning("Could not get confidence scores from models for comparison.")

                except Exception as e:
                    st.warning(f"Could not generate model comparison: {e}")
                    logger.exception("Error during model comparison")
                    add_alert("Model Comparison Error", "Medium", str(e))
            else:
                st.warning("⚠️ Models are not ready according to the API.")
                st.json(status_data)  # Show detailed status from API
        else:
            st.error(f"Failed to get model status from API (Status Code: {response.status_code}). Is the API running and accessible at {FASTAPI_API_URL}?")
            add_alert("API Error", "High", f"Failed to get model status from FastAPI ({response.status_code})")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to FastAPI: {e}")
        add_alert("API Error", "High", f"Connection error checking model status: {e}")
    except Exception as e_main:
         st.error(f"An unexpected error occurred: {e_main}")
         logger.exception("Error on Model Information page")


elif page == "Monitoring":
    st.title("📊 System Monitoring")

    # System metrics
    st.subheader("System Metrics Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_predictions = len(st.session_state.recent_predictions)
        st.metric("Total Predictions (Last 10)", total_predictions)

    with col2:
        fraud_count = sum(1 for pred in st.session_state.recent_predictions
                         if pred.get('result', {}).get('is_fraud', False))
        st.metric("Fraudulent Predictions (Last 10)", fraud_count)

    with col3:
        st.metric("Active Alerts (Last 10)", len(st.session_state.alerts))

    # Prediction trends
    st.subheader("Prediction Trends (Last 10 Predictions)")
    if st.session_state.recent_predictions:
        try:
            predictions_data = []
            for pred in st.session_state.recent_predictions:
                 if 'result' in pred: # Ensure result exists
                     predictions_data.append({
                         'Time': pred['time'],
                         'Fraud': int(pred['result'].get('is_fraud', False)), # Convert bool to int for aggregation
                         'Confidence': pred['result'].get('probability', 0.0)
                     })

            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                predictions_df['Time'] = pd.to_datetime(predictions_df['Time'])
                predictions_df = predictions_df.sort_values(by='Time') # Sort by time for plotting

                # Fraud rate over time (using a rolling window if more data was available)
                # For only 10 points, just plot the fraud indicator
                fig_fraud_trend = px.scatter(predictions_df, x='Time', y='Fraud',
                                     title='Fraudulent Predictions Over Time (Last 10)',
                                     labels={'Time': 'Time', 'Fraud': 'Fraud (1=Yes, 0=No)'},
                                     color='Fraud',
                                     color_continuous_scale=[colors['chart_normal_1'], colors['chart_fraud_2']],
                                     template='plotly_dark' if st.session_state.dark_mode else 'plotly_white'
                                     )
                fig_fraud_trend.update_layout(yaxis=dict(range=[-0.1, 1.1], tickvals=[0, 1]), coloraxis_showscale=False,
                                     paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
                st.plotly_chart(fig_fraud_trend, use_container_width=True)

                # Confidence distribution
                fig_conf_dist = px.histogram(predictions_df, x='Confidence',
                                  title='Prediction Confidence Distribution (Last 10)',
                                  labels={'Confidence': 'Fraud Probability'},
                                  nbins=5, # Fewer bins for small sample size
                                  template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
                                  color_discrete_sequence=[colors['accent_1']])
                fig_conf_dist.update_layout(yaxis_title='Count', paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
                st.plotly_chart(fig_conf_dist, use_container_width=True)
            else:
                 st.info("No valid prediction results to analyze.")
        except Exception as e:
            st.error(f"Error processing prediction data for monitoring: {e}")
            logger.exception("Error processing prediction data for monitoring")
    else:
        st.info("No prediction data available for trend analysis.")

    # Alert analysis
    st.subheader("Alert Analysis (Last 10 Alerts)")
    if st.session_state.alerts:
        try:
            alerts_df = pd.DataFrame(st.session_state.alerts)
            alerts_df['Time'] = pd.to_datetime(alerts_df['time'])
            alerts_df = alerts_df.sort_values(by='Time') # Sort by time

            # Alert count by severity
            severity_counts = alerts_df['severity'].value_counts()
            fig_sev = px.pie(values=severity_counts.values, names=severity_counts.index,
                        title='Alert Distribution by Severity (Last 10)',
                        color_discrete_map={'High': colors['alert_high'], 'Medium': colors['alert_medium'], 'Low': colors['alert_low']},
                        template='plotly_dark' if st.session_state.dark_mode else 'plotly_white'
                        )
            fig_sev.update_layout(paper_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
            st.plotly_chart(fig_sev, use_container_width=True)

            # Alert timeline (simple scatter plot for few points)
            fig_time = px.scatter(alerts_df, x='Time', y='severity', color='severity',
                         title='Alert Timeline (Last 10)',
                         labels={'Time': 'Time', 'severity': 'Severity Level'},
                         color_discrete_map={'High': colors['alert_high'], 'Medium': colors['alert_medium'], 'Low': colors['alert_low']},
                         template='plotly_dark' if st.session_state.dark_mode else 'plotly_white'
                         )
            fig_time.update_layout(yaxis={'categoryorder':'array', 'categoryarray':['Low', 'Medium', 'High']}, # Order severity levels
                                   paper_bgcolor=colors['bg_secondary'], plot_bgcolor=colors['bg_secondary'], font_color=colors['text_primary'])
            st.plotly_chart(fig_time, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing alert data for monitoring: {e}")
            logger.exception("Error processing alert data for monitoring")
    else:
        st.info("No alert data available for analysis.")


# --- FOOTER (Keep as is) ---
# Footer with error handling
try:
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: {colors['text_secondary']};'>
            <p>Advanced Fraud Detection System Dashboard</p>
            <p>© {datetime.now().year} - All rights reserved</p>
        </div>
    """, unsafe_allow_html=True)
except Exception as e:
    logger.error(f"Error in footer: {e}")
