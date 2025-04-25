# Dynamic Fraud Detection System

A comprehensive fraud detection system that supports dynamic dataset switching and multiple machine learning models.

## Features

- Dynamic dataset upload and processing
- Multiple model support (Decision Tree, Random Forest, XGBoost, ANN)
- Automatic model selection based on transaction characteristics
- Real-time transaction monitoring
- Interactive dashboard with visualizations
- RESTful API (FastAPI and Flask implementations)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

You can use either FastAPI or Flask implementation:

FastAPI:
```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

Flask:
```bash
python app.py
```

### Starting the Dashboard

```bash
streamlit run dashboard.py
```

## System Components

### 1. API Server
- FastAPI/Flask implementation
- Endpoints for dataset upload, model training, and predictions
- Automatic model selection
- Health monitoring

### 2. Dashboard
- Dataset management
- Model training and evaluation
- Real-time transaction monitoring
- Transaction analysis
- System settings

### 3. Models
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (ANN)

## Workflow

1. Upload a dataset through the dashboard
2. Preprocess and analyze the data
3. Train the models
4. Use the system for:
   - Real-time transaction monitoring
   - Individual transaction analysis
   - Model performance comparison

## API Endpoints

### FastAPI/Flask
- `POST /upload-dataset`: Upload and process a new dataset
- `POST /train-models`: Train models with current dataset
- `GET /health`: Check API health status
- `GET /models`: List available models and metrics
- `POST /predict`: Make predictions on transactions

## Error Handling

The system includes comprehensive error handling for:
- Dataset upload and processing
- Model training and evaluation
- API communication
- Real-time monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 



The services will run on the same ports:
- Flask API: http://localhost:5000
- FastAPI: http://localhost:8000
- Dashboard: http://localhost:8501

You can verify the services are running by checking their health endpoints:
- Flask API: http://localhost:5000/health
- FastAPI: http://localhost:8000/health