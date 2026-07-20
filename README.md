# Car Damage Detection System

An end-to-end AI project that classifies real-time car damage images using Deep Learning.

## Features
- Deep Learning model using Transfer Learning (ResNet50).
- Hyperparameter tuning using Optuna.
- FastAPI backend serving a REST API for real-time predictions.
- Streamlit frontend for user-friendly interactions and displaying predictions along with confidence scores.
- Real-time logging and evaluation metrics.

## Tech Stack
- **Python**
- **PyTorch & Torchvision** (CNN, Transfer Learning)
- **Optuna** (Hyperparameter Tuning)
- **FastAPI** (Backend)
- **Streamlit** (Frontend)

## Project Structure
```
dataset/
├── backend/            # FastAPI application (main.py)
├── frontend/           # Streamlit application (app.py)
├── model/              # PyTorch model definitions (model.py)
├── dataset/            # Data loader & augmentation scripts (dataset.py)
├── training/           # Training pipeline with Optuna (train.py)
├── utils/              # Helper functions for logging and metrics (logger.py, metrics.py)
├── models/             # Saved PyTorch models (.pth files)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup Instructions

1. **Install Dependencies**
   Navigate to the `dataset` folder and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Backend (FastAPI)**
   Open a terminal, navigate to the `dataset` directory, and run:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```
   API Docs available at: `http://localhost:8000/docs`

3. **Run Frontend (Streamlit)**
   Open a separate terminal, navigate to the `dataset` directory, and run:
   ```bash
   streamlit run frontend/app.py
   ```

4. **Training (Optional)**
   To train a new model using Optuna:
   ```bash
   python training/train.py --epochs 5 --trials 10
   ```
   The best model will be saved in `models/best_model.pth`.
