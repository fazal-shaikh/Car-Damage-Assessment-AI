import os
import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import base64

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import load_model
from data_loader.dataset import get_transforms, CLASSES
from utils.logger import get_logger

logger = get_logger("Backend")

app = FastAPI(title="Car Damage Detection API", description="API for classifying car damages", version="1.0.0")

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = get_transforms(is_train=False)

@app.on_event("startup")
def load_ml_model():
    global model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models", "saved_model.pth")
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path, num_classes=6, device=device)
    else:
        logger.warning("No model found. Predictions will be random initialized model.")
        model = load_model(None, num_classes=6, device=device)

@app.get("/")
def read_root():
    return {"message": "Welcome to Car Damage Detection API"}

DAMAGE_INFO = {
    'F_Breakage': {
        'severity': 'Medium',
        'min_cost': 15000,
        'max_cost': 45000,
        'recommendation': 'Front Bumper/Grille/Headlight replacement/repair. Align sensors.'
    },
    'F_Crushed': {
        'severity': 'High',
        'min_cost': 120000,
        'max_cost': 450000,
        'recommendation': 'Structural frame check, front radiator repair, bumper replacement, engine bay inspection.'
    },
    'F_Normal': {
        'severity': 'None',
        'min_cost': 0,
        'max_cost': 0,
        'recommendation': 'No front damage detected. Normal status.'
    },
    'R_Breakage': {
        'severity': 'Medium',
        'min_cost': 12000,
        'max_cost': 35000,
        'recommendation': 'Rear bumper repair, tail light assembly replacement.'
    },
    'R_Crushed': {
        'severity': 'High',
        'min_cost': 100000,
        'max_cost': 400000,
        'recommendation': 'Rear frame alignment, trunk lid replacement, exhaust system inspection.'
    },
    'R_Normal': {
        'severity': 'None',
        'min_cost': 0,
        'max_cost': 0,
        'recommendation': 'No rear damage detected. Normal status.'
    }
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Inference with gradients for saliency map
        model.eval()
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get gradient of predicted class
        score = outputs[0, predicted.item()]
        model.zero_grad()
        score.backward()
        
        # Generate Saliency Map
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze(0)
        
        # Normalize
        smin, smax = saliency.min(), saliency.max()
        if smax > smin:
            saliency = (saliency - smin) / (smax - smin + 1e-8)
        else:
            saliency = torch.zeros_like(saliency)
            
        saliency_np = (saliency.cpu().numpy() * 255).astype(np.uint8)
        
        # Create Heatmap (Red channel high saliency, Blue channel low)
        heatmap = np.zeros((saliency_np.shape[0], saliency_np.shape[1], 3), dtype=np.uint8)
        heatmap[:, :, 0] = saliency_np  # Red
        heatmap[:, :, 2] = 255 - saliency_np  # Blue
        
        # Superimpose
        heatmap_img = Image.fromarray(heatmap).resize(image.size, Image.Resampling.LANCZOS)
        blended_img = Image.blend(image, heatmap_img, alpha=0.45)
        
        # Encode to base64
        buffered = io.BytesIO()
        blended_img.save(buffered, format="JPEG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        class_name = CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        logger.info(f"Predicted: {class_name} with confidence: {confidence_score:.4f}")
        
        # All probabilities
        probs_dict = {CLASSES[i]: float(p) for i, p in enumerate(probabilities[0])}
        
        # Info
        info = DAMAGE_INFO.get(class_name, {
            'severity': 'Unknown',
            'min_cost': 0,
            'max_cost': 0,
            'recommendation': 'Manual assessment advised.'
        })
        
        return {
            "predicted_class": class_name,
            "confidence": float(confidence_score),
            "probabilities": probs_dict,
            "heatmap": heatmap_base64,
            "severity": info['severity'],
            "min_cost": info['min_cost'],
            "max_cost": info['max_cost'],
            "recommendation": info['recommendation'],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

