import torch
import torch.nn as nn
from torchvision import models

class CarDamageModel(nn.Module):
    def __init__(self, num_classes=6, model_name="resnet50"):
        super(CarDamageModel, self).__init__()
        self.model_name = model_name
        
        if model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            # Replace the fc layer
            # Based on the inspected weights, fc is a Sequential block with at least an index 1.
            # We'll use Dropout and Linear.
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError("Unsupported model_name")

    def forward(self, x):
        return self.model(x)

def load_model(model_path, num_classes=6, device='cpu'):
    model = CarDamageModel(num_classes=num_classes, model_name="resnet50")
    if model_path:
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Try to load state dict. If it fails due to the fc layer, we can load with strict=False
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    model.to(device)
    model.eval()
    return model
