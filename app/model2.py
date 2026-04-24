import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c = x.shape
        s = self.squeeze(x.transpose(1, 2)).squeeze(-1)
        e = self.excitation(s).unsqueeze(1)
        return x * e

class DeiTWithSE(nn.Module):
    def __init__(self, num_classes):
        super(DeiTWithSE, self).__init__()
        self.backbone = timm.create_model(
            "deit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )
        feature_dim = 768
        self.se_block = SEBlock(channel=feature_dim, reduction=16)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.se_block(features)
        cls_token = features[:, 0, :]
        out = self.classifier(cls_token)
        return out

# Global variables
_model2 = None
CLASS_NAMES_2 = [
    "Acne",
    "Actinic Keratosis", 
    "Basal Cell Carcinoma", "Chickenpox",
    "Dermato Fibroma", "Dyshidrotic Eczema", "Melanoma",
    "Nail Fungus", "Nevus", "Normal Skin",
    "Pigmented Benign Keratosis", "Ringworm",
    "Seborrheic Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"
]

def load_model2():
    """Load Model 2 from weights file"""
    global _model2
    
    try:
        # Get current working directory and construct absolute path
        current_dir = os.getcwd()
        model2_path = os.path.join(current_dir, "weights", "model2.pth")
        print(f"Current working directory: {current_dir}")
        print(f"Looking for Model 2 at: {model2_path}")
        print(f"Model 2 file exists: {os.path.exists(model2_path)}")
        
        # Check if classes are configured
        if len(CLASS_NAMES_2) == 0:
            print("Model 2 classes not configured yet. Update CLASS_NAMES_2 in model2.py")
            return None
        
        # Check if weights file exists
        if not os.path.exists(model2_path):
            print(f"Model 2 weights not found at {model2_path}")
            return None
        
        # Initialize model
        model = DeiTWithSE(num_classes=len(CLASS_NAMES_2))
        model.to(DEVICE)
        
        # Load weights
        checkpoint = torch.load(model2_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        
        _model2 = model
        print("Model 2 loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading Model 2: {e}")
        return None

def predict2(image_tensor):
    """
    Make prediction using Model 2
    
    Args:
        image_tensor: Preprocessed image tensor (should be on correct device)
        
    Returns:
        Tuple: (class_name, confidence_float, all_probs_list) or error dict
    """
    global _model2
    
    try:
        # Check if classes are configured
        if len(CLASS_NAMES_2) == 0:
            return {"error": "Model 2 classes not configured yet. Update CLASS_NAMES_2 in model2.py"}
        
        # Check if model is loaded
        if _model2 is None:
            return {"error": "Model 2 not ready. Add model2.pth to weights/ folder"}
        
        # Ensure model is in eval mode
        _model2.eval()
        
        # Move input to device if needed
        if image_tensor.device != torch.device(DEVICE):
            image_tensor = image_tensor.to(DEVICE)
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = _model2(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Convert to numpy for easier handling
            probs_np = probabilities.cpu().numpy()[0]
            predicted_idx = predicted.item()
            confidence_float = confidence.item()
            
            # Get class name
            class_name = CLASS_NAMES_2[predicted_idx]
            
            # Create list of all probabilities (class_name, probability) tuples
            all_probs_list = [(CLASS_NAMES_2[i], float(probs_np[i])) for i in range(len(CLASS_NAMES_2))]
            
            return (class_name, confidence_float, all_probs_list)
            
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Initialize model on module import
if __name__ != "__main__":
    load_model2()
