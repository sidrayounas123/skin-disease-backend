import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global average pooling
        squeezed = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        
        # Excitation
        weights = self.fc1(squeezed)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = torch.sigmoid(weights)
        
        # Scale: Apply channel-wise weights
        weights = weights.view(batch_size, channels, 1, 1)
        return x * weights

class ModifiedDeiT(nn.Module):
    """Modified DeiT model with SE Block and custom classifier"""
    def __init__(self, num_classes=6):
        super(ModifiedDeiT, self).__init__()
        
        # Base DeiT model without classification head
        self.backbone = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=0)
        
        # Custom classifier layers (matching saved model)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # SE Block for feature refinement
        self.se = SEBlock(512)
        
    def forward(self, x):
        # Extract features using DeiT backbone
        features = self.backbone(x)  # Shape: (batch_size, 768)
        
        # First linear layer
        x = self.fc1(features)  # Linear(768, 512)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Reshape for SE Block (add spatial dimensions)
        batch_size, channels = x.size()
        x = x.view(batch_size, channels, 1, 1)  # (batch_size, 512, 1, 1)
        
        # Apply SE Block
        x = self.se(x)
        x = x.view(batch_size, channels)  # Back to (batch_size, 512)
        
        # Continue with classifier
        x = self.fc2(x)  # Linear(512, 256)
        x = self.relu(x)
        x = self.output(x)  # Linear(256, num_classes)
        
        return x

# Global variables
_model1 = None
CLASS_NAMES_1 = ["AD (Atopic Dermatitis)", "CD (Contact Dermatitis)", "EC (Eczema)", 
                  "SC (Scabies)", "SD (Seborrheic Dermatitis)", "TC (Tinea Corporis)"]

def load_model1():
    """Load Model 1 from weights file"""
    global _model1
    
    try:
        # Get current working directory and construct absolute path
        current_dir = os.getcwd()
        model1_path = os.path.join(current_dir, "weights", "model1.pth")
        print(f"Current working directory: {current_dir}")
        print(f"Looking for Model 1 at: {model1_path}")
        print(f"Model 1 file exists: {os.path.exists(model1_path)}")
        
        # Check if weights file exists
        if not os.path.exists(model1_path):
            print(f"Model 1 weights not found at {model1_path}")
            return None
        
        # Initialize model
        model = ModifiedDeiT(num_classes=len(CLASS_NAMES_1))
        model.to(DEVICE)
        
        # Load weights
        checkpoint = torch.load(model1_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        
        _model1 = model
        print("Model 1 loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading Model 1: {e}")
        return None

def predict1(image_tensor):
    """
    Make prediction using Model 1
    
    Args:
        image_tensor: Preprocessed image tensor (should be on correct device)
        
    Returns:
        Tuple: (class_name, confidence_float, all_probs_list) or error dict
    """
    global _model1
    
    try:
        # Check if model is loaded
        if _model1 is None:
            return {"error": "Model 1 not ready. Add model1.pth to weights/ folder"}
        
        # Ensure model is in eval mode
        _model1.eval()
        
        # Move input to device if needed
        if image_tensor.device != torch.device(DEVICE):
            image_tensor = image_tensor.to(DEVICE)
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = _model1(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Convert to numpy for easier handling
            probs_np = probabilities.cpu().numpy()[0]
            predicted_idx = predicted.item()
            confidence_float = confidence.item()
            
            # Get class name
            class_name = CLASS_NAMES_1[predicted_idx]
            
            # Create list of all probabilities (class_name, probability) tuples
            all_probs_list = [(CLASS_NAMES_1[i], float(probs_np[i])) for i in range(len(CLASS_NAMES_1))]
            
            return (class_name, confidence_float, all_probs_list)
            
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Initialize model on module import
if __name__ != "__main__":
    load_model1()
