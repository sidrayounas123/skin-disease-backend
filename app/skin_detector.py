import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

class SkinBinaryClassifier(nn.Module):
    """Binary classifier to detect skin disease images vs non-skin images"""
    
    def __init__(self):
        super(SkinBinaryClassifier, self).__init__()
        
        # Simple CNN for binary classification
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),  # 224x224 -> 14x14 after 4 max pools
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Binary output
            nn.Sigmoid()  # Sigmoid for binary classification
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Global model instance
_skin_detector = None

def load_skin_detector():
    """Load the skin detector model"""
    global _skin_detector
    try:
        _skin_detector = SkinBinaryClassifier()
        _skin_detector.eval()
        print("Skin detector model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading skin detector: {str(e)}")
        return False

def is_skin_image(image_tensor, threshold=0.5):
    """
    Determine if image is likely a skin disease image
    
    Args:
        image_tensor: Preprocessed image tensor
        threshold: Confidence threshold for skin detection
        
    Returns:
        bool: True if skin image, False if non-skin image
        float: Confidence score
    """
    global _skin_detector
    
    if _skin_detector is None:
        # Fallback to basic heuristics if model not loaded
        return heuristic_skin_check(image_tensor), 0.5
    
    try:
        with torch.no_grad():
            # Get prediction
            output = _skin_detector(image_tensor)
            confidence = output.item()
            
            # Determine if skin image
            is_skin = confidence >= threshold
            
            print(f"Skin detector confidence: {confidence:.3f}, threshold: {threshold}")
            print(f"Result: {'Skin image' if is_skin else 'Non-skin image'}")
            
            return is_skin, confidence
            
    except Exception as e:
        print(f"Error in skin detection: {str(e)}")
        # Fallback to heuristics
        return heuristic_skin_check(image_tensor), 0.5

def heuristic_skin_check(image_tensor):
    """
    Basic heuristic-based skin detection as fallback
    
    Args:
        image_tensor: Preprocessed image tensor
        
    Returns:
        bool: True if likely skin image
    """
    try:
        # Convert tensor back to numpy for analysis
        if image_tensor.dim() == 4:  # Batch dimension
            img_np = image_tensor[0].permute(1, 2, 0).numpy()
        else:
            img_np = image_tensor.permute(1, 2, 0).numpy()
        
        # Denormalize (reverse ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Skin color detection (simplified)
        # Skin typically has higher red and lower blue values
        red_channel = img_np[:, :, 0]
        green_channel = img_np[:, :, 1]
        blue_channel = img_np[:, :, 2]
        
        # Calculate skin-like pixels (simplified criteria)
        skin_like_mask = (
            (red_channel > 0.3) &  # Red channel should be relatively high
            (green_channel > 0.2) &  # Green channel moderate
            (blue_channel < 0.4) &  # Blue channel should be lower
            (red_channel > green_channel) &  # Red > Green
            (red_channel > blue_channel)  # Red > Blue
        )
        
        skin_pixel_ratio = np.sum(skin_like_mask) / skin_like_mask.size
        
        print(f"Heuristic skin detection: {skin_pixel_ratio:.3f} skin-like pixels")
        
        # If more than 15% of pixels are skin-like, consider it a skin image
        is_skin = skin_pixel_ratio > 0.15
        
        return is_skin
        
    except Exception as e:
        print(f"Error in heuristic skin detection: {str(e)}")
        return True  # Default to skin image on error

# Preprocessing for skin detector (same as main models)
skin_detector_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_for_skin_detection(image_bytes):
    """
    Preprocess image bytes for skin detection
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        tensor = skin_detector_preprocess(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        print(f"Error preprocessing for skin detection: {str(e)}")
        raise ValueError(f"Error preprocessing image: {e}")

# Initialize skin detector on module import
load_skin_detector()
