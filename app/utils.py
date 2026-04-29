import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import torch
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_image(file_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert('RGB')
        tensor = preprocess(image).unsqueeze(0)
        return tensor
    except Exception as e:
        raise ValueError(f"Image error: {str(e)}")
