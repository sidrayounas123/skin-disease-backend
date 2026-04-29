import io
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_image(file_bytes: bytes):
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 failed")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
    except:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Cannot open image: {str(e)}")
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = preprocess(image).unsqueeze(0)
    return tensor
