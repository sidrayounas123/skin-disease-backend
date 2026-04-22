"""
Railway deployment script - optimized for smaller deployment size
"""
import os
import sys
import subprocess

def install_pytorch():
    """Install PyTorch after deployment to reduce Docker image size"""
    print("Installing PyTorch (CPU version)...")
    try:
        subprocess.run([
            "pip", "install", 
            "torch==2.0.1+cpu", 
            "torchvision==0.15.2+cpu",
            "timm==0.9.7",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)
        print("PyTorch installed successfully!")
    except subprocess.CalledProcessError:
        print("Failed to install PyTorch")

def download_models():
    """Download models from external source after deployment"""
    print("Downloading ML models...")
    
    # Create weights directory
    os.makedirs("weights", exist_ok=True)
    
    # Model URLs (you should host these files somewhere accessible)
    model_urls = {
        "model1.pth": "https://drive.google.com/uc?export=download&id=1oLDvfGVCI2ijYGbTRnMS2ATMGTKdpZoL",
        "model2.pth": "https://drive.google.com/uc?export=download&id=1mIrE7SoQDG9DRUWkaUNNvMfk4BExmzpV"
    }
    
    # Download models if not present
    for model_name, url in model_urls.items():
        if not os.path.exists(f"weights/{model_name}"):
            print(f"Downloading {model_name}...")
            try:
                subprocess.run(["wget", "-O", f"weights/{model_name}", url], check=True)
            except:
                print(f"Failed to download {model_name}. Please upload manually.")

if __name__ == "__main__":
    print("Starting Railway deployment setup...")
    
    # Install PyTorch first
    install_pytorch()
    
    # Download models
    download_models()
    
    print("Setup complete! Starting FastAPI server...")
    
    # Start FastAPI server
    os.system("python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT")
