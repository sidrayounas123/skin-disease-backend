from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Try to import ML modules, but handle gracefully if not available
try:
    from app.model1 import load_model1, predict1, CLASS_NAMES_1
    from app.model2 import load_model2, predict2, CLASS_NAMES_2
    from app.utils import preprocess_image
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML modules not available: {e}")
    ML_AVAILABLE = False

from app.disease_info import DISEASE_INFO
from app import firebase_service
from app import auth_service

app = FastAPI(title="Skin Disease Detection API", version="1.0.0")

# Pydantic models for auth requests
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    if ML_AVAILABLE:
        try:
            load_model1()
            load_model2()
            print("ML models loaded successfully")
        except Exception as e:
            print(f"Failed to load ML models: {e}")
    else:
        print("ML modules not available - running in API-only mode")

@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "message": "Backend is running"
    }

@app.get("/status")
async def status():
    """Detailed status of both models"""
    if not ML_AVAILABLE:
        return {
            "ml_available": False,
            "message": "ML modules not installed - running in API-only mode"
        }
    
    model1_ready = os.path.exists("weights/model1.pth")
    model2_ready = os.path.exists("weights/model2.pth")
    model2_classes_configured = len(CLASS_NAMES_2) > 0
    
    return {
        "ml_available": True,
        "model1": {
            "ready": model1_ready,
            "classes": CLASS_NAMES_1,
            "message": "Ready" if model1_ready else "Add model1.pth to weights/ folder"
        },
        "model2": {
            "ready": model2_ready and model2_classes_configured,
            "classes": CLASS_NAMES_2,
            "message": "Ready" if (model2_ready and model2_classes_configured) 
                      else "Add model2.pth to weights/ folder or update CLASS_NAMES_2 in model2.py"
        }
    }

@app.post("/predict/dataset1")
async def predict_dataset1(file: UploadFile = File(...), user_id: str = Query(None)):
    """Predict skin disease using Model 1 (Dataset 1)"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML modules not available - PyTorch not installed")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images (jpg, png) are allowed.")
        
        # Check if model is ready
        if not os.path.exists("weights/model1.pth"):
            raise HTTPException(status_code=503, detail="Model 1 not loaded yet")
        
        # Read and preprocess image
        file_bytes = await file.read()
        image_tensor = preprocess_image(file_bytes)
        
        # Make prediction
        result = predict1(image_tensor)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        class_name, confidence, all_probs = result
        
        # Get disease information
        disease_key = class_name
        info = DISEASE_INFO.get(disease_key, {
            "severity": "Unknown",
            "severity_score": 0,
            "description": "Please consult a doctor",
            "precautions": ["Consult a dermatologist"],
            "initial_treatment": ["See a doctor"],
            "see_doctor": True
        })
        
        # Save to Firebase if user_id is provided
        if user_id:
            firebase_service.save_scan(user_id, class_name, round(confidence * 100, 2), 
                                     info["severity"], info["see_doctor"], "dataset1")
        
        # Format response
        return {
            "model": "dataset1",
            "predicted_disease": class_name,
            "confidence_percent": round(confidence * 100, 2),
            "severity": info["severity"],
            "severity_score": info["severity_score"],
            "description": info["description"],
            "precautions": info["precautions"],
            "initial_treatment": info["initial_treatment"],
            "see_doctor": info["see_doctor"],
            "all_probabilities": {cls: round(prob * 100, 2) for cls, prob in all_probs}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/dataset2")
async def predict_dataset2(file: UploadFile = File(...), user_id: str = Query(None)):
    """Predict skin disease using Model 2 (Dataset 2)"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML modules not available - PyTorch not installed")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images (jpg, png) are allowed.")
        
        # Check if classes are configured
        if len(CLASS_NAMES_2) == 0:
            raise HTTPException(status_code=503, detail="Model 2 classes not configured yet. Update CLASS_NAMES_2 in model2.py")
        
        # Check if model is ready
        if not os.path.exists("weights/model2.pth"):
            raise HTTPException(status_code=503, detail="Model 2 not loaded yet")
        
        # Read and preprocess image
        file_bytes = await file.read()
        image_tensor = preprocess_image(file_bytes)
        
        # Make prediction
        result = predict2(image_tensor)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        class_name, confidence, all_probs = result
        
        # Get disease information
        disease_key = class_name
        info = DISEASE_INFO.get(disease_key, {
            "severity": "Unknown",
            "severity_score": 0,
            "description": "Please consult a doctor",
            "precautions": ["Consult a dermatologist"],
            "initial_treatment": ["See a doctor"],
            "see_doctor": True
        })
        
        # Save to Firebase if user_id is provided
        if user_id:
            firebase_service.save_scan(user_id, class_name, round(confidence * 100, 2), 
                                     info["severity"], info["see_doctor"], "dataset2")
        
        # Format response
        return {
            "model": "dataset2",
            "predicted_disease": class_name,
            "confidence_percent": round(confidence * 100, 2),
            "severity": info["severity"],
            "severity_score": info["severity_score"],
            "description": info["description"],
            "precautions": info["precautions"],
            "initial_treatment": info["initial_treatment"],
            "see_doctor": info["see_doctor"],
            "all_probabilities": {cls: round(prob * 100, 2) for cls, prob in all_probs}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/scans/{user_id}")
async def get_user_scans(user_id: str):
    """Get all scans for a specific user"""
    try:
        scans = firebase_service.get_scans(user_id)
        return {
            "user_id": user_id,
            "scans": scans,
            "total": len(scans)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scans: {str(e)}")

# Authentication endpoints
@app.post("/auth/register")
async def register_user(request: RegisterRequest):
    """Register a new user"""
    try:
        uid = auth_service.register_user(request.name, request.email, request.password)
        return {
            "message": "User created successfully",
            "uid": uid
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login_user(request: LoginRequest):
    """Login user and return token"""
    try:
        result = auth_service.login_user(request.email, request.password)
        # Get user profile data
        profile = auth_service.get_user_profile(result["uid"])
        return {
            "token": result["token"],
            "uid": result["uid"],
            "name": profile["name"]
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.get("/auth/profile/{uid}")
async def get_user_profile(uid: str):
    """Get user profile"""
    try:
        profile = auth_service.get_user_profile(uid)
        return profile
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
