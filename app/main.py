from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ConfigDict, model_validator
import os
import re
from typing import Any
from app.model1 import load_model1, predict1, CLASS_NAMES_1
from app.model2 import load_model2, predict2, CLASS_NAMES_2
from app.utils import preprocess_image
from app.disease_info import DISEASE_INFO
from app import firebase_service
from app import auth_service

app = FastAPI(title="Skin Disease Detection API", version="1.0.0")

# Pydantic models for auth requests
class RegisterRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    # Support all name field formats
    name: str = Field(min_length=1, description="User full name")
    full_name: str = Field(default=None, description="User full name (alternative)")
    fullName: str = Field(default=None, description="User full name (alternative)")
    email: EmailStr = Field(description="Valid email address")
    password: str = Field(min_length=6, description="Password must be at least 6 characters")
    
    @model_validator(mode='before')
    @classmethod
    def normalize_name_fields(cls, data: dict | Any) -> dict | Any:
        if isinstance(data, dict):
            # Normalize all name fields to 'name'
            normalized_name = None
            
            if 'full_name' in data and data['full_name']:
                normalized_name = data.pop('full_name')
            elif 'fullName' in data and data['fullName']:
                normalized_name = data.pop('fullName')
            elif 'name' in data and data['name']:
                normalized_name = data['name']
            
            if normalized_name:
                data['name'] = normalized_name
            else:
                raise ValueError("Name field is required (full_name, fullName, or name)")
                
            # Clear any remaining name fields to avoid conflicts
            data.pop('full_name', None)
            data.pop('fullName', None)
            
        return data
    
    def get_normalized_name(self) -> str:
        """Get the normalized name value safely"""
        return self.name

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
    load_model1()
    load_model2()

@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "message": "Backend is running",
        "status": "healthy"
    }

@app.get("/status")
async def status():
    """Detailed status of both models"""
    current_dir = os.getcwd()
    model1_path = os.path.join(current_dir, "weights", "model1.pth")
    model2_path = os.path.join(current_dir, "weights", "model2.pth")
    
    model1_ready = os.path.exists(model1_path)
    model2_ready = os.path.exists(model2_path)
    model2_classes_configured = len(CLASS_NAMES_2) > 0
    
    return {
        "working_directory": current_dir,
        "model1": {
            "path": model1_path,
            "ready": model1_ready,
            "classes": CLASS_NAMES_1,
            "message": "Ready" if model1_ready else f"Model 1 not found at {model1_path}"
        },
        "model2": {
            "path": model2_path,
            "ready": model2_ready and model2_classes_configured,
            "classes": CLASS_NAMES_2,
            "message": "Ready" if (model2_ready and model2_classes_configured) 
                      else f"Model 2 not found at {model2_path} or classes not configured"
        }
    }

@app.post("/predict/dataset1")
async def predict_dataset1(file: UploadFile = File(...), user_id: str = Query(None)):
    """Predict skin disease using Model 1 (Dataset 1)"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images (jpg, png) are allowed.")
        
        # Check if model is ready
        current_dir = os.getcwd()
        model1_path = os.path.join(current_dir, "weights", "model1.pth")
        if not os.path.exists(model1_path):
            raise HTTPException(status_code=503, detail=f"Model 1 not available at {model1_path}. Please upload model1.pth to the weights/ directory.")
        
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
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images (jpg, png) are allowed.")
        
        # Check if classes are configured
        if len(CLASS_NAMES_2) == 0:
            raise HTTPException(status_code=503, detail="Model 2 classes not configured yet. Update CLASS_NAMES_2 in model2.py")
        
        # Check if model is ready
        current_dir = os.getcwd()
        model2_path = os.path.join(current_dir, "weights", "model2.pth")
        if not os.path.exists(model2_path):
            raise HTTPException(status_code=503, detail=f"Model 2 not available at {model2_path}. Please upload model2.pth to the weights/ directory.")
        
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
        # Print incoming request body
        print(f"Registration request received: {request.model_dump()}")
        
        # Get normalized name safely
        full_name = request.get_normalized_name()
        print(f"Normalized name: {full_name}")
        
        # Check if user already exists
        existing_users = auth_service.get_user_by_email(request.email)
        if existing_users:
            print(f"User already exists with email: {request.email}")
            raise HTTPException(
                status_code=409, 
                detail={"success": False, "message": "User already exists"}
            )
        
        # Register user with Firebase
        print(f"Creating new user: {full_name}, {request.email}")
        result = auth_service.register_user(full_name, request.email, request.password)
        
        if "error" in result:
            print(f"Registration failed: {result['error']}")
            raise HTTPException(
                status_code=400, 
                detail={"success": False, "message": result["error"]}
            )
        
        print(f"User created successfully: {result.get('uid')}")
        
        return {
            "success": True,
            "message": "Account created successfully",
            "user": {
                "name": full_name,
                "email": request.email
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail={"success": False, "message": f"Registration failed: {str(e)}"}
        )

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
