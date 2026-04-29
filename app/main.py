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
    model_config = ConfigDict(extra='allow')
    
    # Make all name fields optional to avoid attribute errors
    name: str | None = None
    full_name: str | None = None
    fullName: str | None = None
    email: EmailStr = Field(description="Valid email address")
    password: str = Field(min_length=6, description="Password must be at least 6 characters")
    
    def get_normalized_name(self) -> str:
        """Get the normalized name value safely"""
        if self.full_name:
            return self.full_name
        elif self.fullName:
            return self.fullName
        elif self.name:
            return self.name
        else:
            raise ValueError("Name field is required (name, full_name, or fullName)")
    
    @model_validator(mode='after')
    def validate_name_field(self) -> 'RegisterRequest':
        """Ensure at least one name field is provided"""
        normalized_name = self.get_normalized_name()
        # Store normalized name in the primary 'name' field
        self.name = normalized_name
        return self

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
        import time
        start_time = time.time()
        
        print(f"Dataset1 - Request received for file: {file.filename}")
        Audit my skin disease backend model API. Currently every uploaded image gets classified into one disease class even if image is irrelevant (clothes, objects, random backgrounds). Add irrelevant image detection and reject non-skin images with response:
{
 "success": false,
 "message": "Irrelevant image. Please upload a clear skin disease image."
}
Use confidence threshold and image validation.
        # Debug logging
        filename = file.filename.lower().strip() if file.filename else ''
        print(f"Dataset1 - filename: {filename}")
        print(f"Dataset1 - content_type: {file.content_type}")
        
        # Validate using filename extension only
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Check file extension (case-insensitive)
        if filename:
            file_extension = '.' + filename.split('.')[-1] if '.' in filename else ''
            if file_extension not in allowed_extensions:
                print(f"Dataset1 - Invalid file extension: {file_extension}")
                raise HTTPException(status_code=400, detail="Invalid file type. Only images (jpg, jpeg, png) are allowed.")
        else:
            raise HTTPException(status_code=400, detail="No filename provided. Only images (jpg, jpeg, png) are allowed.")
        
        print("Dataset1 - File validation passed - proceeding with prediction")
        print("Dataset1 - Prediction started")
        
        # Check if model is ready (models loaded globally at startup)
        from app.model1 import _model1
        if _model1 is None:
            raise HTTPException(status_code=503, detail="Model 1 not loaded. Please restart the application.")
        
        # Read and preprocess image (optimized to 224x224)
        file_bytes = await file.read()
        image_tensor = preprocess_image(file_bytes)
        
        # Make prediction (model already loaded)
        result = predict1(image_tensor)
        
        print("Dataset1 - Prediction finished")
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        class_name, confidence, all_probs = result
        
        # Validate confidence threshold and image relevance
        MIN_CONFIDENCE = 30.0  # Minimum 30% confidence required
        
        if confidence < MIN_CONFIDENCE:
            print(f"Dataset1 - Low confidence ({confidence}%) - rejecting as irrelevant image")
            return {
                "success": False,
                "message": "Irrelevant image. Please upload a clear skin disease image with confidence above 30%."
            }
        
        # Basic image content analysis (simple heuristic)
        try:
            # Read image for basic analysis
            from PIL import Image
            image = Image.open(io.BytesIO(file_bytes))
            
            # Check if image is likely skin-related
            # This is a basic heuristic - can be enhanced with proper ML models
            is_likely_skin = True  # Default to True for now
            
            # Simple checks for obviously non-skin content
            # (This can be expanded with more sophisticated analysis)
            if image.mode not in ['RGB', 'RGBA']:
                is_likely_skin = False
                print(f"Dataset1 - Non-RGB image mode: {image.mode}")
            
            # Size check - very small or very large images might be irrelevant
            width, height = image.size
            if width < 100 or height < 100 or width > 2000 or height > 2000:
                is_likely_skin = False
                print(f"Dataset1 - Suspicious image size: {width}x{height}")
            
            image.close()
            
            if not is_likely_skin:
                print(f"Dataset1 - Image appears to be non-skin related")
                return {
                    "success": False,
                    "message": "Irrelevant image. Please upload a clear skin disease image."
                }
                
        except Exception as e:
            print(f"Dataset1 - Error during image analysis: {str(e)}")
            # Continue with prediction even if analysis fails
        
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
        
        # Save to Firebase if user_id is provided (async operation)
        if user_id:
            firebase_service.save_scan(user_id, class_name, round(confidence * 100, 2), 
                                     info["severity"], info["see_doctor"], "dataset1")
        
        # Calculate total processing time
        processing_time = round((time.time() - start_time) * 1000, 2)
        print(f"Dataset1 - Response sent in {processing_time}ms")
        
        # Format response (immediate return)
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
            "all_probabilities": {cls: round(prob * 100, 2) for cls, prob in all_probs},
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/dataset2")
async def predict_dataset2(file: UploadFile = File(...), user_id: str = Query(None)):
    """Predict skin disease using Model 2 (Dataset 2)"""
    try:
        import time
        start_time = time.time()
        
        print(f"Dataset2 - Request received for file: {file.filename}")
        
        # Debug logging
        filename = file.filename.lower().strip() if file.filename else ''
        print(f"Dataset2 - filename: {filename}")
        print(f"Dataset2 - content_type: {file.content_type}")
        
        # Validate using filename extension only
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Check file extension (case-insensitive)
        if filename:
            file_extension = '.' + filename.split('.')[-1] if '.' in filename else ''
            if file_extension not in allowed_extensions:
                print(f"Dataset2 - Invalid file extension: {file_extension}")
                raise HTTPException(status_code=400, detail="Invalid file type. Only images (jpg, jpeg, png) are allowed.")
        else:
            raise HTTPException(status_code=400, detail="No filename provided. Only images (jpg, jpeg, png) are allowed.")
        
        print("Dataset2 - File validation passed - proceeding with prediction")
        print("Dataset2 - Prediction started")
        
        # Check if classes are configured
        if len(CLASS_NAMES_2) == 0:
            raise HTTPException(status_code=503, detail="Model 2 classes not configured yet. Update CLASS_NAMES_2 in model2.py")
        
        # Check if model is ready (models loaded globally at startup)
        from app.model2 import _model2
        if _model2 is None:
            raise HTTPException(status_code=503, detail="Model 2 not loaded. Please restart the application.")
        
        # Read and preprocess image (optimized to 224x224)
        file_bytes = await file.read()
        image_tensor = preprocess_image(file_bytes)
        
        # Make prediction (model already loaded)
        result = predict2(image_tensor)
        
        print("Dataset2 - Prediction finished")
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        class_name, confidence, all_probs = result
        
        # Validate confidence threshold and image relevance
        MIN_CONFIDENCE = 30.0  # Minimum 30% confidence required
        
        if confidence < MIN_CONFIDENCE:
            print(f"Dataset2 - Low confidence ({confidence}%) - rejecting as irrelevant image")
            return {
                "success": False,
                "message": "Irrelevant image. Please upload a clear skin disease image with confidence above 30%."
            }
        
        # Basic image content analysis (simple heuristic)
        try:
            # Read image for basic analysis
            from PIL import Image
            image = Image.open(io.BytesIO(file_bytes))
            
            # Check if image is likely skin-related
            # This is a basic heuristic - can be enhanced with proper ML models
            is_likely_skin = True  # Default to True for now
            
            # Simple checks for obviously non-skin content
            # (This can be expanded with more sophisticated analysis)
            if image.mode not in ['RGB', 'RGBA']:
                is_likely_skin = False
                print(f"Dataset2 - Non-RGB image mode: {image.mode}")
            
            # Size check - very small or very large images might be irrelevant
            width, height = image.size
            if width < 100 or height < 100 or width > 2000 or height > 2000:
                is_likely_skin = False
                print(f"Dataset2 - Suspicious image size: {width}x{height}")
            
            image.close()
            
            if not is_likely_skin:
                print(f"Dataset2 - Image appears to be non-skin related")
                return {
                    "success": False,
                    "message": "Irrelevant image. Please upload a clear skin disease image."
                }
                
        except Exception as e:
            print(f"Dataset2 - Error during image analysis: {str(e)}")
            # Continue with prediction even if analysis fails
        
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
        
        # Save to Firebase if user_id is provided (async operation)
        if user_id:
            firebase_service.save_scan(user_id, class_name, round(confidence * 100, 2), 
                                     info["severity"], info["see_doctor"], "dataset2")
        
        # Calculate total processing time
        processing_time = round((time.time() - start_time) * 1000, 2)
        print(f"Dataset2 - Response sent in {processing_time}ms")
        
        # Format response (immediate return)
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
            "all_probabilities": {cls: round(prob * 100, 2) for cls, prob in all_probs},
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/scans/{user_id}")
async def get_user_scans(user_id: str, filter: str = Query(None, description="Filter scans: high_risk, low_risk")):
    """Get detailed scan history for a specific user"""
    try:
        print(f"Retrieving scans for user: {user_id}, filter: {filter}")
        
        # Get scans with optional filtering
        scan_data = firebase_service.get_scans(user_id, filter_type=filter)
        
        # Format scan data for Flutter
        formatted_scans = []
        for scan in scan_data["scans"]:
            formatted_scan = {
                "id": scan.get("id", ""),
                "disease": scan.get("disease", "Unknown"),
                "confidence": float(scan.get("confidence", 0)),
                "severity": scan.get("severity", "Unknown"),
                "see_doctor": bool(scan.get("see_doctor", False)),
                "is_high_risk": bool(scan.get("is_high_risk", False)),
                "dataset": scan.get("dataset", "unknown"),
                "timestamp": scan.get("timestamp", "")
            }
            formatted_scans.append(formatted_scan)
        
        print(f"Returning {len(formatted_scans)} scans for user {user_id}")
        
        return {
            "total": scan_data["total"],
            "this_month": scan_data["this_month"],
            "this_week": scan_data["this_week"],
            "scans": formatted_scans
        }
        
    except Exception as e:
        print(f"Error retrieving scans: {str(e)}")
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
