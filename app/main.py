from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ConfigDict, model_validator
from contextlib import asynccontextmanager
import os
import re
from typing import Any
from PIL import Image
import io
from app.model1 import load_model1, predict1, CLASS_NAMES_1
from app.model2 import load_model2, predict2, CLASS_NAMES_2
from app.utils import preprocess_image
from app.disease_info import DISEASE_INFO

# Try to import Firebase services, but don't fail if not available
try:
    from app import firebase_service
    FIREBASE_SERVICE_AVAILABLE = True
except Exception as e:
    print(f"Firebase service not available: {e}")
    FIREBASE_SERVICE_AVAILABLE = False

try:
    from app import auth_service
    AUTH_SERVICE_AVAILABLE = True
except Exception as e:
    print(f"Auth service not available: {e}")
    AUTH_SERVICE_AVAILABLE = False

# Load models on startup using modern FastAPI approach
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading models on startup...")
    try:
        load_model1()
        print("Model 1 loading completed")
    except Exception as e:
        print(f"Error loading Model 1: {str(e)}")
        print("Model 1 will be loaded on-demand")
    
    try:
        load_model2()
        print("Model 2 loading completed")
    except Exception as e:
        print(f"Error loading Model 2: {str(e)}")
        print("Model 2 will be loaded on-demand")
    
        
    print("FastAPI application startup completed")
    yield
    # Shutdown (if needed)
    print("Application shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Skin Disease Detection API", 
    version="1.0.0",
    lifespan=lifespan
)

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


@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "message": "Skin Disease Detection API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "message": "API is working correctly",
        "timestamp": "test",
        "status": "ok"
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
async def predict_dataset1(file: UploadFile = File(...), user_id: str = None):
    contents = await file.read()
    
    if len(contents) == 0:
        raise HTTPException(400, "Empty file received")
    
    try:
        image_tensor = preprocess_image(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    try:
        disease, confidence, all_probs = predict1(image_tensor)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")
    
    confidence_percent = round(float(confidence) * 100, 2)
    
    # Check if confidence is less than 40% for irrelevant image detection
    if float(confidence) < 0.4:
        return {
            "success": False,
            "predicted_disease": "Unknown",
            "message": "Image does not appear to be a skin condition. Please upload a clear skin image.",
            "confidence_percent": confidence_percent,
            "is_valid_skin_image": False
        }
    
    info = DISEASE_INFO.get(disease, {
        "severity": "Unknown",
        "severity_score": 0,
        "description": "Please consult a doctor",
        "precautions": ["Consult a dermatologist"],
        "initial_treatment": ["See a doctor"],
        "see_doctor": True
    })
    
    if user_id and FIREBASE_SERVICE_AVAILABLE:
        try:
            firebase_service.save_scan(
                user_id, disease, confidence,
                info["severity"], info["see_doctor"], "dataset1"
            )
        except Exception as e:
            print(f"Firebase save error: {e}")
    
    return {
        "success": True,
        "predicted_disease": disease,
        "confidence_percent": confidence_percent,
        "severity": info["severity"],
        "description": info["description"],
        "precautions": info["precautions"],
        "initial_treatment": info["initial_treatment"],
        "is_valid_skin_image": True
    }

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
        
        # Accept any image format - no content_type validation
        print("Dataset2 - Accepting any image format")
        print("Dataset2 - Prediction started")
        
        # Check if classes are configured
        if len(CLASS_NAMES_2) == 0:
            raise HTTPException(status_code=503, detail="Model 2 classes not configured yet. Update CLASS_NAMES_2 in model2.py")
        
        # Check if model is ready (models loaded globally at startup)
        from app.model2 import _model2, load_model2
        if _model2 is None:
            print("Model 2 not loaded, attempting on-demand loading...")
            try:
                load_model2()
                if _model2 is None:
                    raise HTTPException(status_code=503, detail="Model 2 failed to load. Please check the weights file and configuration.")
                print("Model 2 loaded successfully on-demand")
            except Exception as e:
                print(f"On-demand Model 2 loading failed: {str(e)}")
                raise HTTPException(status_code=503, detail=f"Model 2 loading failed: {str(e)}")
        
        # Read file with proper error handling
        contents = await file.read()
        print(f"Received {len(contents)} bytes")
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        file_bytes = contents
        
        # STEP D: Preprocess and run disease model prediction
        try:
            image_tensor = preprocess_image(file_bytes)
        except ValueError as e:
            print(f"Dataset2 - Image processing error: {str(e)}")
            return {
                "success": False,
                "message": "Cannot decode image file. Please upload a valid image."
            }
        except Exception as e:
            print(f"Dataset2 - Error preprocessing image: {str(e)}")
            return {
                "success": False,
                "message": "Image preprocessing failed. Please try a different image."
            }
        
        print("Dataset2 - Performing disease classification...")
        result = predict2(image_tensor)
        
        print("Dataset2 - Prediction finished")
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        class_name, confidence, all_probs = result
        
        # STEP E: Get top predicted class confidence and debug logging
        print(f"Dataset2 - DEBUG LOGS:")
        print(f"  Filename: {file.filename}")
        print(f"  Image size: {Image.open(io.BytesIO(file_bytes)).size}")
        print(f"  Preprocessed shape: {image_tensor.shape}")
        
        # Log top 3 class probabilities
        top_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)[:3]
        print(f"  Top 3 probabilities:")
        for i, idx in enumerate(top_indices):
            prob_percent = all_probs[idx] * 100
            class_name_top = CLASS_NAMES_2[idx] if idx < len(CLASS_NAMES_2) else f"Class_{idx}"
            print(f"    {i+1}. {class_name_top}: {prob_percent:.2f}% (index: {idx})")
        
        print(f"  Final predicted class: {class_name}")
        print(f"  Top confidence: {float(confidence):.2f}%")
        
        prediction_time = round((time.time() - start_time) * 1000, 2)
        print(f"  Response time: {prediction_time}ms")
        
        # STEP F: Relevance logic - confidence-based rejection only
        MIN_CONFIDENCE = 40.0  # Minimum 40% confidence required
        
        confidence_percent = round(float(confidence) * 100, 2)
        
        if confidence_percent < 40:
            print(f"Dataset2 - Low confidence ({confidence_percent:.2f}%) - rejecting as irrelevant image")
            return {
                "success": False,
                "predicted_disease": "Unknown",
                "message": "Image does not appear to be a skin condition. Please upload a clear skin image.",
                "confidence_percent": confidence_percent,
                "is_valid_skin_image": False
            }
        
        print(f"Dataset2 - Confidence acceptable ({confidence_percent:.2f}%) - proceeding with result")
        
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
        if user_id and FIREBASE_SERVICE_AVAILABLE:
            firebase_service.save_scan(user_id, class_name, round(confidence * 100, 2), 
                                     info["severity"], info["see_doctor"], "dataset2")
        
        # Calculate total processing time
        processing_time = round((time.time() - start_time) * 1000, 2)
        print(f"Dataset2 - Response sent in {processing_time}ms")
        
        # Format standardized response matching dataset1 format
        return {
            "success": True,
            "predicted_disease": class_name,
            "confidence_percent": confidence_percent,
            "severity": info["severity"],
            "description": info["description"],
            "precautions": info["precautions"],
            "initial_treatment": info["initial_treatment"],
            "is_valid_skin_image": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/scans/{user_id}")
async def get_user_scans(user_id: str, filter: str = Query(None, description="Filter scans: high_risk, low_risk")):
    """Get detailed scan history for a specific user"""
    if not FIREBASE_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail={"success": False, "message": "Firebase service is not available"}
        )
    
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
    if not AUTH_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail={"success": False, "message": "Authentication service is not available"}
        )
    
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
    if not AUTH_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail={"success": False, "message": "Authentication service is not available"}
        )
    
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
    if not AUTH_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail={"success": False, "message": "Authentication service is not available"}
        )
    
    try:
        profile = auth_service.get_user_profile(uid)
        return profile
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
