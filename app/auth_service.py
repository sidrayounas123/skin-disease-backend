import firebase_admin
from firebase_admin import credentials, firestore, auth
import requests
import datetime

# Firebase Web API Key
FIREBASE_WEB_API_KEY = "AIzaSyCqC7fkcWhLLdMlb080P7vQHwe3t-1YSSs"

# Initialize Firebase (using same initialization from firebase_service)
try:
    db = firestore.client()
    FIREBASE_AVAILABLE = True
except:
    print("Firebase not available - authentication features disabled")
    db = None
    FIREBASE_AVAILABLE = False

def register_user(name, email, password):
    """
    Creates user in Firebase Auth and saves name+email to Firestore
    Returns user uid on success
    """
    if not FIREBASE_AVAILABLE:
        return {"error": "Firebase authentication not configured. Please add FIREBASE_CREDENTIALS environment variable in Hugging Face Space settings."}
    
    try:
        # Create user in Firebase Auth
        user = auth.create_user(
            email=email,
            password=password
        )
        
        # Save user info to Firestore
        user_doc = {
            "name": name,
            "email": email,
            "created_at": datetime.datetime.now()
        }
        db.collection("users").document(user.uid).set(user_doc)
        
        return {"uid": user.uid, "message": "User created successfully"}
        
    except Exception as e:
        return {"error": str(e)}

def login_user(email, password):
    """
    Verifies user credentials using Firebase Auth REST API
    Returns user token and uid on success
    """
    if not FIREBASE_AVAILABLE:
        return {"error": "Firebase authentication not configured. Please add FIREBASE_CREDENTIALS environment variable in Hugging Face Space settings."}
    
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
        
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "token": data["idToken"],
                "uid": data["localId"]
            }
        else:
            error_data = response.json()
            raise Exception(f"Login failed: {error_data.get('error', {}).get('message', 'Unknown error')}")
            
    except Exception as e:
        raise Exception(f"Login failed: {str(e)}")

def get_user_profile(uid):
    """
    Gets user data from Firestore "users" collection
    Returns {name, email}
    """
    if not FIREBASE_AVAILABLE:
        raise Exception("Firebase authentication not configured. Please add FIREBASE_CREDENTIALS environment variable in Hugging Face Space settings.")
    
    try:
        doc_ref = db.collection("users").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            return {
                "name": data["name"],
                "email": data["email"]
            }
        else:
            raise Exception("User not found")
            
    except Exception as e:
        raise Exception(f"Failed to get user profile: {str(e)}")
