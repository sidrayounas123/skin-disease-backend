import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
import json

# Initialize Firebase
if os.environ.get("FIREBASE_CREDENTIALS"):
    # Use environment variable for deployment
    firebase_creds = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(firebase_creds)
else:
    # No Firebase credentials available - skip Firebase initialization
    print("Firebase credentials not found. Firebase features will be disabled.")
    cred = None

if cred and not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Initialize database only if Firebase is available
db = firestore.client() if cred else None

def save_scan(user_id, disease, confidence, severity, see_doctor, dataset):
    if not db:
        print("Firebase not available - scan not saved")
        return False
    
    doc = {
        "user_id": user_id,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "see_doctor": see_doctor,
        "dataset": dataset,
        "timestamp": datetime.datetime.now()
    }
    db.collection("scans").add(doc)
    return True

def get_scans(user_id, filter_type=None):
    if not db:
        print("Firebase not available - returning empty scan history")
        return {"scans": [], "total": 0, "this_month": 0, "this_week": 0}
    
    # Get all scans for user
    scans = db.collection("scans")\
              .where("user_id", "==", user_id)\
              .order_by("timestamp", direction=firestore.Query.DESCENDING)\
              .stream()
    
    result = []
    now = datetime.datetime.now()
    one_week_ago = now - datetime.timedelta(days=7)
    current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    this_month_count = 0
    this_week_count = 0
    
    for scan in scans:
        data = scan.to_dict()
        data["id"] = scan.id
        
        # Convert timestamp to ISO format
        if "timestamp" in data:
            timestamp = data["timestamp"]
            if isinstance(timestamp, datetime.datetime):
                data["timestamp"] = timestamp.isoformat()
                
                # Calculate statistics
                if timestamp >= current_month_start:
                    this_month_count += 1
                if timestamp >= one_week_ago:
                    this_week_count += 1
        
        # Determine if high risk
        severity = data.get("severity", "").lower()
        see_doctor = data.get("see_doctor", False)
        is_high_risk = severity == "severe" or see_doctor
        data["is_high_risk"] = is_high_risk
        
        # Apply filter if specified
        if filter_type == "high_risk" and not is_high_risk:
            continue
        elif filter_type == "low_risk" and is_high_risk:
            continue
        
        result.append(data)
    
    return {
        "scans": result,
        "total": len(result),
        "this_month": this_month_count if not filter_type else sum(1 for scan in result if scan.get("timestamp")),
        "this_week": this_week_count if not filter_type else sum(1 for scan in result if scan.get("timestamp"))
    }
