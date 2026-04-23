from flask import Flask, jsonify, request
import os

# Import Firebase services
try:
    from app.firebase_service import save_scan, get_scans
    from app.auth_service import register_user, login_user, get_user_profile
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def health_check():
    """Simple health check that responds immediately"""
    return jsonify({
        "message": "Backend is running",
        "status": "healthy",
        "service": "skin-disease-backend"
    })

@app.route('/status')
def status():
    """Status endpoint"""
    return jsonify({
        "status": "running",
        "mode": "api-only",
        "firebase_available": FIREBASE_AVAILABLE,
        "message": "Backend ready for testing"
    })

# Authentication endpoints
@app.route('/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not all([name, email, password]):
            return jsonify({"error": "Missing required fields"}), 400
        
        result = register_user(name, email, password)
        return jsonify(result), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({"error": "Missing email or password"}), 400
        
        result = login_user(email, password)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auth/profile/<uid>', methods=['GET'])
def get_profile(uid):
    """Get user profile endpoint"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        result = get_user_profile(uid)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Scan history endpoint
@app.route('/scans/<user_id>', methods=['GET'])
def get_user_scans(user_id):
    """Get user scan history"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        scans = get_scans(user_id)
        return jsonify({"scans": scans}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Prediction endpoints (placeholder for now)
@app.route('/predict/dataset1', methods=['POST'])
def predict_dataset1():
    """Predict skin disease using Model 1 (Dataset 1)"""
    return jsonify({
        "message": "ML predictions coming soon!",
        "status": "placeholder",
        "note": "Full ML functionality will be added after basic deployment works"
    }), 200

@app.route('/predict/dataset2', methods=['POST'])
def predict_dataset2():
    """Predict skin disease using Model 2 (Dataset 2)"""
    return jsonify({
        "message": "ML predictions coming soon!",
        "status": "placeholder", 
        "note": "Full ML functionality will be added after basic deployment works"
    }), 200

if __name__ == '__main__':
    # Start Flask app
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
