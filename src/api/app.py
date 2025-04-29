from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local imports
from src.evaluation.strength_evaluator import PasswordStrengthEvaluator
from src.simulator.password_cracker import PasswordCrackerSimulator

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dictionary for common passwords
dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         "data", "rockyou.txt")

# Initialize services
common_passwords = set()
if os.path.exists(dict_path):
    with open(dict_path, 'r', encoding='latin-1', errors='ignore') as f:
        # Load just the top passwords for efficiency
        for i, line in enumerate(f):
            if i >= 10000:  # Limit to top 10,000 passwords
                break
            common_passwords.add(line.strip())
    print(f"Loaded {len(common_passwords)} common passwords")
else:
    print(f"Warning: Dictionary file not found at {dict_path}")
    print("Proceeding without common passwords list")

# Initialize evaluator
password_evaluator = PasswordStrengthEvaluator(common_passwords=common_passwords)

# Initialize cracker simulator (if dictionary is available)
password_cracker = None
if os.path.exists(dict_path):
    password_cracker = PasswordCrackerSimulator(dictionary_path=dict_path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Password Strength Evaluator API is running'
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate_password():
    """Evaluate password strength"""
    data = request.get_json()
    
    if not data or 'password' not in data:
        return jsonify({
            'error': 'Invalid request. Password is required.'
        }), 400
    
    password = data['password']
    
    # Evaluate password
    result = password_evaluator.evaluate(password)
    
    return jsonify(result)

@app.route('/api/simulate', methods=['POST'])
def simulate_attacks():
    """Simulate password cracking attacks"""
    data = request.get_json()
    
    if not data or 'password' not in data:
        return jsonify({
            'error': 'Invalid request. Password is required.'
        }), 400
    
    password = data['password']
    max_attempts = data.get('max_attempts', 100000)
    
    if not password_cracker:
        return jsonify({
            'error': 'Password cracker simulator not available. Dictionary file missing.'
        }), 500
    
    # Run attack simulations
    results = password_cracker.run_all_attacks(
        password, 
        max_attempts_per_attack=max_attempts,
        verbose=False
    )
    
    # Clean up results for JSON serialization
    for attack_type, attack_result in results['attacks'].items():
        if 'error' not in attack_result:
            # Convert numpy types to native types for JSON serialization
            for key, value in attack_result.items():
                if hasattr(value, 'item'):  # Check if it's a numpy type
                    attack_result[key] = value.item()
    
    return jsonify(results)

@app.route('/api/crack-time', methods=['POST'])
def crack_time():
    """Get estimated crack time for a password"""
    data = request.get_json()
    
    if not data or 'password' not in data:
        return jsonify({
            'error': 'Invalid request. Password is required.'
        }), 400
    
    password = data['password']
    attack_type = data.get('attack_type', 'offline_fast')
    
    # Calculate crack time
    seconds = password_evaluator.estimate_crack_time(password, attack_type)
    formatted = password_evaluator.format_time(seconds)
    
    return jsonify({
        'seconds': seconds,
        'formatted': formatted,
        'attack_type': attack_type
    })

@app.route('/api/improve', methods=['POST'])
def improve_password():
    """Get recommendations to improve a password"""
    data = request.get_json()
    
    if not data or 'password' not in data:
        return jsonify({
            'error': 'Invalid request. Password is required.'
        }), 400
    
    password = data['password']
    
    # Evaluate password
    result = password_evaluator.evaluate(password)
    
    # Return just the improvements
    return jsonify({
        'strength_score': result['strength_score'],
        'strength_category': result['strength_category'],
        'improvements': result['improvements']
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 