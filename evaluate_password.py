import os
import pickle
import numpy as np
from src.models.transformer_model import PasswordTransformer
from src.models.lstm_model import PasswordLSTM

def load_vocabulary(models_dir):
    """Load vocabulary and character mappings"""
    vocab_path = os.path.join(models_dir, "vocab.pkl")
    char_to_idx_path = os.path.join(models_dir, "char_to_idx.pkl")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(char_to_idx_path, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    return vocab, char_to_idx

def tokenize_password(password, char_to_idx, max_length=30):
    """Convert password to token sequence"""
    tokens = [char_to_idx[c] for c in password if c in char_to_idx]
    if len(tokens) < max_length:
        tokens = tokens + [0] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
    return np.array(tokens)

def analyze_password_characteristics(password):
    """Analyze password characteristics and return detailed metrics"""
    analysis = {
        'length': len(password),
        'has_uppercase': any(c.isupper() for c in password),
        'has_lowercase': any(c.islower() for c in password),
        'has_digits': any(c.isdigit() for c in password),
        'has_special': any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password),
        'character_types': 0,
        'sequential_chars': False,
        'repeated_chars': False,
        'common_patterns': [],
        'entropy': 0
    }
    
    # Count character types
    analysis['character_types'] = sum([
        analysis['has_uppercase'],
        analysis['has_lowercase'],
        analysis['has_digits'],
        analysis['has_special']
    ])
    
    # Check for sequential characters
    for i in range(len(password)-3):
        if (password[i:i+4] in '0123456789' or 
            password[i:i+4] in 'abcdefghijklmnopqrstuvwxyz' or
            password[i:i+4] in 'qwertyuiopasdfghjklzxcvbnm'):
            analysis['sequential_chars'] = True
            break
    
    # Check for repeated characters
    for i in range(len(password)-2):
        if password[i] == password[i+1] == password[i+2]:
            analysis['repeated_chars'] = True
            break
    
    # Check for common patterns
    common_patterns = [
        'password', '123456', 'qwerty', 'admin', 'welcome',
        'letmein', 'monkey', 'dragon', 'baseball', 'football'
    ]
    for pattern in common_patterns:
        if pattern in password.lower():
            analysis['common_patterns'].append(pattern)
    
    # Calculate entropy
    char_set_size = 0
    if analysis['has_uppercase']: char_set_size += 26
    if analysis['has_lowercase']: char_set_size += 26
    if analysis['has_digits']: char_set_size += 10
    if analysis['has_special']: char_set_size += 32
    
    if char_set_size > 0:
        analysis['entropy'] = len(password) * np.log2(char_set_size)
    
    return analysis

def generate_recommendations(analysis, model_scores):
    """Generate detailed recommendations based on password analysis and model scores"""
    recommendations = []
    
    # Length recommendations
    if analysis['length'] < 12:
        recommendations.append({
            'type': 'critical',
            'message': f"Password is too short ({analysis['length']} characters). Use at least 12 characters for better security."
        })
    elif analysis['length'] < 16:
        recommendations.append({
            'type': 'warning',
            'message': "Consider increasing password length to 16+ characters for optimal security."
        })
    
    # Character type recommendations
    if analysis['character_types'] < 3:
        recommendations.append({
            'type': 'critical',
            'message': f"Password uses only {analysis['character_types']} character types. Use a mix of uppercase, lowercase, numbers, and special characters."
        })
    elif analysis['character_types'] < 4:
        recommendations.append({
            'type': 'warning',
            'message': "Consider adding special characters to increase password strength."
        })
    
    # Pattern recommendations
    if analysis['sequential_chars']:
        recommendations.append({
            'type': 'warning',
            'message': "Avoid using sequential characters (e.g., '1234', 'abcd') as they are easy to guess."
        })
    
    if analysis['repeated_chars']:
        recommendations.append({
            'type': 'warning',
            'message': "Avoid repeating characters multiple times in a row."
        })
    
    if analysis['common_patterns']:
        recommendations.append({
            'type': 'critical',
            'message': f"Avoid using common patterns like '{', '.join(analysis['common_patterns'])}'."
        })
    
    # Entropy recommendations
    if analysis['entropy'] < 40:
        recommendations.append({
            'type': 'critical',
            'message': "Password entropy is too low. Increase complexity by using more character types and length."
        })
    elif analysis['entropy'] < 60:
        recommendations.append({
            'type': 'warning',
            'message': "Consider increasing password entropy for better security."
        })
    
    # Model score recommendations
    if model_scores['final_score'] < 0.5:
        recommendations.append({
            'type': 'critical',
            'message': "AI models indicate this password is weak. Consider using a password manager to generate a stronger one."
        })
    elif model_scores['final_score'] < 0.7:
        recommendations.append({
            'type': 'warning',
            'message': "AI models suggest this password could be stronger. Consider adding more complexity."
        })
    
    # Add confidence-based recommendations
    if model_scores['model_confidence'] < 0.5:
        recommendations.append({
            'type': 'warning',
            'message': "The AI models have low confidence in this evaluation. Consider using a more conventional password pattern."
        })
    elif model_scores['model_confidence'] < 0.8:
        recommendations.append({
            'type': 'info',
            'message': "The AI models have moderate confidence in this evaluation. The password may have some unusual patterns."
        })
    
    return recommendations

def calculate_model_confidence(transformer_score, lstm_score, analysis):
    """Calculate model confidence based on scores and password characteristics"""
    # Base confidence is weighted differently for each model
    transformer_base = transformer_score * 0.6  # Transformer gets more weight
    lstm_base = lstm_score * 0.4  # LSTM gets less weight
    
    # Adjust confidence based on password characteristics
    confidence_factors = {
        'length_factor': min(1.0, analysis['length'] / 16),  # Normalize to 16 chars
        'type_factor': analysis['character_types'] / 4,  # Normalize to 4 types
        'entropy_factor': min(1.0, analysis['entropy'] / 80),  # Normalize to 80 bits
        'pattern_factor': 0.8 if analysis['sequential_chars'] or analysis['repeated_chars'] else 1.0,
        'common_pattern_factor': 0.7 if analysis['common_patterns'] else 1.0
    }
    
    # Calculate weighted confidence with different weights for each model
    weights = {
        'transformer': {
            'base': 0.5,
            'length': 0.15,
            'type': 0.15,
            'entropy': 0.1,
            'patterns': 0.1
        },
        'lstm': {
            'base': 0.4,
            'length': 0.2,
            'type': 0.2,
            'entropy': 0.1,
            'patterns': 0.1
        }
    }
    
    # Calculate confidence for each model separately
    transformer_confidence = (
        weights['transformer']['base'] * transformer_base +
        weights['transformer']['length'] * confidence_factors['length_factor'] +
        weights['transformer']['type'] * confidence_factors['type_factor'] +
        weights['transformer']['entropy'] * confidence_factors['entropy_factor'] +
        weights['transformer']['patterns'] * (confidence_factors['pattern_factor'] * confidence_factors['common_pattern_factor'])
    )
    
    lstm_confidence = (
        weights['lstm']['base'] * lstm_base +
        weights['lstm']['length'] * confidence_factors['length_factor'] +
        weights['lstm']['type'] * confidence_factors['type_factor'] +
        weights['lstm']['entropy'] * confidence_factors['entropy_factor'] +
        weights['lstm']['patterns'] * (confidence_factors['pattern_factor'] * confidence_factors['common_pattern_factor'])
    )
    
    # Ensure confidences are between 0 and 1
    transformer_confidence = max(0.0, min(1.0, transformer_confidence))
    lstm_confidence = max(0.0, min(1.0, lstm_confidence))
    
    return {
        'transformer_confidence': transformer_confidence,
        'lstm_confidence': lstm_confidence,
        'overall_confidence': (transformer_confidence + lstm_confidence) / 2
    }

def evaluate_password(password, models_dir="models"):
    """Evaluate password strength using both models and detailed analysis"""
    # Basic password pattern checks
    if len(password) < 8:
        return {
            'password': password,
            'transformer_score': 0.1,
            'lstm_score': 0.1,
            'final_score': 0.1,
            'strength_level': "Very Weak",
            'analysis': {
                'length': len(password),
                'character_types': 0,
                'entropy': 0
            },
            'recommendations': [{
                'type': 'critical',
                'message': "Password is too short (minimum 8 characters required)."
            }]
        }
    
    # Analyze password characteristics
    analysis = analyze_password_characteristics(password)
    
    # Load vocabulary and models
    vocab, char_to_idx = load_vocabulary(models_dir)
    tokens = tokenize_password(password, char_to_idx)
    
    # Initialize models
    transformer = PasswordTransformer(
        vocab_size=len(vocab),
        embedding_dim=64,
        num_heads=4,
        ff_dim=128,
        max_length=30
    )
    
    lstm = PasswordLSTM(
        vocab_size=len(vocab),
        embedding_dim=64,
        lstm_units=128,
        max_length=30
    )
    
    # Load trained models
    transformer.load_models(strength_model_path=os.path.join(models_dir, "transformer_model.keras"))
    lstm.load_models(strength_model_path=os.path.join(models_dir, "lstm_model.keras"))
    
    # Get predictions
    transformer_score = transformer.predict_strength(tokens)
    lstm_score = lstm.predict_strength(tokens)
    
    # Calculate model confidence
    model_confidence = calculate_model_confidence(transformer_score, lstm_score, analysis)
    
    # Adjust scores based on password characteristics with different weights
    if not analysis['has_uppercase']:
        transformer_score *= 0.85  # Less penalty for transformer
        lstm_score *= 0.75  # More penalty for LSTM
    if not analysis['has_lowercase']:
        transformer_score *= 0.85
        lstm_score *= 0.75
    if not analysis['has_digits']:
        transformer_score *= 0.85
        lstm_score *= 0.75
    if not analysis['has_special']:
        transformer_score *= 0.85
        lstm_score *= 0.75
    
    # Apply pattern penalties with different weights
    if analysis['sequential_chars']:
        transformer_score *= 0.8  # Less penalty for transformer
        lstm_score *= 0.6  # More penalty for LSTM
    if analysis['repeated_chars']:
        transformer_score *= 0.8
        lstm_score *= 0.6
    if analysis['common_patterns']:
        transformer_score *= 0.7
        lstm_score *= 0.5
    
    # Calculate final score with weighted model confidence
    final_score = (
        transformer_score * model_confidence['transformer_confidence'] * 0.6 +
        lstm_score * model_confidence['lstm_confidence'] * 0.4
    )
    
    # Determine strength level with adjusted thresholds
    if final_score < 0.2:
        strength_level = "Very Weak"
    elif final_score < 0.4:
        strength_level = "Weak"
    elif final_score < 0.6:
        strength_level = "Medium"
    elif final_score < 0.8:
        strength_level = "Strong"
    else:
        strength_level = "Very Strong"
    
    # Generate recommendations
    model_scores = {
        'transformer_score': transformer_score,
        'lstm_score': lstm_score,
        'final_score': final_score,
        'model_confidence': model_confidence['overall_confidence']
    }
    recommendations = generate_recommendations(analysis, model_scores)
    
    return {
        'password': password,
        'transformer_score': transformer_score,
        'lstm_score': lstm_score,
        'final_score': final_score,
        'strength_level': strength_level,
        'analysis': analysis,
        'recommendations': recommendations,
        'model_confidence': model_confidence['overall_confidence'],
        'transformer_confidence': model_confidence['transformer_confidence'],
        'lstm_confidence': model_confidence['lstm_confidence']
    }

def main():
    # Example usage
    while True:
        password = input("\nEnter a password to evaluate (or 'quit' to exit): ")
        if password.lower() == 'quit':
            break
            
        result = evaluate_password(password)
        print("\nPassword Strength Evaluation:")
        print(f"Password: {result['password']}")
        print(f"Transformer Score: {result['transformer_score']:.4f}")
        print(f"LSTM Score: {result['lstm_score']:.4f}")
        print(f"Final Score: {result['final_score']:.4f}")
        print(f"Strength Level: {result['strength_level']}")
        
        print("\nDetailed Analysis:")
        print(f"Length: {result['analysis']['length']} characters")
        print(f"Character Types: {result['analysis']['character_types']}")
        print(f"Entropy: {result['analysis']['entropy']:.2f} bits")
        
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"- [{rec['type'].upper()}] {rec['message']}")

if __name__ == "__main__":
    main() 