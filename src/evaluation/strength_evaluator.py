import math
import re
import numpy as np
from collections import Counter
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local imports
# from src.models.lstm_model import PasswordLSTM

class PasswordStrengthEvaluator:
    """
    Class for evaluating password strength using various metrics
    and simulation of different attack models.
    """
    
    # Entropy bits required for each strength level
    STRENGTH_THRESHOLDS = {
        'very_weak': 28,    # < 28 bits
        'weak': 36,         # 28-36 bits
        'moderate': 60,     # 36-60 bits
        'strong': 128,      # 60-128 bits
        'very_strong': 128  # >= 128 bits
    }
    
    # Character sets for entropy calculation
    CHARSET_SIZES = {
        'lowercase': 26,    # a-z
        'uppercase': 26,    # A-Z
        'digits': 10,       # 0-9
        'special': 33       # special characters
    }
    
    # Average cracking speeds (guesses per second) for different attack types
    CRACKING_SPEEDS = {
        'online': 10,                    # Online attack (throttled)
        'offline_slow': 1_000,           # Slow hash (bcrypt, Argon2)
        'offline_fast': 1_000_000,       # Fast hash (SHA-256)
        'offline_gpu': 1_000_000_000,    # GPU attack (fast hash)
        'offline_botnet': 10_000_000_000 # Distributed attack
    }
    
    def __init__(self, ml_model=None, common_passwords=None):
        self.ml_model = ml_model
        self.common_passwords = common_passwords if common_passwords else set()
        
        # Common password patterns (regex)
        self.patterns = {
            'dates': r'(19|20)\d{2}',
            'repeats': r'(.+)\1{2,}',
            'sequences': r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|012|123|234|345|456|567|678|789)',
            'keyboard_rows': r'(qwert|asdfg|zxcvb|yuiop|hjkl|bnm|1234|567|890)',
            'common_words': r'(password|welcome|admin|login|user|qwerty)'
        }
        
    def calculate_entropy(self, password):
        """
        Calculate the entropy of a password based on shannon entropy and character set size
        
        Shannon Entropy = -sum(p_i * log2(p_i)) where p_i is the probability of character i
        Character Set Entropy = log2(N^L) = L * log2(N) where N is charset size and L is length
        """
        # Character set entropy calculation
        length = len(password)
        
        # Determine character sets used
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digits = bool(re.search(r'[0-9]', password))
        has_special = bool(re.search(r'[^a-zA-Z0-9]', password))
        
        # Calculate effective charset size
        charset_size = 0
        if has_lower:
            charset_size += self.CHARSET_SIZES['lowercase']
        if has_upper:
            charset_size += self.CHARSET_SIZES['uppercase']
        if has_digits:
            charset_size += self.CHARSET_SIZES['digits']
        if has_special:
            charset_size += self.CHARSET_SIZES['special']
            
        # Calculate entropy bits (ideal case: all characters equally probable)
        naive_entropy = length * math.log2(max(charset_size, 1))
        
        # Shannon entropy calculation (actual character distribution)
        char_counts = Counter(password)
        shannon_entropy = 0
        for count in char_counts.values():
            p_i = count / length
            shannon_entropy -= p_i * math.log2(p_i)
        
        # Final entropy is the minimum of the two approaches
        # This gives a conservative estimate
        entropy = min(naive_entropy, shannon_entropy * length)
        
        return entropy
    
    def check_common_patterns(self, password):
        """Check for common patterns in the password"""
        patterns_found = []
        
        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern, password, re.IGNORECASE):
                patterns_found.append(pattern_name)
                
        return patterns_found
    
    def estimate_crack_time(self, password, attack_type='offline_fast'):
        """
        Estimate the time it would take to crack a password using
        different attack methods.
        
        Returns time in seconds.
        """
        # Get the cracking speed for the given attack type
        speed = self.CRACKING_SPEEDS.get(attack_type, self.CRACKING_SPEEDS['offline_fast'])
        
        # If password is in common list, assume it's cracked immediately
        if password in self.common_passwords:
            return 0
        
        # Calculate entropy
        entropy = self.calculate_entropy(password)
        
        # Calculate average guesses needed (2^entropy / 2 on average)
        avg_guesses = 2 ** (entropy - 1)
        
        # Calculate time in seconds
        time_seconds = avg_guesses / speed
        
        return time_seconds
    
    def format_time(self, seconds):
        """Format seconds into a human-readable time string"""
        if seconds < 1:
            return "instantaneous"
        elif seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        elif seconds < 31536000:
            return f"{seconds/86400:.1f} days"
        elif seconds < 3153600000:  # 100 years
            return f"{seconds/31536000:.1f} years"
        else:
            return "centuries"
    
    def get_strength_score(self, password):
        """
        Calculate a password strength score from 0-100
        based on entropy, pattern checking, and ML model if available
        """
        # Start with base score from entropy
        entropy = self.calculate_entropy(password)
        
        # Normalize entropy to 0-100 scale (caps at 128 bits for very_strong)
        base_score = min(entropy / 1.28, 100)
        
        # Penalty for common patterns
        patterns_found = self.check_common_patterns(password)
        pattern_penalty = len(patterns_found) * 10
        
        # Penalty for common passwords
        common_penalty = 50 if password in self.common_passwords else 0
        
        # ML model prediction if available
        ml_penalty = 0
        if self.ml_model:
            # Convert password to tokens using model's tokenizer
            # and get strength prediction
            # This is a placeholder - actual implementation depends on the ML model
            # strength_class, confidence = self.ml_model.predict_strength(...)
            # ml_penalty = (5 - strength_class) * 10  # Assuming 5 classes (0-4)
            pass
        
        # Calculate final score
        score = max(0, base_score - pattern_penalty - common_penalty - ml_penalty)
        
        return round(score)
    
    def get_strength_category(self, score):
        """Convert a numerical score to a strength category"""
        if score < 20:
            return "Very Weak"
        elif score < 40:
            return "Weak"
        elif score < 60:
            return "Moderate"
        elif score < 80:
            return "Strong"
        else:
            return "Very Strong"
    
    def generate_improvements(self, password, score):
        """Generate recommendations to improve password strength"""
        recommendations = []
        
        # If too short
        if len(password) < 12:
            recommendations.append("Make your password longer (at least 12 characters)")
        
        # If missing character classes
        if not re.search(r'[a-z]', password):
            recommendations.append("Add lowercase letters")
        if not re.search(r'[A-Z]', password):
            recommendations.append("Add uppercase letters")
        if not re.search(r'[0-9]', password):
            recommendations.append("Add numbers")
        if not re.search(r'[^a-zA-Z0-9]', password):
            recommendations.append("Add special characters")
        
        # If has patterns
        patterns_found = self.check_common_patterns(password)
        if patterns_found:
            recommendations.append(f"Avoid common patterns ({', '.join(patterns_found)})")
        
        # If already strong but could be stronger
        if score >= 70 and len(recommendations) == 0:
            recommendations.append("Consider using a password manager to generate and store even stronger passwords")
        
        return recommendations
    
    def evaluate(self, password):
        """
        Full password evaluation
        
        Returns a dictionary with:
        - strength_score (0-100)
        - strength_category (Very Weak, Weak, Moderate, Strong, Very Strong)
        - entropy_bits
        - crack_times for different attack scenarios
        - patterns_found
        - improvements
        """
        # Calculate score
        score = self.get_strength_score(password)
        category = self.get_strength_category(score)
        
        # Calculate entropy
        entropy = self.calculate_entropy(password)
        
        # Find patterns
        patterns_found = self.check_common_patterns(password)
        
        # Estimate crack times
        crack_times = {}
        for attack_type in self.CRACKING_SPEEDS.keys():
            seconds = self.estimate_crack_time(password, attack_type)
            crack_times[attack_type] = {
                'seconds': seconds,
                'formatted': self.format_time(seconds)
            }
        
        # Generate recommendations
        improvements = self.generate_improvements(password, score)
        
        # Compile results
        result = {
            'password': '*' * len(password),  # Mask the password in output
            'strength_score': score,
            'strength_category': category,
            'entropy_bits': entropy,
            'length': len(password),
            'patterns_found': patterns_found,
            'crack_times': crack_times,
            'improvements': improvements
        }
        
        return result

if __name__ == "__main__":
    # Example usage
    evaluator = PasswordStrengthEvaluator()
    
    # Test passwords
    test_passwords = [
        "password123",
        "P@ssw0rd!",
        "correct-horse-battery-staple",
        "xkcd-inspired-passphrase-is-strong"
    ]
    
    # Evaluate each password
    for pwd in test_passwords:
        result = evaluator.evaluate(pwd)
        
        print(f"\nPassword: {result['password']}")
        print(f"Strength: {result['strength_score']}/100 ({result['strength_category']})")
        print(f"Entropy: {result['entropy_bits']:.2f} bits")
        print(f"Offline Fast Hash Crack Time: {result['crack_times']['offline_fast']['formatted']}")
        
        if result['patterns_found']:
            print(f"Patterns Found: {', '.join(result['patterns_found'])}")
            
        if result['improvements']:
            print("Recommendations:")
            for rec in result['improvements']:
                print(f"- {rec}")
        print("-" * 40) 