import time
import string
import random
import re
import itertools
from tqdm import tqdm
import numpy as np
import sys
import os
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local imports
from src.evaluation.strength_evaluator import PasswordStrengthEvaluator

class PasswordCrackerSimulator:
    """
    Simulator for various password cracking techniques:
    - Brute Force Attack
    - Dictionary Attack
    - Rule-based (Hybrid) Attack
    """
    
    def __init__(self, dictionary_path=None, rules_path=None, ml_model=None):
        """
        Initialize the cracker simulator
        
        Args:
            dictionary_path: Path to dictionary file with common passwords (one per line)
            rules_path: Path to file with transformation rules (JSON format)
            ml_model: Optional ML model for intelligent password generation
        """
        self.dictionary = []
        self.rules = {}
        self.ml_model = ml_model
        self.load_resources(dictionary_path, rules_path)
        
        # Character sets for brute force
        self.charsets = {
            'lowercase': string.ascii_lowercase,
            'uppercase': string.ascii_uppercase,
            'digits': string.digits,
            'special': string.punctuation,
            'all': string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation
        }
        
        # Common substitutions for rule-based attacks
        self.substitutions = {
            'a': ['@', '4'],
            'b': ['8'],
            'e': ['3'],
            'i': ['1', '!'],
            'l': ['1'],
            'o': ['0'],
            's': ['$', '5'],
            't': ['7'],
            'z': ['2']
        }
    
    def load_resources(self, dictionary_path, rules_path):
        """Load dictionary and rules from files"""
        # Load dictionary if provided
        if dictionary_path and os.path.exists(dictionary_path):
            with open(dictionary_path, 'r', encoding='latin-1', errors='ignore') as f:
                self.dictionary = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.dictionary)} words from dictionary")
        
        # Load rules if provided
        if rules_path and os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                self.rules = json.load(f)
            print(f"Loaded {len(self.rules)} rule sets")
    
    def brute_force_attack(self, password, charset='all', max_length=8, max_attempts=1000000, verbose=True):
        """
        Simulate a brute force attack
        
        Args:
            password: The password to crack
            charset: Character set to use ('lowercase', 'uppercase', 'digits', 'special', 'all')
            max_length: Maximum password length to try
            max_attempts: Maximum number of attempts to simulate
            verbose: Whether to show progress
        
        Returns:
            dict: Results of the attack (success, attempts, time)
        """
        # Get the character set
        chars = self.charsets.get(charset, self.charsets['all'])
        
        # Start timing
        start_time = time.time()
        
        # Initialize variables
        attempts = 0
        success = False
        cracked_password = None
        
        if verbose:
            print(f"Starting brute force attack (charset: {charset}, max_length: {max_length})")
            
        # Try different password lengths
        for length in range(1, min(max_length + 1, len(password) + 1)):
            if verbose:
                print(f"Trying length {length}...")
                
            # Estimate total combinations for this length
            total_combinations = len(chars) ** length
            
            # Create an iterator for all combinations
            combinations = itertools.product(chars, repeat=length)
            
            # Iterate through combinations
            for combo in combinations:
                # Form the password
                guess = ''.join(combo)
                attempts += 1
                
                # Check if we found the password
                if guess == password:
                    success = True
                    cracked_password = guess
                    break
                
                # Check if we've reached max attempts
                if attempts >= max_attempts:
                    break
                
                # Show progress occasionally if verbose
                if verbose and attempts % 1000000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {attempts:,} attempts, {elapsed:.2f} seconds elapsed")
            
            # Break if we've found the password or reached max attempts
            if success or attempts >= max_attempts:
                break
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Format results
        result = {
            'attack_type': 'brute_force',
            'success': success,
            'attempts': attempts,
            'elapsed_time': elapsed_time,
            'cracked_password': cracked_password if success else None,
            'charset': charset,
            'max_length_tried': length,
            'attempts_per_second': attempts / elapsed_time if elapsed_time > 0 else 0
        }
        
        if verbose:
            if success:
                print(f"Password cracked in {attempts:,} attempts and {elapsed_time:.2f} seconds!")
                print(f"Password: {cracked_password}")
            else:
                print(f"Failed to crack password after {attempts:,} attempts and {elapsed_time:.2f} seconds")
        
        return result
    
    def dictionary_attack(self, password, use_common_mutations=True, max_attempts=1000000, verbose=True):
        """
        Simulate a dictionary attack
        
        Args:
            password: The password to crack
            use_common_mutations: Whether to apply common mutations (e.g., adding numbers)
            max_attempts: Maximum number of attempts to simulate
            verbose: Whether to show progress
        
        Returns:
            dict: Results of the attack (success, attempts, time)
        """
        if not self.dictionary:
            raise ValueError("Dictionary not loaded. Please provide a dictionary file.")
        
        # Start timing
        start_time = time.time()
        
        # Initialize variables
        attempts = 0
        success = False
        cracked_password = None
        
        if verbose:
            print(f"Starting dictionary attack ({len(self.dictionary):,} words in dictionary)")
        
        # Sort dictionary by frequency (assuming most common words are first)
        sorted_dict = self.dictionary
        
        # Try each word in the dictionary
        for word in sorted_dict:
            attempts += 1
            
            # Check if we found the password
            if word == password:
                success = True
                cracked_password = word
                break
            
            # Check if we've reached max attempts
            if attempts >= max_attempts:
                break
            
            # Show progress occasionally if verbose
            if verbose and attempts % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  {attempts:,} attempts, {elapsed:.2f} seconds elapsed")
        
        # If not found and common mutations are enabled
        if not success and use_common_mutations and attempts < max_attempts:
            if verbose:
                print("No match found in dictionary. Trying common mutations...")
            
            # Try common mutations (up to a limit to avoid too many combinations)
            for word in sorted_dict[:min(len(sorted_dict), 10000)]:
                if attempts >= max_attempts:
                    break
                
                # Try adding common numbers/years to the end
                suffixes = ['', '123', '1234', '12345', '123456', 
                           '2023', '2022', '2021', '2020', 
                           '!', '#', '@', '$']
                
                for suffix in suffixes:
                    mutated = word + suffix
                    attempts += 1
                    
                    if mutated == password:
                        success = True
                        cracked_password = mutated
                        break
                    
                    # Check if we've reached max attempts
                    if attempts >= max_attempts:
                        break
                
                # Break if we've found the password
                if success:
                    break
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Format results
        result = {
            'attack_type': 'dictionary',
            'success': success,
            'attempts': attempts,
            'elapsed_time': elapsed_time,
            'cracked_password': cracked_password if success else None,
            'dictionary_size': len(self.dictionary),
            'used_mutations': use_common_mutations,
            'attempts_per_second': attempts / elapsed_time if elapsed_time > 0 else 0
        }
        
        if verbose:
            if success:
                print(f"Password cracked in {attempts:,} attempts and {elapsed_time:.2f} seconds!")
                print(f"Password: {cracked_password}")
            else:
                print(f"Failed to crack password after {attempts:,} attempts and {elapsed_time:.2f} seconds")
        
        return result
    
    def rule_based_attack(self, password, max_attempts=1000000, verbose=True):
        """
        Simulate a rule-based (hybrid) attack that applies transformations
        
        Args:
            password: The password to crack
            max_attempts: Maximum number of attempts to simulate
            verbose: Whether to show progress
        
        Returns:
            dict: Results of the attack (success, attempts, time)
        """
        if not self.dictionary:
            raise ValueError("Dictionary not loaded. Please provide a dictionary file.")
        
        # Start timing
        start_time = time.time()
        
        # Initialize variables
        attempts = 0
        success = False
        cracked_password = None
        last_rule_applied = None
        
        if verbose:
            print(f"Starting rule-based attack")
        
        # Define common transformation rules
        rules = [
            # Capitalization rules
            lambda w: w.capitalize(),
            lambda w: w.upper(),
            lambda w: w.lower(),
            
            # Append/prepend digits
            lambda w: w + "1",
            lambda w: w + "123",
            lambda w: w + "!",
            lambda w: "123" + w,
            
            # Leet speak transformations
            lambda w: w.replace('a', '@').replace('e', '3').replace('i', '1').replace('o', '0').replace('s', '$'),
            
            # Reversed word
            lambda w: w[::-1],
            
            # Duplicated word
            lambda w: w + w,
        ]
        
        # Apply rules to dictionary words
        for word in self.dictionary[:min(len(self.dictionary), 5000)]:  # Limit to reduce computation
            original_word = word
            
            # Try the original word first
            attempts += 1
            if word == password:
                success = True
                cracked_password = word
                break
            
            # Apply each rule
            for i, rule in enumerate(rules):
                if attempts >= max_attempts:
                    break
                
                try:
                    transformed = rule(word)
                    attempts += 1
                    
                    if transformed == password:
                        success = True
                        cracked_password = transformed
                        last_rule_applied = i
                        break
                except:
                    # Skip if rule application fails
                    continue
            
            # Break if we've found the password or reached max attempts
            if success or attempts >= max_attempts:
                break
            
            # Show progress occasionally if verbose
            if verbose and attempts % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  {attempts:,} attempts, {elapsed:.2f} seconds elapsed")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Format results
        result = {
            'attack_type': 'rule_based',
            'success': success,
            'attempts': attempts,
            'elapsed_time': elapsed_time,
            'cracked_password': cracked_password if success else None,
            'rule_applied': last_rule_applied,
            'attempts_per_second': attempts / elapsed_time if elapsed_time > 0 else 0
        }
        
        if verbose:
            if success:
                print(f"Password cracked in {attempts:,} attempts and {elapsed_time:.2f} seconds!")
                print(f"Password: {cracked_password}")
                if last_rule_applied is not None:
                    print(f"Rule applied: {last_rule_applied}")
            else:
                print(f"Failed to crack password after {attempts:,} attempts and {elapsed_time:.2f} seconds")
        
        return result
    
    def ml_guided_attack(self, password, max_attempts=1000000, verbose=True):
        """
        Simulate an ML-guided attack using the trained model to generate likely passwords
        
        Args:
            password: The password to crack
            max_attempts: Maximum number of attempts to simulate
            verbose: Whether to show progress
        
        Returns:
            dict: Results of the attack (success, attempts, time)
        """
        if not self.ml_model:
            raise ValueError("ML model not provided. Cannot run ML-guided attack.")
        
        # Start timing
        start_time = time.time()
        
        # Initialize variables
        attempts = 0
        success = False
        cracked_password = None
        
        if verbose:
            print(f"Starting ML-guided attack")
        
        # This is a placeholder for the actual ML-guided attack logic
        # In a real implementation, you would:
        # 1. Use the ML model to generate likely passwords
        # 2. Try these passwords first
        # 3. Refine based on feedback
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Format results
        result = {
            'attack_type': 'ml_guided',
            'success': success,
            'attempts': attempts,
            'elapsed_time': elapsed_time,
            'cracked_password': cracked_password if success else None,
            'attempts_per_second': attempts / elapsed_time if elapsed_time > 0 else 0
        }
        
        if verbose:
            if success:
                print(f"Password cracked in {attempts:,} attempts and {elapsed_time:.2f} seconds!")
                print(f"Password: {cracked_password}")
            else:
                print(f"Failed to crack password after {attempts:,} attempts and {elapsed_time:.2f} seconds")
        
        return result
    
    def run_all_attacks(self, password, max_attempts_per_attack=100000, verbose=True):
        """
        Run all attack types and compare results
        
        Args:
            password: The password to crack
            max_attempts_per_attack: Maximum attempts per attack type
            verbose: Whether to show progress
            
        Returns:
            dict: Results of all attacks
        """
        results = {}
        
        # Calculate theoretical metrics using the strength evaluator
        evaluator = PasswordStrengthEvaluator(common_passwords=set(self.dictionary[:10000]) if self.dictionary else set())
        theoretical_metrics = evaluator.evaluate(password)
        
        if verbose:
            print("\n===== Password Strength Analysis =====")
            print(f"Strength Score: {theoretical_metrics['strength_score']}/100 ({theoretical_metrics['strength_category']})")
            print(f"Entropy: {theoretical_metrics['entropy_bits']:.2f} bits")
            print("Theoretical crack times:")
            for attack_type, time_info in theoretical_metrics['crack_times'].items():
                print(f"  {attack_type}: {time_info['formatted']}")
            print("\n===== Attack Simulations =====")
        
        # Run brute force attack
        try:
            results['brute_force'] = self.brute_force_attack(
                password, 
                charset='all', 
                max_length=min(len(password), 8),  # Limit for simulation speed
                max_attempts=max_attempts_per_attack,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"Brute force attack error: {str(e)}")
            results['brute_force'] = {'error': str(e), 'success': False}
        
        # Run dictionary attack
        try:
            results['dictionary'] = self.dictionary_attack(
                password, 
                use_common_mutations=True,
                max_attempts=max_attempts_per_attack,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"Dictionary attack error: {str(e)}")
            results['dictionary'] = {'error': str(e), 'success': False}
        
        # Run rule-based attack
        try:
            results['rule_based'] = self.rule_based_attack(
                password,
                max_attempts=max_attempts_per_attack,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"Rule-based attack error: {str(e)}")
            results['rule_based'] = {'error': str(e), 'success': False}
        
        # Run ML-guided attack if model available
        if self.ml_model:
            try:
                results['ml_guided'] = self.ml_guided_attack(
                    password,
                    max_attempts=max_attempts_per_attack,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"ML-guided attack error: {str(e)}")
                results['ml_guided'] = {'error': str(e), 'success': False}
        
        # Summarize results
        if verbose:
            print("\n===== Attack Results Summary =====")
            for attack_type, result in results.items():
                if 'error' in result:
                    print(f"{attack_type}: Error - {result['error']}")
                else:
                    print(f"{attack_type}: {'Success' if result['success'] else 'Failed'} - "
                          f"{result['attempts']:,} attempts in {result['elapsed_time']:.2f} seconds")
            print("\nTheoretical vs. Simulated Comparison:")
            for attack_type, result in results.items():
                if 'error' not in result:
                    theoretical_key = 'offline_fast'  # Assuming the simulator is analogous to fast offline
                    if theoretical_key in theoretical_metrics['crack_times']:
                        theo_time = theoretical_metrics['crack_times'][theoretical_key]['seconds']
                        actual_time = result['attempts'] / result['attempts_per_second'] if result['attempts_per_second'] > 0 else float('inf')
                        print(f"{attack_type}: Theoretical: {evaluator.format_time(theo_time)}, "
                              f"Simulated: {evaluator.format_time(actual_time)}")
        
        return {
            'attacks': results,
            'theoretical': theoretical_metrics
        }

if __name__ == "__main__":
    # Example usage
    # Dictionary path - adjust as needed
    dictionary_path = "../../data/rockyou.txt"
    
    # Check if dictionary exists
    if not os.path.exists(dictionary_path):
        print(f"Dictionary file not found at {dictionary_path}")
        print("Using a small built-in list for demo purposes")
        # Create a small dictionary for demo
        with open("demo_dict.txt", "w") as f:
            f.write("\n".join([
                "password", "123456", "qwerty", "admin", "welcome",
                "monkey", "sunshine", "superman", "iloveyou", "football"
            ]))
        dictionary_path = "demo_dict.txt"
    
    # Create simulator
    simulator = PasswordCrackerSimulator(dictionary_path=dictionary_path)
    
    # Test passwords
    test_passwords = [
        "password123",  # Common password with numbers
        "P@ssw0rd!",    # Simple substitutions
        "zQ5%tY9@",     # Complex but short
    ]
    
    # Run attacks on each password
    for pwd in test_passwords:
        print("\n" + "=" * 50)
        print(f"Testing password: {pwd}")
        print("=" * 50)
        
        results = simulator.run_all_attacks(pwd, max_attempts_per_attack=100000)
        
        print("\n" + "-" * 50 + "\n") 