import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import string
import sys

def load_rockyou_data(data_path, sample_size=None, min_length=8, max_length=20):
    """
    Load passwords from the RockYou dataset with filtering
    
    Args:
        data_path (str): Path to the rockyou.txt file
        sample_size (int, optional): Number of passwords to sample. If None, load all.
        min_length (int): Minimum password length to include
        max_length (int): Maximum password length to include
        
    Returns:
        list: List of passwords meeting the criteria
    """
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return []
    
    print(f"Loading passwords from {data_path}...")
    passwords = []
    
    # Define allowed characters (ASCII printable)
    allowed_chars = set(string.printable)
    
    with open(data_path, 'rb') as f:
        for line in tqdm(f, desc="Reading passwords", unit="lines"):
            try:
                # Decode bytes to string, handle encoding errors
                password = line.strip().decode('utf-8', errors='ignore')
                
                # Apply filtering criteria
                if (min_length <= len(password) <= max_length and
                    all(c in allowed_chars for c in password)):
                    passwords.append(password)
                    
                    # If we've reached the sample size, stop
                    if sample_size and len(passwords) >= sample_size:
                        break
            except Exception as e:
                # Skip problematic lines
                continue
                
    print(f"Loaded {len(passwords)} passwords")
    return passwords

def create_password_dataframe(passwords):
    """
    Create a DataFrame with password features and strength scores
    
    Args:
        passwords (list): List of password strings
        
    Returns:
        pd.DataFrame: DataFrame with password features and strength scores
    """
    # Initialize empty lists for features
    lengths = []
    has_uppercase = []
    has_lowercase = []
    has_digit = []
    has_special = []
    char_sets = []
    strengths = []  # New list for strength scores
    
    special_chars = set(string.punctuation)
    
    for password in tqdm(passwords, desc="Extracting features"):
        # Length
        length = len(password)
        lengths.append(length)
        
        # Character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_dig = any(c.isdigit() for c in password)
        has_spec = any(c in special_chars for c in password)
        
        has_uppercase.append(int(has_upper))
        has_lowercase.append(int(has_lower))
        has_digit.append(int(has_dig))
        has_special.append(int(has_spec))
        
        # Count different character sets used (0-4)
        char_set_count = sum([has_upper, has_lower, has_dig, has_spec])
        char_sets.append(char_set_count)
        
        # Calculate strength score (0-1)
        # Formula: (0.4 * length + 0.4 * char_sets + 0.2 * entropy) normalized to 0-1
        base_score = 0.4 * min(length / 20, 1.0)  # Length score (max 20 chars)
        charset_score = 0.4 * (char_set_count / 4)  # Character set score
        
        # Calculate entropy
        char_counts = Counter(password)
        entropy = 0
        for char, count in char_counts.items():
            prob = count / length
            entropy -= prob * np.log2(prob)
        entropy_score = 0.2 * min(entropy / 4, 1.0)  # Normalize entropy (max 4 bits)
        
        strength = base_score + charset_score + entropy_score
        strengths.append(strength)
    
    # Create DataFrame
    df = pd.DataFrame({
        'password': passwords,
        'length': lengths,
        'has_uppercase': has_uppercase,
        'has_lowercase': has_lowercase,
        'has_digit': has_digit,
        'has_special': has_special,
        'char_sets': char_sets,
        'strength': strengths
    })
    
    return df

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        df (pd.DataFrame): Password DataFrame
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split: training + validation vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Second split: training vs validation
    # Calculate validation size relative to train_val_df
    relative_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=random_state
    )
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, output_dir):
    """
    Save processed data splits to CSV files
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        output_dir (str): Directory to save files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    train_path = os.path.join(output_dir, "train_passwords.csv")
    val_path = os.path.join(output_dir, "val_passwords.csv")
    test_path = os.path.join(output_dir, "test_passwords.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved processed data to {output_dir}")
    print(f"  - Training set: {train_path}")
    print(f"  - Validation set: {val_path}")
    print(f"  - Test set: {test_path}")

def generate_vocabulary(passwords, output_dir):
    """
    Generate character vocabulary from passwords
    
    Args:
        passwords (list): List of passwords
        output_dir (str): Directory to save vocabulary
    """
    # Get all unique characters
    all_chars = set()
    for password in passwords:
        all_chars.update(set(password))
    
    # Sort characters for consistency
    vocab = sorted(list(all_chars))
    
    # Create char-to-index and index-to-char mappings
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    # Save vocabulary
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    char_to_idx_path = os.path.join(output_dir, "char_to_idx.pkl")
    idx_to_char_path = os.path.join(output_dir, "idx_to_char.pkl")
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    with open(char_to_idx_path, 'wb') as f:
        pickle.dump(char_to_idx, f)
    
    with open(idx_to_char_path, 'wb') as f:
        pickle.dump(idx_to_char, f)
    
    print(f"Generated vocabulary with {len(vocab)} unique characters")
    print(f"Vocabulary saved to {output_dir}")

def main():
    """
    Main function to process the RockYou dataset
    """
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    vocab_dir = os.path.join(data_dir, "vocab")
    
    # Input file
    rockyou_path = os.path.join(data_dir, "rockyou.txt")
    
    # Check if RockYou dataset exists
    if not os.path.exists(rockyou_path):
        print(f"Error: RockYou dataset not found at {rockyou_path}")
        print("Please run the download_rockyou.py script first")
        return False

    # Parse command line arguments
    sample_size = None
    min_length = 8
    max_length = 20
    
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            print(f"Using sample size: {sample_size}")
        except ValueError:
            print("Invalid sample size, using all passwords")
    
    # Load passwords
    passwords = load_rockyou_data(
        rockyou_path, 
        sample_size=sample_size,
        min_length=min_length,
        max_length=max_length
    )
    
    if not passwords:
        print("No passwords loaded. Exiting.")
        return False
    
    # Create DataFrame with features
    df = create_password_dataframe(passwords)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df, processed_dir)
    
    # Generate vocabulary
    generate_vocabulary(passwords, vocab_dir)
    
    print("Processing complete!")
    return True

if __name__ == "__main__":
    main() 