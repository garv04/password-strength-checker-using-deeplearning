import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

class PasswordDataProcessor:
    """
    Class for processing and cleaning password datasets
    """
    def __init__(self, min_length=6, max_length=30):
        self.min_length = min_length
        self.max_length = max_length
        self.vocabulary = None
        self.char_to_idx = None
        self.idx_to_char = None
    
    def clean_password(self, password):
        """Clean a single password by removing non-ASCII characters"""
        # Convert to string if not already
        password = str(password).strip()
        
        # Remove non-ASCII characters
        password = re.sub(r'[^\x00-\x7F]+', '', password)
        
        return password
    
    def filter_passwords(self, passwords):
        """Filter passwords based on criteria like length"""
        filtered = []
        
        for pwd in passwords:
            # Skip passwords that don't meet length criteria
            if len(pwd) < self.min_length or len(pwd) > self.max_length:
                continue
                
            # Skip passwords with non-printable characters
            if not all(32 <= ord(c) <= 126 for c in pwd):
                continue
                
            filtered.append(pwd)
            
        return filtered
    
    def remove_duplicates(self, passwords):
        """Remove duplicate passwords while preserving order"""
        seen = set()
        unique_passwords = []
        
        for pwd in passwords:
            if pwd not in seen:
                seen.add(pwd)
                unique_passwords.append(pwd)
                
        return unique_passwords
    
    def build_vocabulary(self, passwords):
        """Build character vocabulary from passwords"""
        # Flatten all passwords into characters
        all_chars = ''.join(passwords)
        
        # Count characters
        char_counts = Counter(all_chars)
        
        # Sort by frequency
        sorted_chars = sorted(char_counts.keys())
        
        # Create mappings
        self.vocabulary = sorted_chars
        self.char_to_idx = {c: i for i, c in enumerate(sorted_chars)}
        self.idx_to_char = {i: c for i, c in enumerate(sorted_chars)}
        
        return self.vocabulary
    
    def calculate_features(self, passwords):
        """
        Calculate features for each password:
        - length
        - character diversity (unique chars / length)
        - contains_lowercase
        - contains_uppercase
        - contains_digits
        - contains_special
        """
        features = []
        
        for pwd in passwords:
            char_diversity = len(set(pwd)) / len(pwd) if len(pwd) > 0 else 0
            contains_lowercase = any(c.islower() for c in pwd)
            contains_uppercase = any(c.isupper() for c in pwd)
            contains_digits = any(c.isdigit() for c in pwd)
            contains_special = any(not c.isalnum() for c in pwd)
            
            features.append({
                'password': pwd,
                'length': len(pwd),
                'char_diversity': char_diversity,
                'contains_lowercase': contains_lowercase,
                'contains_uppercase': contains_uppercase,
                'contains_digits': contains_digits,
                'contains_special': contains_special
            })
            
        return pd.DataFrame(features)
    
    def tokenize_passwords(self, passwords, char_level=True):
        """
        Tokenize passwords either at character level or subword level
        For character level, each character becomes a token
        """
        if not self.vocabulary:
            self.build_vocabulary(passwords)
            
        if char_level:
            # Character-level tokenization
            tokenized = []
            for pwd in passwords:
                tokens = [self.char_to_idx.get(c, 0) for c in pwd]  # Use 0 for unknown chars
                tokenized.append(tokens)
            return tokenized
        else:
            # For subword tokenization, we would need to implement BPE or similar
            raise NotImplementedError("Subword tokenization not implemented yet")
    
    def process_file(self, file_path, output_path=None):
        """Process a password file from start to finish"""
        print(f"Loading passwords from {file_path}")
        
        # Read passwords
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            raw_passwords = [line.strip() for line in tqdm(f) if line.strip()]
        
        print(f"Loaded {len(raw_passwords)} raw passwords")
        
        # Clean passwords
        print("Cleaning passwords...")
        cleaned_passwords = [self.clean_password(pwd) for pwd in tqdm(raw_passwords)]
        
        # Filter passwords
        print("Filtering passwords...")
        filtered_passwords = self.filter_passwords(cleaned_passwords)
        print(f"Filtered down to {len(filtered_passwords)} passwords")
        
        # Remove duplicates
        print("Removing duplicates...")
        unique_passwords = self.remove_duplicates(filtered_passwords)
        print(f"Removed duplicates, resulting in {len(unique_passwords)} passwords")
        
        # Calculate features
        print("Calculating features...")
        password_df = self.calculate_features(unique_passwords)
        
        # Save if output path provided
        if output_path:
            print(f"Saving processed passwords to {output_path}")
            password_df.to_csv(output_path, index=False)
        
        return password_df

    def split_data(self, password_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split data into train, validation, and test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
        
        # Shuffle data
        password_df = password_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        n = len(password_df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split data
        train_df = password_df.iloc[:train_end]
        val_df = password_df.iloc[train_end:val_end]
        test_df = password_df.iloc[val_end:]
        
        return train_df, val_df, test_df

if __name__ == "__main__":
    # Example usage
    processor = PasswordDataProcessor(min_length=8)
    
    # Paths - adjust as necessary
    data_dir = "../../data"
    raw_file = os.path.join(data_dir, "rockyou.txt")
    processed_file = os.path.join(data_dir, "processed_passwords.csv")
    
    # Check if raw file exists
    if not os.path.exists(raw_file):
        print(f"Error: {raw_file} not found. Please download the dataset first.")
    else:
        # Process the data
        password_df = processor.process_file(raw_file, processed_file)
        
        # Split the data
        train_df, val_df, test_df = processor.split_data(password_df)
        
        # Save splits
        train_df.to_csv(os.path.join(data_dir, "train_passwords.csv"), index=False)
        val_df.to_csv(os.path.join(data_dir, "val_passwords.csv"), index=False)
        test_df.to_csv(os.path.join(data_dir, "test_passwords.csv"), index=False)
        
        print("Data processing and splitting complete!") 