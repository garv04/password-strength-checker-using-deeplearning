import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd

def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """
    Load character vocabulary from file
    
    Args:
        vocab_path: Path to the vocabulary file
        
    Returns:
        Dictionary mapping characters to indices
    """
    with open(vocab_path, 'rb') as f:
        char_to_idx = pickle.load(f)
    return char_to_idx

def get_idx_to_char(char_to_idx: Dict[str, int]) -> Dict[int, str]:
    """
    Get mapping from indices to characters
    
    Args:
        char_to_idx: Dictionary mapping characters to indices
        
    Returns:
        Dictionary mapping indices to characters
    """
    return {idx: char for char, idx in char_to_idx.items()}

def tokenize_password(password: str, char_to_idx: Dict[str, int]) -> List[int]:
    """
    Convert password string to token indices
    
    Args:
        password: Password string
        char_to_idx: Dictionary mapping characters to indices
        
    Returns:
        List of token indices
    """
    # Handle unknown characters by replacing with 0 (reserved for padding)
    return [char_to_idx.get(c, 1) for c in password]  # 1 is reserved for unknown

def detokenize_password(tokens: List[int], idx_to_char: Dict[int, str]) -> str:
    """
    Convert token indices back to password string
    
    Args:
        tokens: List of token indices
        idx_to_char: Dictionary mapping indices to characters
        
    Returns:
        Password string
    """
    return ''.join([idx_to_char.get(token, '') for token in tokens if token > 0])  # Skip padding (0)

def prepare_passwords_for_models(
    passwords: List[str], 
    char_to_idx: Dict[str, int], 
    max_length: int = 30
) -> np.ndarray:
    """
    Tokenize and pad passwords for model input
    
    Args:
        passwords: List of password strings
        char_to_idx: Dictionary mapping characters to indices
        max_length: Maximum password length
        
    Returns:
        Numpy array of tokenized and padded passwords
    """
    tokenized_passwords = []
    
    for password in passwords:
        # Tokenize
        tokens = tokenize_password(password, char_to_idx)
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))
            
        tokenized_passwords.append(tokens)
    
    return np.array(tokenized_passwords)

def calculate_password_strength(password: str) -> int:
    """
    Calculate password strength score (0-4)
    
    Args:
        password: Password string
        
    Returns:
        Strength score (0: Very Weak, 1: Weak, 2: Medium, 3: Strong, 4: Very Strong)
    """
    score = 0
    
    # Length check
    if len(password) >= 8:
        score += 1
    if len(password) >= 12:
        score += 1
        
    # Complexity checks
    has_lowercase = any(c.islower() for c in password)
    has_uppercase = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    
    # Add points for complexity
    complexity_score = sum([has_lowercase, has_uppercase, has_digit, has_special])
    score += min(complexity_score, 2)  # Max 2 points for complexity
    
    # Ensure score is between 0-4
    return min(score, 4)

def prepare_sequence_data(
    passwords: List[str], 
    char_to_idx: Dict[str, int], 
    max_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for sequence prediction
    
    Args:
        passwords: List of password strings
        char_to_idx: Dictionary mapping characters to indices
        max_length: Maximum password length
        
    Returns:
        Tuple of (X, y) where X is input sequences and y is the next character
    """
    X, y = [], []
    
    for password in passwords:
        tokens = tokenize_password(password, char_to_idx)
        
        for i in range(1, len(tokens)):
            # Pad the input sequence to max_length
            padded_sequence = tokens[:i] + [0] * (max_length - i)
            
            # Ensure we don't exceed max_length
            if i <= max_length:
                X.append(padded_sequence[:max_length])
                y.append(tokens[i])
    
    return np.array(X), np.array(y)

def prepare_strength_data(
    passwords: List[str], 
    char_to_idx: Dict[str, int], 
    max_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for strength prediction
    
    Args:
        passwords: List of password strings
        char_to_idx: Dictionary mapping characters to indices
        max_length: Maximum password length
        
    Returns:
        Tuple of (X, y) where X is tokenized passwords and y is strength scores
    """
    X = []
    y = []
    
    for password in passwords:
        # Calculate strength
        strength = calculate_password_strength(password)
        
        # Tokenize password
        tokens = tokenize_password(password, char_to_idx)
        
        # Truncate or pad
        padded_tokens = tokens[:max_length] if len(tokens) >= max_length else tokens + [0] * (max_length - len(tokens))
        
        X.append(padded_tokens)
        y.append(strength)
    
    return np.array(X), np.array(y)

def load_processed_data(
    data_dir: str,
    file_name: str = 'train.csv',
    password_col: str = 'password'
) -> pd.DataFrame:
    """
    Load processed password data from CSV
    
    Args:
        data_dir: Directory containing data files
        file_name: Name of the file to load
        password_col: Name of the column containing passwords
        
    Returns:
        DataFrame containing the data
    """
    file_path = os.path.join(data_dir, file_name)
    return pd.read_csv(file_path)

def get_sample_batch(
    df: pd.DataFrame, 
    batch_size: int = 1000, 
    password_col: str = 'password',
    random_state: Optional[int] = None
) -> List[str]:
    """
    Get a random sample batch of passwords
    
    Args:
        df: DataFrame containing passwords
        batch_size: Number of passwords to sample
        password_col: Name of the column containing passwords
        random_state: Random seed for reproducibility
        
    Returns:
        List of password strings
    """
    if batch_size >= len(df):
        return df[password_col].tolist()
    
    sample = df.sample(batch_size, random_state=random_state)
    return sample[password_col].tolist()

def generate_batch_data(
    df: pd.DataFrame,
    char_to_idx: Dict[str, int],
    batch_size: int = 1000,
    password_col: str = 'password',
    max_length: int = 30,
    for_sequence_model: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of data for model training
    
    Args:
        df: DataFrame containing passwords
        char_to_idx: Dictionary mapping characters to indices
        batch_size: Number of passwords to sample
        password_col: Name of the column containing passwords
        max_length: Maximum password length
        for_sequence_model: Whether to prepare data for sequence model or strength model
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) prepared for the specified model type
    """
    passwords = get_sample_batch(df, batch_size, password_col, random_state)
    
    if for_sequence_model:
        return prepare_sequence_data(passwords, char_to_idx, max_length)
    else:
        return prepare_strength_data(passwords, char_to_idx, max_length) 