import os
import sys
from src.preprocessing.process_rockyou import (
    load_rockyou_data,
    create_password_dataframe,
    split_data,
    save_processed_data,
    generate_vocabulary
)
from src.models.transformer_model import PasswordTransformer
from src.models.lstm_model import PasswordLSTM
import tensorflow as tf
import numpy as np
import pickle

def tokenize_passwords(passwords, char_to_idx, max_length=30):
    """Convert passwords to token sequences"""
    tokenized = []
    for password in passwords:
        # Convert characters to indices
        tokens = [char_to_idx[c] for c in password if c in char_to_idx]
        # Pad or truncate to max_length
        if len(tokens) < max_length:
            tokens = tokens + [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        tokenized.append(tokens)
    return np.array(tokenized)

def main():
    # Set paths
    data_dir = "data"
    output_dir = "models"
    rockyou_path = os.path.join(data_dir, "rockyou.txt")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    passwords = load_rockyou_data(rockyou_path, sample_size=1000000)  # Sample 1M passwords
    df = create_password_dataframe(passwords)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Generate vocabulary
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    char_to_idx_path = os.path.join(output_dir, "char_to_idx.pkl")
    generate_vocabulary(passwords, output_dir)
    
    # Load vocabulary and char_to_idx mapping
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(char_to_idx_path, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    # Initialize models
    print("Initializing models...")
    transformer = PasswordTransformer(
        vocab_size=len(vocab),
        embedding_dim=64,
        num_heads=4,
        ff_dim=128,
        max_length=30,
        num_transformer_blocks=2
    )
    
    lstm = PasswordLSTM(
        vocab_size=len(vocab),
        embedding_dim=64,
        lstm_units=128,
        max_length=30
    )
    
    # Prepare data for training
    print("Preparing data for training...")
    X_train = tokenize_passwords(train_df['password'].values, char_to_idx)
    y_train = train_df['strength'].values
    
    X_val = tokenize_passwords(val_df['password'].values, char_to_idx)
    y_val = val_df['strength'].values
    
    # Train transformer model
    print("Training transformer model...")
    transformer.train_strength_model(
        X_train, y_train,
        X_val, y_val,
        batch_size=128,
        epochs=10,
        save_path=os.path.join(output_dir, "transformer_model.keras")
    )
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm.train_strength_model(
        X_train, y_train,
        X_val, y_val,
        batch_size=128,
        epochs=10,
        save_path=os.path.join(output_dir, "lstm_model.keras")
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 