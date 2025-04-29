import os
import pickle
from collections import Counter
from typing import Dict, List, Set, Optional, Tuple

def build_vocabulary(
    passwords: List[str], 
    min_freq: int = 1,
    special_tokens: Optional[Dict[str, int]] = None
) -> Dict[str, int]:
    """
    Build character vocabulary from list of passwords
    
    Args:
        passwords: List of password strings
        min_freq: Minimum frequency for a character to be included
        special_tokens: Dictionary of special tokens to be included with indices
                       (e.g., {'<PAD>': 0, '<UNK>': 1})
    
    Returns:
        Dictionary mapping characters to indices
    """
    # Count character frequencies
    char_counts = Counter()
    for password in passwords:
        char_counts.update(password)
    
    # Filter by minimum frequency
    valid_chars = {char for char, count in char_counts.items() if count >= min_freq}
    
    # Create vocabulary mapping
    char_to_idx = {}
    
    # Add special tokens first if provided
    if special_tokens:
        char_to_idx.update(special_tokens)
    
    # Add remaining characters
    idx = len(char_to_idx)
    for char in sorted(valid_chars):
        if char not in char_to_idx:
            char_to_idx[char] = idx
            idx += 1
    
    return char_to_idx

def save_vocabulary(vocab: Dict[str, int], output_path: str) -> None:
    """
    Save vocabulary to file
    
    Args:
        vocab: Dictionary mapping characters to indices
        output_path: Path to save the vocabulary file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save vocabulary
    with open(output_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    # Also save a human-readable version
    txt_path = os.path.splitext(output_path)[0] + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        for char, idx in sorted(vocab.items(), key=lambda x: x[1]):
            # Handle special characters for readability
            char_str = repr(char) if not char.isprintable() or char.isspace() else char
            f.write(f"{char_str}\t{idx}\n")

def get_vocabulary_stats(vocab: Dict[str, int]) -> Dict[str, int]:
    """
    Get statistics about the vocabulary
    
    Args:
        vocab: Dictionary mapping characters to indices
    
    Returns:
        Dictionary with statistics like size, etc.
    """
    return {
        "vocab_size": len(vocab),
        "special_tokens": sum(1 for c in vocab if not c.isalnum() and len(c) > 1),
        "digits": sum(1 for c in vocab if c.isdigit()),
        "lowercase": sum(1 for c in vocab if c.islower()),
        "uppercase": sum(1 for c in vocab if c.isupper()),
        "special_chars": sum(1 for c in vocab if not c.isalnum() and len(c) == 1)
    }

def create_default_vocabulary() -> Dict[str, int]:
    """
    Create a default vocabulary with common characters
    
    Returns:
        Dictionary mapping characters to indices
    """
    # Start with special tokens
    char_to_idx = {
        '<PAD>': 0,  # Padding token
        '<UNK>': 1,  # Unknown token
        '<SOS>': 2,  # Start of sequence
        '<EOS>': 3,  # End of sequence
    }
    
    # Add digits
    for i in range(10):
        char_to_idx[str(i)] = len(char_to_idx)
    
    # Add lowercase letters
    for c in 'abcdefghijklmnopqrstuvwxyz':
        char_to_idx[c] = len(char_to_idx)
    
    # Add uppercase letters
    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        char_to_idx[c] = len(char_to_idx)
    
    # Add common special characters used in passwords
    for c in '!@#$%^&*()-_+={}[]|\\:;"\'<>,.?/~`':
        char_to_idx[c] = len(char_to_idx)
    
    return char_to_idx

def augment_vocabulary_with_tokens(
    base_vocab: Dict[str, int], 
    new_tokens: Set[str]
) -> Dict[str, int]:
    """
    Add new tokens to an existing vocabulary
    
    Args:
        base_vocab: Existing vocabulary dictionary
        new_tokens: Set of new tokens to add
        
    Returns:
        Updated vocabulary dictionary
    """
    vocab = base_vocab.copy()
    next_idx = max(vocab.values()) + 1
    
    for token in sorted(new_tokens):
        if token not in vocab:
            vocab[token] = next_idx
            next_idx += 1
    
    return vocab

def map_vocabularies(
    source_vocab: Dict[str, int], 
    target_vocab: Dict[str, int]
) -> Dict[int, int]:
    """
    Create a mapping between two vocabularies
    
    Args:
        source_vocab: Source vocabulary mapping
        target_vocab: Target vocabulary mapping
        
    Returns:
        Dictionary mapping source indices to target indices
    """
    idx_map = {}
    
    for token, source_idx in source_vocab.items():
        if token in target_vocab:
            idx_map[source_idx] = target_vocab[token]
        else:
            # Map to unknown token if available, otherwise skip
            if '<UNK>' in target_vocab:
                idx_map[source_idx] = target_vocab['<UNK>']
    
    return idx_map

def get_char_categories(vocab: Dict[str, int]) -> Dict[str, List[int]]:
    """
    Group vocabulary indices by character categories
    
    Args:
        vocab: Dictionary mapping characters to indices
    
    Returns:
        Dictionary with character categories and their indices
    """
    categories = {
        "special_tokens": [],
        "digits": [],
        "lowercase": [],
        "uppercase": [],
        "special_chars": []
    }
    
    for char, idx in vocab.items():
        if len(char) > 1:  # Special token like <PAD>
            categories["special_tokens"].append(idx)
        elif char.isdigit():
            categories["digits"].append(idx)
        elif char.islower():
            categories["lowercase"].append(idx)
        elif char.isupper():
            categories["uppercase"].append(idx)
        else:
            categories["special_chars"].append(idx)
    
    return categories 