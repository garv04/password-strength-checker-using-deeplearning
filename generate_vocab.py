import os
from src.utils.vocab_utils import create_and_save_default_vocabulary

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Generate and save default vocabulary
    create_and_save_default_vocabulary("models")
    
    print("Vocabulary files generated successfully!")

if __name__ == "__main__":
    main() 