#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import streamlit.web.cli as streamlit_cli

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Determine if data needs to be downloaded
def check_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    rockyou_path = os.path.join(data_dir, "rockyou.txt")
    
    if not os.path.exists(rockyou_path):
        print("RockYou dataset not found. Please make sure it's in the data directory.")
        print("You can try running the download script manually:")
        print("python -m src.preprocessing.download_rockyou")
        return False
    else:
        print(f"RockYou dataset found at {rockyou_path}")
        return True

def main():
    """Main entry point for the application"""
    print("Starting Password Strength Evaluator...")
    
    # Check data
    if not check_data():
        return
    
    # Run Streamlit app
    print("Starting Streamlit app...")
    streamlit_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "app.py")
    
    # Use Streamlit CLI to run the app
    sys.argv = ["streamlit", "run", streamlit_file, "--server.port=8501"]
    streamlit_cli.main()

if __name__ == "__main__":
    main() 