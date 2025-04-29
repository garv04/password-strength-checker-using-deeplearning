import os
import requests
import gzip
import shutil
from tqdm import tqdm
import sys

def download_file(url, local_filename):
    """
    Download a file from URL with progress bar
    
    Args:
        url (str): URL to download from
        local_filename (str): Local path to save file
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_filename) as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

def extract_gzip(gzip_path, output_path):
    """
    Extract a gzip file
    
    Args:
        gzip_path (str): Path to gzip file
        output_path (str): Path to extract to
    """
    with gzip.open(gzip_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the gzip file after extraction
    os.remove(gzip_path)

def check_rockyou_exists(data_dir):
    """
    Check if the rockyou.txt file exists
    
    Args:
        data_dir (str): Directory to check
        
    Returns:
        bool: True if file exists, False otherwise
    """
    rockyou_path = os.path.join(data_dir, "rockyou.txt")
    return os.path.exists(rockyou_path)

def main():
    """
    Main function to download and extract rockyou.txt if not already available
    """
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "data")
    rockyou_path = os.path.join(data_dir, "rockyou.txt")
    rockyou_gz_path = os.path.join(data_dir, "rockyou.txt.gz")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if rockyou.txt already exists
    if check_rockyou_exists(data_dir):
        print(f"RockYou dataset already exists at: {rockyou_path}")
        print("No download needed.")
        return True
    
    print("RockYou dataset not found. You need to download it manually.")
    print(f"Expected location: {rockyou_path}")
    print("You can:")
    print("1. Place an existing rockyou.txt file in the data directory")
    print("2. Download it from another source (e.g., https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt)")
    print("3. Run this script with '--download' flag to attempt automatic download")
    
    # Check if download flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == '--download':
        try:
            # URL for the rockyou.txt.gz file
            url = "https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt.gz"
            
            print(f"Downloading RockYou dataset from {url}...")
            download_file(url, rockyou_gz_path)
            
            print("Extracting RockYou dataset...")
            extract_gzip(rockyou_gz_path, rockyou_path)
            
            print(f"RockYou dataset downloaded and extracted to {rockyou_path}")
            return True
        except Exception as e:
            print(f"Error downloading RockYou dataset: {e}")
            return False
    
    return False

if __name__ == "__main__":
    main() 