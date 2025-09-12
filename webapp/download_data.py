import os
import requests
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the project root assuming the script is in webapp/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# These are the destinations on the persistent volume
PERSISTENT_DATA_DIR = Path("/data")
LOCAL_FALLBACK_DIR = PROJECT_ROOT / "webapp" / "backend"

# Ensure directories exist
PERSISTENT_DATA_DIR.mkdir(exist_ok=True)
LOCAL_FALLBACK_DIR.mkdir(exist_ok=True)

# Define a single source of truth for data URLs
DATA_SOURCES = {
    "data_with_embeddings.json": "https://storage.googleapis.com/climateinaction/data_with_embeddings.json",
    "all_abstracts.parquet": "https://storage.googleapis.com/climateinaction/all_abstracts.parquet"
}

def download_file(url: str, destination: Path):
    """Downloads a file with progress, but only if it doesn't exist."""
    if destination.exists():
        logging.info(f"✅ File already exists at {destination}. Skipping download.")
        return True

    logging.info(f"📥 Downloading {url} to {destination}...")
    try:
        response = requests.get(url, stream=True, timeout=1800)  # 30 min timeout
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        # Log every 10%
                        if int(progress) % 10 == 0 and int(progress) != int((downloaded - len(chunk)) / total_size * 100):
                             logging.info(f"    -> Progress: {progress:.1f}% ({downloaded/(1024*1024):.0f}/{total_size/(1024*1024):.0f} MB)")
        
        logging.info(f"✅ Successfully downloaded {destination.name}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to download {url}: {e}")
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        return False

if __name__ == "__main__":
    logging.info("--- Starting Pre-boot Data Download ---")
    
    success = True
    for filename, url in DATA_SOURCES.items():
        # Try to download to the persistent volume first
        if not download_file(url, PERSISTENT_DATA_DIR / filename):
            # If that fails (e.g., permissions, not on Railway), try local fallback
            logging.warning(f"Could not download to persistent storage. Trying local fallback.")
            if not download_file(url, LOCAL_FALLBACK_DIR / filename):
                success = False

    if success:
        logging.info("--- ✅ All data files are present. ---")
    else:
        logging.error("--- ❌ One or more data files failed to download. The app might not function correctly. ---")
