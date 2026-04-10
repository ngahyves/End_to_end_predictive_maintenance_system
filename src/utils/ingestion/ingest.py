# ingest/py

#Importing libraries and modules
import requests
import pandas as pd
import hashlib
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

#Calling my logging configuration
logger = get_logger("Ingestion")

#Creating a class for errors
class IngestionError(Exception):
    """Custom exceptions for ingestion step"""
    pass

#Class for ingesting my working file
class DataIngestor:
    def __init__(self, data_url: str, save_path: str, expected_hash: Optional[str] = None):
        self.data_url = data_url
        self.save_path = Path(save_path)
        self.expected_hash = expected_hash

    # Verifying data integrity
    def _verify_checksum(self) -> bool:
        """Check whether the local file matches the expected hash"""
        if not self.expected_hash:
            logger.warning("No expected hash provided. Integrity check ignored.")
            return True
        
        sha256_hash = hashlib.sha256()
        try:
            with open(self.save_path, "rb") as f: #Streaming binary read
                # Read in 4KB chunks to avoid overloading the RAM
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest() == self.expected_hash
        except Exception as e:
            logger.error(f"Error while calculating the checksum : {e}")
            return False

    #Download data from url source
    def download_data(self, force_download: bool = False) -> None:
        """HTTPS verification"""
        if not self.data_url.startswith("https://"):
            raise IngestionError("HTTPS required for security.")
        """Download the file with support for streaming and network error handling."""
        if self.save_path.exists() and not force_download:# idempotency
            logger.info(f"File already exists at {self.save_path}. Download skipped.")
            return

        logger.info(f"Download started : {self.data_url}")
        try:
            with requests.get(self.data_url, timeout=30, stream=True) as response:
                response.raise_for_status()
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(self.save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536): # 64KB chunks
                        if chunk:
                            f.write(chunk)
            
            logger.info("Download completed successfully.")

        except requests.exceptions.HTTPError as e:
            raise IngestionError(f"HTTP Error (404, 500, etc.) : {e}")
        except requests.exceptions.Timeout:
            raise IngestionError("The server did not respond (Timeout).")
        except requests.exceptions.RequestException as e:
            raise IngestionError(f"Unspecified network error : {e}")

    #Load data as pandas data frame
    def load_as_dataframe(self) -> pd.DataFrame:
        """Load the CSV into a DataFrame with basic validation."""
        if not self.save_path.exists() or self.save_path.stat().st_size == 0:
            raise IngestionError("The file is missing or empty on the disk.")
        
        try:
            df = pd.read_csv(self.save_path, low_memory=False)
            # low_memory=False is crucial for stable dtypes in production
            logger.info(f"Data loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            raise IngestionError(f"Unable to read the CSV file : {e}")

    #Execution method
    def execute(self, force: bool = False) -> pd.DataFrame:
        """
        The principal orchestrator
        """
        # 1. Download if necessary
        self.download_data(force_download=force)

        # 2. Data integrity check
        if self.expected_hash and not self._verify_checksum():
            logger.warning("Hash mismatch détected ! Attempt at correction...")
            self.download_data(force_download=True)
            if not self._verify_checksum():
                raise IngestionError("Critical error: The downloaded file is corrupted (invalid hash).")

        # 3. Returning the data
        return self.load_as_dataframe()

# --- Execution block ---
if __name__ == "__main__":
    try:
        config = load_config()
        
        ingestor = DataIngestor(
            data_url=config["paths"]["data_url"],
            save_path=config["paths"]["raw_data_path"],
            expected_hash=config["paths"].get("data_hash") 
        )
        df = ingestor.execute(force=False)
        print(df.head())

    except IngestionError as e:
        logger.critical(f"Pipeline stopped : {e}")
    except Exception as e:
        logger.error(f"Unexpected error : {e}")