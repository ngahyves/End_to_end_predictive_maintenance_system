import pytest
from src.utils.ingestion.ingest import DataIngestor, IngestionError

def test_verify_checksum_invalid(tmp_path):
    """Vérifying if Hash mismatch is detected."""
    # Create a virtual file
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("content")
    
    ingestor = DataIngestor(data_url="http://fake.com", save_path=str(p), expected_hash="wrong_hash")
    assert ingestor._verify_checksum() is False

def test_ingestion_wrong_url():
    """Verifying HTTP errors"""
    ingestor = DataIngestor(data_url="http://unsafe.com", save_path="dummy.csv")
    with pytest.raises(IngestionError, match="HTTPS required"):
        ingestor.download_data()