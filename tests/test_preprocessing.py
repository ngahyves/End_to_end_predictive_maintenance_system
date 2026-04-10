import pytest
import pandas as pd
import numpy as np
from src.utils.preprocessing.preprocess import Preprocessor
from src.utils.config_loader import load_config

def test_preprocessing_output_shape(sample_data, tmp_path):
    """Verify data shape without the real csv."""
    # 1. load config
    cfg = load_config()

    # 2. Create a Mocking the disk)
    temp_data_dir = tmp_path / "data" / "raw"
    temp_data_dir.mkdir(parents=True)
    temp_csv_path = temp_data_dir / "ai4i2020.csv"
    
    sample_data.to_csv(temp_csv_path, index=False)

    # 3. Analysing the mocking file
    cfg["paths"]["raw_data_path"] = str(temp_csv_path)
    cfg["paths"]["processed_data_dir"] = str(tmp_path / "processed")
    cfg["paths"]["processor_path"] = str(tmp_path / "preprocessor.joblib")

    # 4. Processor execution
    proc = Preprocessor(cfg)
    X_train, X_test, y_train, y_test = proc.run()

    # 5. Assertion
    assert isinstance(X_train, np.ndarray)
    assert X_train.shape[1] > 0  # Verification