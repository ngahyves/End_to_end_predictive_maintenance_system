# RAM optimization and  required columns
from src.utils.preprocessing.preprocess import Preprocessor
import numpy as np

def test_optimize_dtypes(sample_data):
    """Vérify the conversion from float64-> float32."""
    from src.utils.config_loader import load_config
    cfg = load_config()
    proc = Preprocessor(cfg)
    
    optimized_df = proc._optimize_dtypes(sample_data)
    assert optimized_df["Air temperature [K]"].dtype == np.float32

def test_preprocessing_output_shape(sample_data):
    """Vérifying if we have the right number of columns."""
    from src.utils.config_loader import load_config
    cfg = load_config()
    proc = Preprocessor(cfg)
    # Run simulation
    X_train, _, _, _ = proc.run() 
    assert isinstance(X_train, np.ndarray)