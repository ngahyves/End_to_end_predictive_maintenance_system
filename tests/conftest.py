# create a data set for tests
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    """Generate a valid data set for tests."""
    return pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["M14860", "L47181"],
        "Type": ["M", "L"],
        "Air temperature [K]": [298.1, 298.2],
        "Process temperature [K]": [308.6, 308.7],
        "Rotational speed [rpm]": [1551, 1408],
        "Torque [Nm]": [42.8, 46.3],
        "Tool wear [min]": [0, 3],
        "Machine failure": [0, 0],
        "TWF": [0, 0], "HDF": [0, 0], "PWF": [0, 0], "OSF": [0, 0], "RNF": [0, 0]
    })