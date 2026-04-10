from src.validation.validate import DataValidator
import pandera as pa
import pytest

def test_validation_success(sample_data):
    """Vérifying if a clean data set pass the validation."""
    validator = DataValidator()
    # If no Exceptions, it is okay
    validator.validate(sample_data)

def test_validation_failure_outlier(sample_data):
    """Vérifying if wrong values are blocked."""
    # We try with a outlier value for temperature (5000 Kelvin)
    sample_data.loc[0, "Air temperature [K]"] = 5000.0
    validator = DataValidator()
    
    with pytest.raises(pa.errors.SchemaErrors):
        validator.validate(sample_data)