import joblib
from sklearn.multioutput import MultiOutputClassifier
from src.utils.config_loader import load_config

def test_model_type():
    """Vérify if the saved model is a MultiOutputClassifier."""
    cfg = load_config()
    model = joblib.load(cfg["paths"]["model_path"])
    assert isinstance(model, MultiOutputClassifier)
    # Do we have 5 outputs?
    assert len(model.estimators_) == 5