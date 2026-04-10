#Global explainability

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import mlflow
import os
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Initialize professional logging
logger = get_logger("Global_Explainability")

class ModelExplainer:
    def __init__(self, config):
        self.config = config
        self.model_path = Path(config["paths"]["model_path"])
        self.preprocessor_path = Path(config["paths"]["processor_path"])
        self.target_cols = config["features"]["targets"]
        
        #  Set up absolute path for MLflow tracking
        root_path = os.path.abspath(os.getcwd())
        tracking_uri = f"file:///{root_path}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        logger.info(f"MLflow Tracking URI set to: {tracking_uri}")

    def get_feature_names(self, preprocessor):
        """
        Extract feature names from the ColumnTransformer.
        Essential for readable SHAP plots.
        """
        # Get names from numeric and categorical transformers
        cat_features = preprocessor.transformers_[1][1]['encoder'].get_feature_names_out()
        num_features = self.config["features"]["numerical"]
        return list(num_features) + list(cat_features)

    def run_explanation(self):
        logger.info("Loading model and preprocessor for SHAP analysis...")
        model = joblib.load(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)
        
        # Load processed test data to explain performance on unseen data
        processed_dir = Path(self.config["paths"]["processed_data_dir"])
        _, X_test, _, _ = joblib.load(processed_dir / "data_processed.joblib")
        
        feature_names = self.get_feature_names(preprocessor)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # We start an MLflow run to log XAI (Explainable AI) artifacts
        with mlflow.start_run(run_name="SHAP_XAI_Analysis"):
            # Since it's a MultiOutputClassifier, we explain each failure type separately
            for i, target in enumerate(self.target_cols):
                logger.info(f"Computing SHAP values for failure type: {target}")
                
                # Get the internal model (LightGBM/XGBoost) for this specific target
                base_model = model.estimators_[i]
                
                # TreeExplainer is highly optimized for boosting models
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_test_df)

                # Generate Summary Plot (Feature Importance)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_df, show=False)
                
                # Create artifacts directory if missing
                Path("artifacts").mkdir(exist_ok=True)
                plot_path = f"artifacts/shap_summary_{target}.png"
                
                # Save plot locally
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                
                # --- Logging the image to MLflow ---
                mlflow.log_artifact(plot_path, artifact_path="explanations")
                logger.info(f"SHAP report for {target} successfully logged to MLflow.")

if __name__ == "__main__":
    try:
        cfg = load_config()
        explainer = ModelExplainer(cfg)
        explainer.run_explanation()
    except Exception as e:
        logger.error(f"Explainability module failed: {e}")
        raise