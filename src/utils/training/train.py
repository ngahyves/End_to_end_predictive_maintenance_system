import joblib
import mlflow
import mlflow.sklearn
import optuna
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score, average_precision_score

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Initialize professional logging
logger = get_logger("Training")

def get_model(name, params):
    """
    Factory function for Multi-label classification.
    """
    models = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "svm": SVC
    }
    
    # Probability is required for Average Precision score
    if name == "svm":
        params["probability"] = True
    
    # Wrap the selected model for Multi-output support
    return MultiOutputClassifier(models[name](**params))

def run_workflow():
    """
    Complete MLOps Pipeline: 
    1. Model Comparison -> 2. Optuna Tuning -> 3. Registry & Artifacts
    """
    cfg = load_config()
    
    # MLflow Setup
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    
    # 1. Load data
    data_path = Path(cfg["paths"]["processed_data_dir"]) / "data_processed.joblib"
    X_train, X_test, y_train, y_test = joblib.load(data_path)
    
    cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

    # --- STAGE 1: BASELINE COMPARISON ---
    logger.info("STAGE 1: Comparing Base Models...")
    best_name, best_score = None, 0
    
    for candidate in cfg["models"]["candidates"]:
        name = candidate["type"]
        params = candidate["params"]
        
        with mlflow.start_run(run_name=f"Baseline_{name}", nested=True):
            model = get_model(name, params)
            scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro')
            mean_f1 = np.mean(scores)
            
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1_macro", mean_f1)
            logger.info(f"Model: {name} | CV F1 Macro: {mean_f1:.4f}")
            
            if mean_f1 > best_score:
                best_score, best_name = mean_f1, name

    # --- STAGE 2: HYPERPARAMETER TUNING (OPTUNA) ---
    logger.info(f"Stage 1 Winner: {best_name}. Starting Optuna optimization...")
    space = cfg["models"]["search_space"].get(best_name, {})

    def objective(trial):
        params = {}
        if "n_estimators" in space:
            params["n_estimators"] = trial.suggest_int("n_estimators", space["n_estimators"][0], space["n_estimators"][1])
        if "learning_rate" in space:
            params["learning_rate"] = trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1])
        if "max_depth" in space:
            params["max_depth"] = trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1])
        if "C" in space:
            params["C"] = trial.suggest_float("C", space["C"][0], space["C"][1])

        model = get_model(best_name, params)
        return np.mean(cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro'))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # --- STAGE 3: FINAL TRAINING, REGISTRY & ARTIFACTS ---
    logger.info(f"STAGE 3: Final Evaluation for {best_name}")
    with mlflow.start_run(run_name=f"FINAL_CHAMPION_{best_name}"):
        champion = get_model(best_name, study.best_params)
        champion.fit(X_train, y_train)
        
        # Performance Evaluation
        y_pred = champion.predict(X_test)
        y_proba = champion.predict_proba(X_test)
        
        final_f1 = f1_score(y_test, y_pred, average='macro')
        final_ap = np.mean([average_precision_score(y_test.iloc[:, i], y_proba[i][:, 1]) for i in range(5)])
        
        # Logging Metadata
        mlflow.log_params(study.best_params)
        mlflow.log_metric("test_f1_macro", final_f1)
        mlflow.log_metric("test_avg_precision", final_ap)
        
        # 1. LOG MODEL TO REGISTRY (This creates the 'Models' tab in MLflow)
        mlflow.sklearn.log_model(
            sk_model=champion,
            artifact_path="model",
            registered_model_name="maintenance_prod_model"
        )
        
        # 2. LOG PREPROCESSOR AS ARTIFACT
        mlflow.log_artifact(local_path=cfg["paths"]["processor_path"], artifact_path="preprocessor")
        
        # Save local copy as backup
        save_path = Path(cfg["paths"]["model_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion, save_path)
        
        logger.info(f"FINAL TEST RESULTS | F1 Macro: {final_f1:.4f} | Avg Precision: {final_ap:.4f}")
        logger.info(f"Champion Model registered and saved at: {save_path}")

if __name__ == "__main__":
    try:
        run_workflow()
    except Exception as e:
        logger.error(f"Training Pipeline failed: {e}")
        raise