from prefect import task, flow
from src.utils.ingestion.ingest import DataIngestor
from src.validation.validate import DataValidator
from src.utils.preprocessing.preprocess import Preprocessor
from src.utils.training.train import run_workflow
from src.monitoring import DriftMonitor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("Orchestrator")

@task(name="Ingestion")
def ingestion_step(cfg):
    return DataIngestor(cfg["paths"]["data_url"], cfg["paths"]["raw_data_path"]).execute()

@task(name="Validation")
def validation_step(df):
    return DataValidator().execute(df)

@task(name="Drift Check")
def check_for_drift(cfg):
    """Returns True if the model needs re-training."""
    monitor = DriftMonitor(cfg)
    return monitor.run_drift_analysis()

@task(name="Re-training Pipeline")
def retrain_model(cfg):
    """Executes Preprocessing + Training."""
    # 1. Preprocess fresh data
    preprocessor = Preprocessor(cfg)
    preprocessor.run()
    # 2. Train new model
    run_workflow()
    return "Model updated in Registry"

@flow(name="Autonomous_Maintenance_Pipeline")
def main_flow():
    cfg = load_config()
    
    # 1. Ingest
    df = ingestion_step(cfg)
    
    # 2. Validate
    validation_step(df)
    
    # 3. Check if the data has changed (Drift)
    needs_retraining = check_for_drift(cfg)
    
    # 4. CONDITIONAL LOGIC 
    if needs_retraining:
        logger.critical(" DRIFT DETECTED: Starting automated re-training...")
        retrain_model(cfg)
    else:
        logger.info(" NO DRIFT: Model remains stable. Training skipped.")

if __name__ == "__main__":
    main_flow()