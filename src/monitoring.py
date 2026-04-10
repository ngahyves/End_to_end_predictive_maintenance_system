import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("Monitoring_Service")

class DriftMonitor:
    def __init__(self, config):
        """
        Initialize the Drift Monitor with project configurations.
        """
        self.config = config
        self.raw_data_path = Path(config["paths"]["raw_data_path"])
        self.report_path = Path("artifacts/drift_report.html")
        self.features = config["features"]["numerical"]

    def load_datasets(self):
        """
        Load reference and current datasets.
        In production, 'current' would come from logs or a database.
        """
        logger.info("Loading datasets for drift analysis...")

        df = pd.read_csv(self.raw_data_path)

        # Reference dataset (baseline)
        reference_data = df[self.features].sample(2000, random_state=42)

        # Current dataset (simulated production)
        current_data = df[self.features].sample(2000, random_state=7)

        # Artificial drift for demonstration: forced increase
        logger.warning("Simulating drift on 'Air temperature [K]' (+5.0)")
        current_data["Air temperature [K]"] += 5.0

        return reference_data, current_data

    def run_drift_analysis(self) -> bool:
        """
        Run Evidently drift analysis and generate HTML report.
        Returns:
            True if drift_share > threshold (Retraining needed)
            False otherwise
        """
        logger.info("Starting Evidently drift analysis...")

        try:
            # 1. Load datasets
            reference, current = self.load_datasets()

            # 2. Build and run Evidently report
            drift_report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])
            drift_report.run(reference_data=reference, current_data=current)

            # 3. Extract drift share metric
            report_dict = drift_report.as_dict()
            drift_share = report_dict["metrics"][0]["result"]["drift_share"]
            threshold = self.config["monitoring"]["drift_threshold"]

            logger.info(f"Drift Share detected: {drift_share:.2%}")

            # 4. Save HTML report for human auditing
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            drift_report.save_html(str(self.report_path))
            logger.info(f"HTML report saved at: {self.report_path}")

            # 5. Return decision to the orchestrator (Prefect)
            if drift_share > threshold:
                logger.critical(f"ALERT: Drift detected ({drift_share:.2%} > {threshold:.2%})")
                return True # Signal for retraining
            
            logger.info(" Data distribution stable. No action required.")
            return False

        except Exception as e:
            logger.error(f"Failed to execute drift analysis: {e}")
            raise

if __name__ == "__main__":
    try:
        cfg = load_config()
        monitor = DriftMonitor(cfg)
        # For manual testing, it will just print the result
        needs_retrain = monitor.run_drift_analysis()
        print(f"Retraining Required: {needs_retrain}")
    except Exception as e:
        logger.error(f"Monitoring module failed: {e}")