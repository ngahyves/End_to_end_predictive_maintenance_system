#Import the libaries and modules
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

#Calling the preprocessing logger
logger = get_logger("Preprocessing")

#Prepocessing class
class Preprocessor:
    def __init__(self, config):
        self.config = config
        #Using the features of config.yaml file
        self.target_cols = config["features"]["targets"]
        self.num_features = config["features"]["numerical"]
        self.cat_features = config["features"]["categorical"]
        #Processing paths
        self.processor_path = Path(config["paths"]["processor_path"])
        self.processed_dir = Path(config["paths"]["processed_data_dir"])

    #Data cleaning
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removing duplicates and unecessary columns"""
        df = df.drop_duplicates()
        cols_to_drop = ["UDI", "Product ID", "Machine failure"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        return df
    
    #Optimizing the memory by downcasting
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimisation of memory footprint (~50% reduction)."""
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"RAM optimized : {start_mem:.2f}MB -> {end_mem:.2f}MB")
        return df

    #Scikit-Learn pipeline
    def build_pipeline(self) -> ColumnTransformer:
        """Scikit-Learn pipeline."""
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        return ColumnTransformer([
            ("num", num_transformer, self.num_features),
            ("cat", cat_transformer, self.cat_features)
        ])

    #Workflow execution
    def run(self):
        # 1. Loading
        df = pd.read_csv(self.config["paths"]["raw_data_path"])
        
        # 2. Cleaning and optimization
        df = self.clean_data(df)
        df = self._optimize_dtypes(df)

        # 3. Spliting the data set
        X = df.drop(columns=self.target_cols)
        y = df[self.target_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Transformation pipeline 
        pipeline = self.build_pipeline()
        
        # Learning on train set/applying on test set
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        # 5. Saving the pipeline
        self.processor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.processor_path)
        logger.info(f"Pipeline saved : {self.processor_path}")

        # 6. Saving data processed for training
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            (X_train_transformed, X_test_transformed, y_train, y_test), 
            self.processed_dir / "data_processed.joblib"
        )
        logger.info("Train/test data saved for modeling.")

        return X_train_transformed, X_test_transformed, y_train, y_test

if __name__ == "__main__":
    cfg = load_config()
    preprocessor = Preprocessor(cfg)
    preprocessor.run()