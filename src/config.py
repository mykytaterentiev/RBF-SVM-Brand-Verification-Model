"""
Configuration module for RBF-SVM Brand Verification Model.

This module contains all configuration parameters, constants, and settings
used throughout the project for data processing, feature engineering,
model training, and evaluation.
"""

from pathlib import Path
from typing import Tuple

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Default data paths
DATA_PATH = DATA_DIR / "dummy_data_1k.csv"

class DataConfig:
    """Configuration for data loading and processing."""
    
    # Column names in the dataset
    BRAND_NAME_COL = "brand_name"
    BRAND_FREQ_COL = "brand_freq"
    SNOWDROP_NAME_COL = "snowdrop_name"
    ADDRESS_NORM_COL = "address_norm"
    COUNTRY_COL = "country"
    CITY_COL = "city"
    WEBSITE_COL = "website"
    WEBSITE_MATCH_COL = "website_match"
    TARGET_COL = "target"
    CALCULATED_WEIGHTS_COL = "calculated_weights"
    SPLIT_COL = "split"
    
    # Target values
    VERIFIED_LABEL = "verified"
    NOT_VERIFIED_LABEL = "not_verified"
    
    # Data cleaning defaults
    FILL_MISSING_TEXT = ""
    FILL_MISSING_NUMERIC = 0.0
    
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Text features to process with TF-IDF
    TEXT_FEATURES = [
        "brand_name_processed",
        "address_norm_processed",
        "combined_text"
    ]
    
    # Categorical features for one-hot encoding
    CATEGORICAL_FEATURES = [
        "country",
        "city"
    ]
    
    # Numerical features to include directly
    NUMERICAL_FEATURES = [
        "brand_freq",
        "website_match"
    ]
    
    # TF-IDF parameters
    MAX_FEATURES_TFIDF = 1000
    MIN_DF_TFIDF = 2
    MAX_DF_TFIDF = 0.8
    NGRAM_RANGE: Tuple[int, int] = (1, 2)

class ModelConfig:
    """Configuration for model training and hyperparameter tuning."""
    
    # Random state for reproducibility
    RANDOM_STATE = 42
    
    # SVM parameters
    KERNEL = "rbf"
    PROBABILITY = True
    
    # Default parameters (used when not tuning)
    DEFAULT_C = 1.0
    DEFAULT_GAMMA = "scale"
    
    # Hyperparameter tuning grids
    C_GRID = [0.1, 1.0, 10.0, 100.0]
    GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
    
    # Cross-validation settings
    CV_FOLDS = 5
    
    # Train/test split parameters
    TEST_SIZE = 0.2
    STRATIFY = True

class EvaluationConfig:
    """Configuration for model evaluation and metrics."""
    
    # Metrics to calculate
    AVERAGE_METHODS = ["binary", "macro", "weighted"]
    
    # Classification report settings
    CLASSIFICATION_REPORT_DIGITS = 4
    
    # Confusion matrix settings
    CONFUSION_MATRIX_NORMALIZE = True
    
    # Threshold for binary classification
    CLASSIFICATION_THRESHOLD = 0.5

class LoggingConfig:
    """Configuration for logging."""
    
    # Logging format
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Default logging level
    DEFAULT_LOG_LEVEL = "INFO"

# Global constants
MAX_WORKERS = 4  # For parallel processing
MEMORY_LIMIT_GB = 8  # Memory usage limit
