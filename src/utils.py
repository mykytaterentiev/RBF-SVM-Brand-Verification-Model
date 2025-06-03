"""
Utility functions for RBF-SVM Brand Verification Model.

This module provides common utility functions used across the project
for logging, file operations, and data processing.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import re

def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    from .config import LoggingConfig
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LoggingConfig.LOG_FORMAT,
        datefmt=LoggingConfig.DATE_FORMAT
    )

def create_directory(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path to create
    """
    path.mkdir(parents=True, exist_ok=True)

def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log basic information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        name: Name to use in log messages
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== {name} Info ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info("Missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            pct = (count / len(df)) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
    else:
        logger.info("No missing values found")
    
    # Data types - fix the groupby issue
    logger.info("Data types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype_name, count in dtype_counts.items():
        logger.info(f"  {dtype_name}: {count} columns")

def clean_text(text: Union[str, float, None]) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text