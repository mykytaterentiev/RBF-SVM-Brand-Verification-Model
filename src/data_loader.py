"""
Data loading and validation module for RBF-SVM Brand Verification Model.

This module handles CSV loading, schema validation, and basic data cleaning
with strict type annotations and comprehensive error handling.
"""

import logging
import pandas as pd
from typing import Dict, Any
from pathlib import Path

from .config import DataConfig
from .utils import log_dataframe_info

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader class for brand verification dataset.
    
    Handles CSV loading, schema validation, and basic preprocessing
    with emphasis on data quality and type safety.
    """
    
    def __init__(self, data_path: Path) -> None:
        """
        Initialize DataLoader with dataset path.
        
        Args:
            data_path: Path to the CSV dataset file
        """
        self.data_path = data_path
        self.required_columns = [
            DataConfig.BRAND_NAME_COL,
            DataConfig.BRAND_FREQ_COL,
            DataConfig.SNOWDROP_NAME_COL,
            DataConfig.ADDRESS_NORM_COL,
            DataConfig.COUNTRY_COL,
            DataConfig.CITY_COL,
            DataConfig.WEBSITE_COL,
            DataConfig.WEBSITE_MATCH_COL,
            DataConfig.TARGET_COL,
            DataConfig.CALCULATED_WEIGHTS_COL,
            DataConfig.SPLIT_COL
        ]
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the dataset from CSV.
        
        Returns:
            Validated DataFrame with proper data types
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing or data validation fails
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load CSV with appropriate data types
        try:
            df = pd.read_csv(
                self.data_path,
                dtype={
                    DataConfig.BRAND_NAME_COL: 'string',
                    DataConfig.SNOWDROP_NAME_COL: 'string',
                    DataConfig.ADDRESS_NORM_COL: 'string',
                    DataConfig.COUNTRY_COL: 'string',
                    DataConfig.CITY_COL: 'string',
                    DataConfig.WEBSITE_COL: 'string',
                    DataConfig.TARGET_COL: 'string',
                    DataConfig.SPLIT_COL: 'string'
                },
                na_values=['', 'nan', 'null', 'None', 'NA']
            )
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Validate schema
        self._validate_schema(df)
        
        # Basic data cleaning
        df = self._clean_data(df)
        
        # Log data info
        log_dataframe_info(df, "Loaded Dataset")
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validate that all required columns are present.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate target column values
        valid_targets = {DataConfig.VERIFIED_LABEL, DataConfig.NOT_VERIFIED_LABEL}
        unique_targets = set(df[DataConfig.TARGET_COL].dropna().unique())
        invalid_targets = unique_targets - valid_targets
        
        if invalid_targets:
            logger.warning(f"Found unexpected target values: {invalid_targets}")
        
        logger.info("Schema validation passed")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting basic data cleaning")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing brand names by using snowdrop_name as fallback
        brand_name_missing = df_clean[DataConfig.BRAND_NAME_COL].isna()
        logger.info(f"Found {brand_name_missing.sum()} missing brand names")
        
        df_clean.loc[brand_name_missing, DataConfig.BRAND_NAME_COL] = \
            df_clean.loc[brand_name_missing, DataConfig.SNOWDROP_NAME_COL]
        
        # Validate numeric columns
        numeric_columns = [
            DataConfig.BRAND_FREQ_COL,
            DataConfig.WEBSITE_MATCH_COL,
            DataConfig.CALCULATED_WEIGHTS_COL
        ]
        
        for col in numeric_columns:
            # Convert to numeric, coercing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Log conversion results
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Column {col}: {missing_count} values could not be converted to numeric")
        
        # Fill missing numeric values with configured defaults
        df_clean[DataConfig.BRAND_FREQ_COL] = df_clean[DataConfig.BRAND_FREQ_COL].fillna(
            DataConfig.FILL_MISSING_NUMERIC
        )
        df_clean[DataConfig.WEBSITE_MATCH_COL] = df_clean[DataConfig.WEBSITE_MATCH_COL].fillna(
            DataConfig.FILL_MISSING_NUMERIC
        )
        df_clean[DataConfig.CALCULATED_WEIGHTS_COL] = df_clean[DataConfig.CALCULATED_WEIGHTS_COL].fillna(1.0)
        
        # Fill missing text columns with empty string
        text_columns = [
            DataConfig.ADDRESS_NORM_COL,
            DataConfig.COUNTRY_COL,
            DataConfig.CITY_COL,
            DataConfig.WEBSITE_COL
        ]
        
        for col in text_columns:
            df_clean[col] = df_clean[col].fillna(DataConfig.FILL_MISSING_TEXT)
        
        # Remove rows with missing target values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=[DataConfig.TARGET_COL])
        removed_rows = initial_rows - len(df_clean)
        
        if removed_rows > 0:
            logger.warning(f"Removed {removed_rows} rows with missing target values")
        
        logger.info("Basic data cleaning completed")
        return df_clean
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data summary.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'target_distribution': df[DataConfig.TARGET_COL].value_counts().to_dict(),
            'split_distribution': df[DataConfig.SPLIT_COL].value_counts().to_dict() if DataConfig.SPLIT_COL in df.columns else None,
            'brand_freq_stats': {
                'mean': float(df[DataConfig.BRAND_FREQ_COL].mean()),
                'std': float(df[DataConfig.BRAND_FREQ_COL].std()),
                'min': float(df[DataConfig.BRAND_FREQ_COL].min()),
                'max': float(df[DataConfig.BRAND_FREQ_COL].max()),
                'median': float(df[DataConfig.BRAND_FREQ_COL].median())
            },
            'missing_values': df.isnull().sum().to_dict(),
            'unique_countries': df[DataConfig.COUNTRY_COL].nunique(),
            'unique_brands': df[DataConfig.BRAND_NAME_COL].nunique()
        }
        
        return summary
