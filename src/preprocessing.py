"""
Preprocessing module for RBF-SVM Brand Verification Model.

This module handles feature scaling, train/test splitting, and data preparation
with emphasis on proper handling of sample weights and class imbalance.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

from .config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for brand verification model.
    
    Handles scaling, splitting, and sample weight extraction with proper
    treatment of sparse matrices and class imbalance.
    """
    
    def __init__(self) -> None:
        """Initialize DataPreprocessor."""
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
    
    def prepare_data(
        self, 
        features: csr_matrix, 
        target: np.ndarray, 
        sample_weights: np.ndarray,
        df: pd.DataFrame,
        use_predefined_split: bool = True
    ) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training including scaling and splitting.
        
        Args:
            features: Feature matrix (sparse)
            target: Target array
            sample_weights: Sample weights array
            df: Original DataFrame for split information
            use_predefined_split: Whether to use predefined split column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, weights_train, weights_test)
        """
        logger.info("Preparing data for model training")
        
        # Handle train/test splitting
        if use_predefined_split and DataConfig.SPLIT_COL in df.columns:
            X_train, X_test, y_train, y_test, weights_train, weights_test = \
                self._use_predefined_split(features, target, sample_weights, df)
        else:
            X_train, X_test, y_train, y_test, weights_train, weights_test = \
                self._create_stratified_split(features, target, sample_weights)
        
        # Scale features
        X_train_scaled = self._fit_transform_features(X_train)
        X_test_scaled = self._transform_features(X_test)
        
        # Log split information
        self._log_split_info(y_train, y_test, weights_train, weights_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, weights_train, weights_test
    
    def _use_predefined_split(
        self, 
        features: csr_matrix, 
        target: np.ndarray, 
        sample_weights: np.ndarray,
        df: pd.DataFrame
    ) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use predefined train/test split from the dataset.
        
        Args:
            features: Feature matrix
            target: Target array
            sample_weights: Sample weights
            df: DataFrame with split information
            
        Returns:
            Split data tuple
        """
        logger.info("Using predefined train/test split")
        
        split_col = df[DataConfig.SPLIT_COL].fillna('TRAIN')
        
        # Define training indices (TRAIN + VALIDATE as training)
        train_mask = split_col.isin(['TRAIN', 'VALIDATE']).to_numpy(dtype=bool)
        test_mask = (split_col == 'TEST').to_numpy(dtype=bool)
        
        # Apply masks
        X_train = features[train_mask]
        X_test = features[test_mask]
        y_train = target[train_mask]
        y_test = target[test_mask]
        weights_train = sample_weights[train_mask]
        weights_test = sample_weights[test_mask]
        
        # Log split sizes
        logger.info(f"Predefined split - Train: {len(y_train)}, Test: {len(y_test)}")
        
        return X_train, X_test, y_train, y_test, weights_train, weights_test
    
    def _create_stratified_split(
        self, 
        features: csr_matrix, 
        target: np.ndarray, 
        sample_weights: np.ndarray
    ) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/test split.
        
        Args:
            features: Feature matrix
            target: Target array
            sample_weights: Sample weights
            
        Returns:
            Split data tuple
        """
        logger.info("Creating stratified train/test split")
        
        # Use stratified split to maintain class distribution
        indices = np.arange(len(target))
        
        train_indices, test_indices = train_test_split(
            indices,
            test_size=ModelConfig.TEST_SIZE,
            random_state=ModelConfig.RANDOM_STATE,
            stratify=target if ModelConfig.STRATIFY else None
        )
        
        # Apply splits
        X_train = features[train_indices]
        X_test = features[test_indices]
        y_train = target[train_indices]
        y_test = target[test_indices]
        weights_train = sample_weights[train_indices]
        weights_test = sample_weights[test_indices]
        
        logger.info(f"Stratified split - Train: {len(y_train)}, Test: {len(y_test)}")
        
        return X_train, X_test, y_train, y_test, weights_train, weights_test
    
    def _fit_transform_features(self, X_train: csr_matrix) -> csr_matrix:
        """
        Fit scaler on training data and transform.
        
        Args:
            X_train: Training feature matrix
            
        Returns:
            Scaled training features
        """
        logger.info("Fitting and transforming training features")
        
        # For sparse matrices, we need to be careful with scaling
        # StandardScaler can handle sparse matrices but may make them dense
        
        # Convert to dense for scaling if matrix is small enough
        if X_train.shape[1] <= 10000:  # Arbitrary threshold
            X_train_dense = X_train.toarray()
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_dense)
            
            # Convert back to sparse if it was originally sparse
            X_train_scaled = csr_matrix(X_train_scaled)
            
        else:
            logger.warning("Large feature matrix detected. Scaling with sparse support.")
            
            # Use StandardScaler with sparse support
            self.scaler = StandardScaler(with_mean=False)  # Cannot center sparse matrices
            X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.is_fitted = True
        logger.info(f"Scaled training features shape: {X_train_scaled.shape}")
        
        return X_train_scaled
    
    def _transform_features(self, X_test: csr_matrix) -> csr_matrix:
        """
        Transform test features using fitted scaler.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Scaled test features
            
        Raises:
            ValueError: If scaler hasn't been fitted
        """
        if not self.is_fitted or self.scaler is None:
            raise ValueError("Scaler must be fitted before transforming test data")
        
        logger.info("Transforming test features")
        
        # Apply same transformation logic as training
        if X_test.shape[1] <= 10000:
            X_test_dense = X_test.toarray()
            X_test_scaled = self.scaler.transform(X_test_dense)
            X_test_scaled = csr_matrix(X_test_scaled)
        else:
            X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Scaled test features shape: {X_test_scaled.shape}")
        
        return X_test_scaled
    
    def extract_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and validate sample weights from DataFrame.
        
        Args:
            df: DataFrame containing sample weights
            
        Returns:
            Sample weights array
        """
        logger.info("Extracting sample weights")
        
        if DataConfig.CALCULATED_WEIGHTS_COL in df.columns:
            weights = pd.to_numeric(df[DataConfig.CALCULATED_WEIGHTS_COL], errors='coerce')
            
            # Handle missing or invalid weights
            weights = weights.fillna(1.0)
            
            # Ensure positive weights
            weights = np.maximum(weights.values, 0.1)  # Minimum weight of 0.1
            
            # Log weight statistics
            logger.info(f"Weight statistics - Mean: {weights.mean():.3f}, "
                       f"Std: {weights.std():.3f}, "
                       f"Min: {weights.min():.3f}, "
                       f"Max: {weights.max():.3f}")
            
            return weights
        else:
            logger.warning(f"Sample weights column {DataConfig.CALCULATED_WEIGHTS_COL} not found. Using uniform weights.")
            return np.ones(len(df))
    
    def _log_split_info(
        self, 
        y_train: np.ndarray, 
        y_test: np.ndarray, 
        weights_train: np.ndarray, 
        weights_test: np.ndarray
    ) -> None:
        """
        Log detailed information about the train/test split.
        
        Args:
            y_train: Training targets
            y_test: Test targets
            weights_train: Training weights
            weights_test: Test weights
        """
        # Class distribution
        train_dist = np.bincount(y_train)
        test_dist = np.bincount(y_test)
        
        logger.info("=== Split Information ===")
        logger.info(f"Training set size: {len(y_train)}")
        logger.info(f"Test set size: {len(y_test)}")
        
        if len(train_dist) >= 2:
            logger.info(f"Training - Verified: {train_dist[1]}, Not verified: {train_dist[0]}")
            logger.info(f"Training - Class ratio: {train_dist[1]/train_dist[0]:.3f}")
        
        if len(test_dist) >= 2:
            logger.info(f"Test - Verified: {test_dist[1]}, Not verified: {test_dist[0]}")
            logger.info(f"Test - Class ratio: {test_dist[1]/test_dist[0]:.3f}")
        
        # Weight distribution
        logger.info(f"Training weights - Mean: {weights_train.mean():.3f}, Std: {weights_train.std():.3f}")
        logger.info(f"Test weights - Mean: {weights_test.mean():.3f}, Std: {weights_test.std():.3f}")
        
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing pipeline.
        
        Returns:
            Dictionary with preprocessing information
        """
        info = {
            'scaler_fitted': self.is_fitted,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
        }
        
        if self.scaler and hasattr(self.scaler, 'mean_'):
            info['feature_means'] = self.scaler.mean_
            info['feature_scales'] = self.scaler.scale_
        
        return info
