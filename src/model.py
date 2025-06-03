"""
Model module for RBF-SVM Brand Verification Model.

This module implements RBF-kernel SVM with hyperparameter tuning,
proper handling of class imbalance, and sample weighting.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from scipy.sparse import csr_matrix
import joblib
from pathlib import Path
import warnings

from .config import ModelConfig
from .utils import create_directory

logger = logging.getLogger(__name__)

class RBFSVMModel:
    """
    RBF-kernel SVM model for brand verification.
    
    Implements SVM with RBF kernel, hyperparameter tuning, and proper
    handling of imbalanced data through sample weighting.
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize RBF-SVM model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state or ModelConfig.RANDOM_STATE
        self.model: Optional[SVC] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_results: Optional[Dict[str, Any]] = None
        self.is_fitted: bool = False
        
        # Store training info
        self.training_info: Dict[str, Any] = {}
    
    def train(
        self, 
        X_train: csr_matrix, 
        y_train: np.ndarray, 
        sample_weights: np.ndarray,
        tune_hyperparameters: bool = True
    ) -> None:
        """
        Train the RBF-SVM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            sample_weights: Sample weights for handling imbalance
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        logger.info("Starting RBF-SVM model training")
        
        # Store training data info
        self.training_info = {
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'class_distribution': np.bincount(y_train),
            'weight_stats': {
                'mean': sample_weights.mean(),
                'std': sample_weights.std(),
                'min': sample_weights.min(),
                'max': sample_weights.max()
            }
        }
        
        if tune_hyperparameters:
            self._tune_hyperparameters(X_train, y_train, sample_weights)
        else:
            # Use default parameters
            self.best_params = {
                'C': ModelConfig.DEFAULT_C,
                'gamma': ModelConfig.DEFAULT_GAMMA
            }
        
        # Train final model with best parameters
        self._train_final_model(X_train, y_train, sample_weights)
        
        logger.info("Model training completed")
    
    def _tune_hyperparameters(
        self, 
        X_train: csr_matrix, 
        y_train: np.ndarray, 
        sample_weights: np.ndarray
    ) -> None:
        """
        Perform hyperparameter tuning using Grid Search CV.
        
        Args:
            X_train: Training features
            y_train: Training targets
            sample_weights: Sample weights
        """
        logger.info("Tuning hyperparameters using Grid Search CV")
        
        # Define parameter grid
        param_grid = {
            'C': ModelConfig.C_GRID,
            'gamma': ModelConfig.GAMMA_GRID
        }
        
        # Create base SVM model
        base_svm = SVC(
            kernel=ModelConfig.KERNEL,
            probability=ModelConfig.PROBABILITY,
            random_state=self.random_state,
            class_weight='balanced'  # Additional balancing beyond sample weights
        )
        
        # Create scorer - use F1 score for imbalanced data
        scorer = make_scorer(f1_score, average='binary')
        
        # Create cross-validation strategy
        cv_strategy = StratifiedKFold(
            n_splits=ModelConfig.CV_FOLDS,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_svm,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv_strategy,
            n_jobs=-1,
            verbose=1,
            error_score='raise'
        )
        
        # Fit with sample weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.cv_results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def _train_final_model(
        self, 
        X_train: csr_matrix, 
        y_train: np.ndarray, 
        sample_weights: np.ndarray
    ) -> None:
        """
        Train the final model with best parameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            sample_weights: Sample weights
        """
        logger.info("Training final model with best parameters")
        
        # Create final model
        self.model = SVC(
            kernel=ModelConfig.KERNEL,
            C=self.best_params['C'],
            gamma=self.best_params['gamma'],
            probability=ModelConfig.PROBABILITY,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # Train model
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        self.is_fitted = True
        
        # Store model info
        self.training_info.update({
            'n_support_vectors': self.model.n_support_.tolist(),
            'support_vectors_total': sum(self.model.n_support_),
            'final_params': {
                'C': self.model.C,
                'gamma': self.model.gamma
            }
        })
        
        logger.info(f"Model trained with {sum(self.model.n_support_)} support vectors")
    
    def predict(self, X: csr_matrix) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
            
        Raises:
            ValueError: If model hasn't been trained or doesn't support probabilities
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model doesn't support probability predictions")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': 'RBF-SVM',
            'is_fitted': self.is_fitted,
            'training_info': self.training_info,
            'best_params': self.best_params,
            'cv_results': self.cv_results
        }
        
        if self.is_fitted and self.model:
            info.update({
                'classes': self.model.classes_.tolist(),
                'n_support_vectors': self.model.n_support_.tolist(),
                'support_vectors_total': sum(self.model.n_support_)
            })
        
        return info
    
    def save_model(self, filepath: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        create_directory(filepath.parent)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'training_info': self.training_info,
            'cv_results': self.cv_results,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model and metadata
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.training_info = model_data['training_info']
        self.cv_results = model_data.get('cv_results')
        self.random_state = model_data.get('random_state', ModelConfig.RANDOM_STATE)
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_decision_function(self, X: csr_matrix) -> np.ndarray:
        """
        Get decision function values for the samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Decision function values
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before getting decision function")
        
        return self.model.decision_function(X)
