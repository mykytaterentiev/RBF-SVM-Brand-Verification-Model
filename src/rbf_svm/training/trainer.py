"""Model training orchestration for RBF-SVM brand verification."""

import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

from ..data.loader import DataLoader
from ..preprocessing.feature_engineer import FeatureEngineer
from ..models.rbf_svm import RBFSVMClassifier
from ..evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates the complete model training pipeline."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.data_loader = None
        self.feature_engineer = FeatureEngineer()
        self.model = RBFSVMClassifier(random_state=random_state)
        self.evaluator = ModelEvaluator()
        
        self.training_results = {}
        
    def train_pipeline(self, 
                      data_path: str,
                      output_dir: str = "models",
                      tune_hyperparameters: bool = True,
                      cv_folds: int = 5,
                      test_size: float = 0.2) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        
        Args:
            data_path: Path to training data
            output_dir: Directory to save model artifacts
            tune_hyperparameters: Whether to tune hyperparameters
            cv_folds: Number of CV folds for tuning
            test_size: Test set proportion (if no predefined split)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting complete training pipeline")
        start_time = datetime.now()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Load and validate data
            logger.info("Step 1: Loading and validating data")
            self.data_loader = DataLoader(data_path)
            df = self.data_loader.load_data()
            
            data_info = self.data_loader.get_data_info(df)
            quality_check = self.data_loader.validate_data_quality(df)
            
            logger.info(f"Data loaded: {data_info['total_rows']} rows")
            logger.info(f"Label distribution: {data_info['label_distribution']}")
            
            if not quality_check['is_valid']:
                logger.warning(f"Data quality issues: {quality_check['issues']}")
                logger.info(f"Recommendations: {quality_check['recommendations']}")
            
            # 2. Split data
            logger.info("Step 2: Splitting data")
            train_df, test_df = self.data_loader.split_data(
                df, test_size=test_size, random_state=self.random_state
            )
            
            # 3. Feature engineering
            logger.info("Step 3: Feature engineering")
            X_train, y_train, sample_weights_train = self.feature_engineer.fit_transform(train_df)
            X_test = self.feature_engineer.transform(test_df)
            y_test = self.feature_engineer._encode_labels(test_df['label_clean'])
            sample_weights_test = test_df['calculated_weights'].values
            
            feature_names = self.feature_engineer.get_feature_names()
            logger.info(f"Features engineered: {len(feature_names)} features")
            
            # 4. Train model
            logger.info("Step 4: Training RBF-SVM model")
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights_train,
                tune_hyperparameters=tune_hyperparameters,
                cv_folds=cv_folds
            )
            
            # 5. Evaluate model
            logger.info("Step 5: Evaluating model")
            train_metrics = self.evaluator.evaluate(
                self.model, X_train, y_train, sample_weights_train, "training"
            )
            test_metrics = self.evaluator.evaluate(
                self.model, X_test, y_test, sample_weights_test, "test"
            )
            
            # 6. Generate detailed evaluation
            detailed_evaluation = self.evaluator.detailed_evaluation(
                self.model, X_test, y_test, sample_weights_test, feature_names
            )
            
            # 7. Save model and artifacts
            logger.info("Step 6: Saving model and artifacts")
            self._save_artifacts(output_path, data_info, quality_check, 
                               feature_names, train_metrics, test_metrics, 
                               detailed_evaluation)
            
            # Compile results
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            self.training_results = {
                'training_time_seconds': training_time,
                'data_info': data_info,
                'quality_check': quality_check,
                'feature_count': len(feature_names),
                'model_info': self.model.get_model_info(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'detailed_evaluation': detailed_evaluation,
                'output_directory': str(output_path),
                'timestamp': end_time.isoformat()
            }
            
            logger.info(f"Training pipeline completed in {training_time:.2f} seconds")
            logger.info(f"Test F1-score: {test_metrics['f1_score']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def _save_artifacts(self, output_path: Path, data_info: Dict, quality_check: Dict,
                       feature_names: list, train_metrics: Dict, test_metrics: Dict,
                       detailed_evaluation: Dict) -> None:
        """Save all training artifacts."""
        
        # Save trained model
        model_path = output_path / "rbf_svm_model.joblib"
        self.model.save_model(str(model_path))
        
        # Save feature engineer
        feature_engineer_path = output_path / "feature_engineer.joblib"
        joblib.dump(self.feature_engineer, feature_engineer_path)
        
        # Save training metadata
        metadata = {
            'data_info': data_info,
            'quality_check': quality_check,
            'feature_names': feature_names,
            'model_info': self.model.get_model_info(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        metadata_path = output_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save detailed evaluation report
        eval_path = output_path / "evaluation_report.json"
        with open(eval_path, 'w') as f:
            json.dump(detailed_evaluation, f, indent=2, default=str)
        
        # Save feature importance
        if len(feature_names) > 0:
            feature_importance = self.model.get_feature_importance(feature_names)
            importance_path = output_path / "feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
        
        logger.info(f"All artifacts saved to {output_path}")
    
    def load_trained_model(self, model_dir: str) -> Tuple[RBFSVMClassifier, FeatureEngineer]:
        """
        Load a previously trained model and feature engineer.
        
        Args:
            model_dir: Directory containing saved model artifacts
            
        Returns:
            Tuple of (model, feature_engineer)
        """
        model_path = Path(model_dir)
        
        # Load model
        model_file = model_path / "rbf_svm_model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        loaded_model = RBFSVMClassifier()
        loaded_model.load_model(str(model_file))
        
        # Load feature engineer
        fe_file = model_path / "feature_engineer.joblib"
        if not fe_file.exists():
            raise FileNotFoundError(f"Feature engineer file not found: {fe_file}")
        
        feature_engineer = joblib.load(fe_file)
        
        logger.info(f"Model and feature engineer loaded from {model_path}")
        
        return loaded_model, feature_engineer
    
    def predict_on_new_data(self, data_path: str, model_dir: str, 
                           output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on new data using trained model.
        
        Args:
            data_path: Path to new data
            model_dir: Directory containing trained model
            output_path: Optional path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Making predictions on new data")
        
        # Load trained model
        model, feature_engineer = self.load_trained_model(model_dir)
        
        # Load and prepare new data
        data_loader = DataLoader(data_path)
        df = data_loader.load_data()
        
        # Transform features
        X = feature_engineer.transform(df)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Prepare results
        results_df = df.copy()
        results_df['predicted_label'] = ['verified' if p == 1 else 'not_verified' 
                                        for p in predictions]
        results_df['prediction_confidence'] = probabilities.max(axis=1)
        results_df['probability_verified'] = probabilities[:, 1]
        results_df['probability_not_verified'] = probabilities[:, 0]
        
        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        logger.info(f"Predictions completed for {len(results_df)} samples")
        
        return results_df
