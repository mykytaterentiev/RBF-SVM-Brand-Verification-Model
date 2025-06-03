"""
Main orchestration module for RBF-SVM Brand Verification Model.

This module coordinates the entire machine learning pipeline from data loading
to model training and evaluation, providing a complete end-to-end solution.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import time

# Import our custom modules
from .config import DATA_PATH, RESULTS_DIR, ModelConfig
from .utils import setup_logging, create_directory
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .model import RBFSVMModel
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class BrandVerificationPipeline:
    """
    Complete pipeline for brand verification model training and evaluation.
    
    Orchestrates data loading, feature engineering, preprocessing, model training,
    and evaluation in a structured and reproducible manner.
    """
    
    def __init__(
        self, 
        data_path: Path,
        results_dir: Path,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the brand verification pipeline.
        
        Args:
            data_path: Path to the dataset CSV file
            results_dir: Directory to save results and models
            random_state: Random state for reproducibility
        """
        self.data_path = data_path
        self.results_dir = results_dir
        self.random_state = random_state or ModelConfig.RANDOM_STATE
        
        # Create results directory
        create_directory(self.results_dir)
        
        # Initialize pipeline components
        self.data_loader = DataLoader(data_path)
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.model = RBFSVMModel(random_state=self.random_state)
        self.evaluator = ModelEvaluator(results_dir=self.results_dir)
        
        # Store pipeline results
        self.pipeline_results: Dict[str, Any] = {}
    
    def run_complete_pipeline(
        self, 
        tune_hyperparameters: bool = True,
        save_model: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete machine learning pipeline.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            save_model: Whether to save the trained model
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary containing pipeline results and metrics
        """
        logger.info("=== STARTING BRAND VERIFICATION PIPELINE ===")
        start_time = time.time()
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data")
            df = self.data_loader.load_data()
            data_summary = self.data_loader.get_data_summary(df)
            self.pipeline_results['data_summary'] = data_summary
            
            # Step 2: Feature engineering
            logger.info("Step 2: Feature engineering")
            features, target, feature_names = self.feature_engineer.fit_transform(df)
            self.pipeline_results['feature_info'] = {
                'n_features': features.shape[1],
                'feature_names': feature_names,
                'feature_sparsity': 1.0 - (features.nnz / (features.shape[0] * features.shape[1]))
            }
            
            # Step 3: Extract sample weights and prepare data
            logger.info("Step 3: Data preprocessing and splitting")
            sample_weights = self.preprocessor.extract_sample_weights(df)
            
            X_train, X_test, y_train, y_test, weights_train, weights_test = \
                self.preprocessor.prepare_data(features, target, sample_weights, df)
            
            preprocessing_info = self.preprocessor.get_preprocessing_info()
            self.pipeline_results['preprocessing_info'] = preprocessing_info
            
            # Step 4: Model training
            logger.info("Step 4: Model training")
            self.model.train(
                X_train, 
                y_train, 
                weights_train, 
                tune_hyperparameters=tune_hyperparameters
            )
            
            model_info = self.model.get_model_info()
            self.pipeline_results['model_info'] = model_info
            
            # Step 5: Model evaluation
            logger.info("Step 5: Model evaluation")
            
            # Evaluate on training set
            y_train_pred = self.model.predict(X_train)
            y_train_proba = self.model.predict_proba(X_train)
            
            train_results = self.evaluator.evaluate_model(
                y_train, y_train_pred, y_train_proba, weights_train, "train"
            )
            
            # Evaluate on test set
            y_test_pred = self.model.predict(X_test)
            y_test_proba = self.model.predict_proba(X_test)
            
            test_results = self.evaluator.evaluate_model(
                y_test, y_test_pred, y_test_proba, weights_test, "test"
            )
            
            self.pipeline_results['train_evaluation'] = train_results
            self.pipeline_results['test_evaluation'] = test_results
            
            # Step 6: Save results
            if save_model:
                model_path = self.results_dir / "rbf_svm_model.joblib"
                self.model.save_model(model_path)
                self.pipeline_results['model_path'] = str(model_path)
            
            if save_results:
                self.evaluator.save_results()
                
                # Save complete pipeline results
                pipeline_summary_path = self.results_dir / "pipeline_summary.json"
                self._save_pipeline_summary(pipeline_summary_path)
            
            # Calculate total runtime
            total_time = time.time() - start_time
            self.pipeline_results['total_runtime_seconds'] = total_time
            
            logger.info(f"=== PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS ===")
            
            # Print final summary
            self._print_final_summary()
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise
    
    def _save_pipeline_summary(self, filepath: Path) -> None:
        """
        Save a complete pipeline summary to JSON.
        
        Args:
            filepath: Path to save the summary
        """
        import json
        
        # Convert to serializable format
        serializable_results = self.evaluator._convert_to_serializable(self.pipeline_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Pipeline summary saved to {filepath}")
    
    def _print_final_summary(self) -> None:
        """Print a final summary of pipeline results."""
        print("\n" + "="*80)
        print("BRAND VERIFICATION MODEL - FINAL SUMMARY")
        print("="*80)
        
        # Data summary
        data_summary = self.pipeline_results['data_summary']
        print(f"Dataset: {data_summary['total_rows']} samples, {data_summary['total_columns']} columns")
        print(f"Target distribution: {data_summary['target_distribution']}")
        
        # Feature summary
        feature_info = self.pipeline_results['feature_info']
        print(f"Features: {feature_info['n_features']} total features")
        print(f"Feature sparsity: {feature_info['feature_sparsity']:.3f}")
        
        # Model summary
        model_info = self.pipeline_results['model_info']
        print(f"Model: {model_info['model_type']} with {model_info['support_vectors_total']} support vectors")
        if model_info['best_params']:
            print(f"Best parameters: {model_info['best_params']}")
        
        # Performance summary
        test_eval = self.pipeline_results['test_evaluation']
        train_eval = self.pipeline_results['train_evaluation']
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        print(f"{'Metric':<20} {'Train':<10} {'Test':<10}")
        print("-" * 40)
        
        metrics = ['accuracy', 'precision_binary', 'recall_binary', 'f1_binary']
        for metric in metrics:
            train_val = train_eval['basic_metrics'][metric]
            test_val = test_eval['basic_metrics'][metric]
            print(f"{metric.replace('_', ' ').title():<20} {train_val:<10.4f} {test_val:<10.4f}")
        
        # Probability metrics if available
        if 'probability_metrics' in test_eval:
            test_prob = test_eval['probability_metrics']
            train_prob = train_eval['probability_metrics']
            
            if test_prob['roc_auc'] is not None:
                print(f"{'ROC AUC':<20} {train_prob['roc_auc']:<10.4f} {test_prob['roc_auc']:<10.4f}")
            
            if test_prob['average_precision'] is not None:
                print(f"{'Avg Precision':<20} {train_prob['average_precision']:<10.4f} {test_prob['average_precision']:<10.4f}")
        
        print("\n" + "="*80)
        
        # Runtime
        runtime = self.pipeline_results['total_runtime_seconds']
        print(f"Total runtime: {runtime:.2f} seconds")
        print("="*80)

def main() -> None:
    """
    Main entry point for the brand verification pipeline.
    
    Handles command line arguments and runs the complete pipeline.
    """
    parser = argparse.ArgumentParser(
        description="RBF-SVM Brand Verification Model Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-path',
        type=Path,
        default=DATA_PATH,
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=RESULTS_DIR,
        help='Directory to save results and models'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning (use default parameters)'
    )
    
    parser.add_argument(
        '--no-save-model',
        action='store_true',
        help='Do not save the trained model'
    )
    
    parser.add_argument(
        '--no-save-results',
        action='store_true',
        help='Do not save evaluation results'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=ModelConfig.RANDOM_STATE,
        help='Random state for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate input file
    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = BrandVerificationPipeline(
            data_path=args.data_path,
            results_dir=args.results_dir,
            random_state=args.random_state
        )
        
        results = pipeline.run_complete_pipeline(
            tune_hyperparameters=not args.no_tune,
            save_model=not args.no_save_model,
            save_results=not args.no_save_results
        )
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
