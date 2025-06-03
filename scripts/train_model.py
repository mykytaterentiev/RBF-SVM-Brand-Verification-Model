"""Main training script for RBF-SVM brand verification model."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbf_svm.training.trainer import ModelTrainer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RBF-SVM for brand verification")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/300k.csv",
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--tune_hyperparameters", 
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--cv_folds", 
        type=int, 
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Test set proportion (if no predefined split)"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate data path
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(random_state=args.random_state)
        
        # Run training pipeline
        results = trainer.train_pipeline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            tune_hyperparameters=args.tune_hyperparameters,
            cv_folds=args.cv_folds,
            test_size=args.test_size
        )
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Training time: {results['training_time_seconds']:.2f} seconds")
        print(f"Output directory: {results['output_directory']}")
        print("\nModel Performance:")
        print(f"  Test F1-Score: {results['test_metrics']['f1_score']:.4f}")
        print(f"  Test Precision: {results['test_metrics']['precision']:.4f}")
        print(f"  Test Recall: {results['test_metrics']['recall']:.4f}")
        print(f"  Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
        print(f"  Test PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
        
        print("\nBest Hyperparameters:")
        for param, value in results['model_info']['best_parameters'].items():
            print(f"  {param}: {value}")
        
        print("\nData Statistics:")
        print(f"  Total samples: {results['data_info']['total_rows']}")
        print(f"  Features: {results['feature_count']}")
        print(f"  Label distribution: {results['data_info']['label_distribution']}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\nERROR: Training failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
