"""Main training script for RBF-SVM brand verification model."""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbf_svm.training.trainer import ModelTrainer


def setup_logging(log_level: str = "INFO", log_file: str = "training.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def get_default_data_path() -> str:
    """Get default data path, preferring dummy data if available."""
    # Check environment variable first
    env_path = os.getenv('DATA_PATH')
    if env_path and Path(env_path).exists():
        return env_path
    
    # Check for dummy data
    dummy_path = "data/dummy_300k.csv"
    if Path(dummy_path).exists():
        return dummy_path
    
    # Fallback to original (likely won't exist for users)
    return "data/300k.csv"


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RBF-SVM for brand verification")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=get_default_data_path(),
        help="Path to training data CSV file (default: auto-detect dummy data)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.getenv('MODEL_OUTPUT_DIR', 'models'),
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--tune_hyperparameters", 
        action="store_true",
        default=os.getenv('TUNE_HYPERPARAMETERS', 'false').lower() == 'true',
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--cv_folds", 
        type=int, 
        default=int(os.getenv('CV_FOLDS', '5')),
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=float(os.getenv('TEST_SIZE', '0.2')),
        help="Test set proportion (if no predefined split)"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=int(os.getenv('RANDOM_STATE', '42')),
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default=os.getenv('LOG_LEVEL', 'INFO'),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = os.getenv('LOG_FILE', 'training.log')
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # Check if using dummy data
    if 'dummy' in args.data_path.lower():
        logger.info("Using dummy dataset for training")
        print("🔧 TRAINING WITH DUMMY DATA")
        print("   This is synthetic data for testing purposes.")
        print("   Replace with real data for production use.\n")
    
    # Validate data path
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        print(f"\n❌ ERROR: Data file not found: {args.data_path}")
        print("\n💡 To generate dummy data for testing:")
        print("   python scripts/generate_dummy_data.py")
        print("\n💡 Or set DATA_PATH environment variable:")
        print("   export DATA_PATH=/path/to/your/data.csv")
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
