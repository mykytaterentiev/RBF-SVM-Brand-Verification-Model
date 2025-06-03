"""Prediction script for RBF-SVM brand verification model."""

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
            logging.StreamHandler()
        ]
    )


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make predictions with trained RBF-SVM")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to data for prediction"
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models",
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--output_path", 
        type=str,
        help="Path to save predictions (optional)"
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
    
    # Validate paths
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    if not Path(args.model_dir).exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    try:
        # Initialize trainer (for prediction functionality)
        trainer = ModelTrainer()
        
        # Make predictions
        results_df = trainer.predict_on_new_data(
            data_path=args.data_path,
            model_dir=args.model_dir,
            output_path=args.output_path
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PREDICTIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Processed {len(results_df)} samples")
        
        # Show prediction distribution
        pred_counts = results_df['predicted_label'].value_counts()
        print("\nPrediction Distribution:")
        for label, count in pred_counts.items():
            pct = (count / len(results_df)) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Show confidence statistics
        print(f"\nConfidence Statistics:")
        print(f"  Mean confidence: {results_df['prediction_confidence'].mean():.3f}")
        print(f"  Median confidence: {results_df['prediction_confidence'].median():.3f}")
        print(f"  Min confidence: {results_df['prediction_confidence'].min():.3f}")
        print(f"  Max confidence: {results_df['prediction_confidence'].max():.3f}")
        
        if args.output_path:
            print(f"\nResults saved to: {args.output_path}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        print(f"\nERROR: Prediction failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
