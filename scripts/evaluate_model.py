"""Evaluate trained RBF-SVM model with detailed analysis."""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbf_svm.training.trainer import ModelTrainer
from rbf_svm.evaluation.evaluator import ModelEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained RBF-SVM model")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data CSV file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Generate evaluation plots"
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
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        trainer = ModelTrainer()
        model, feature_engineer = trainer.load_trained_model(args.model_dir)
        
        # Load and prepare data
        df = pd.read_csv(args.data_path)
        logger.info(f"Loaded {len(df)} samples for evaluation")
        
        # Transform features
        X = feature_engineer.transform(df)
        y = feature_engineer._encode_labels(df['label_clean'])
        sample_weights = df['calculated_weights'].values
        
        # Evaluate model
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X, y, sample_weights, "evaluation")
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset: {args.data_path}")
        print(f"Samples: {len(df):,}")
        print(f"Features: {X.shape[1]}")
        
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
        
        # Brand frequency analysis
        if 'brand_freq' in df.columns:
            print("\nBrand Frequency Analysis:")
            df['predictions'] = model.predict(X)
            df['prediction_proba'] = model.predict_proba(X)[:, 1]
            
            # Group by brand frequency quartiles
            df['freq_quartile'] = pd.qcut(df['brand_freq'], q=4, labels=['Q1 (Rare)', 'Q2', 'Q3', 'Q4 (Common)'])
            freq_analysis = df.groupby('freq_quartile').agg({
                'label_clean': lambda x: (x == 'verified').mean(),
                'predictions': lambda x: (x == 1).mean(),
                'prediction_proba': 'mean'
            }).round(4)
            
            print(freq_analysis)
        
        # Save detailed results
        results_file = output_dir / "evaluation_metrics.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create plots if requested
        if args.create_plots:
            logger.info("Generating evaluation plots")
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Prediction confidence distribution
            axes[0, 0].hist(df['prediction_proba'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Prediction Confidence')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Prediction Confidence Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Brand frequency vs prediction confidence
            axes[0, 1].scatter(df['brand_freq'], df['prediction_proba'], alpha=0.5)
            axes[0, 1].set_xlabel('Brand Frequency')
            axes[0, 1].set_ylabel('Prediction Confidence')
            axes[0, 1].set_title('Brand Frequency vs Prediction Confidence')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, model.predict(X))
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], 
                       xticklabels=['Not Verified', 'Verified'],
                       yticklabels=['Not Verified', 'Verified'])
            axes[1, 0].set_title('Confusion Matrix')
            
            # Plot 4: Performance by brand frequency quartile
            if 'freq_quartile' in df.columns:
                freq_performance = df.groupby('freq_quartile')['prediction_proba'].mean()
                axes[1, 1].bar(range(len(freq_performance)), freq_performance.values)
                axes[1, 1].set_xticks(range(len(freq_performance)))
                axes[1, 1].set_xticklabels(freq_performance.index, rotation=45)
                axes[1, 1].set_ylabel('Average Prediction Confidence')
                axes[1, 1].set_title('Performance by Brand Frequency Quartile')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / "evaluation_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Plots saved to {plot_file}")
        
        # Save predictions
        predictions_file = output_dir / "predictions.csv"
        df[['brand_name', 'label_clean', 'predictions', 'prediction_proba', 'brand_freq']].to_csv(
            predictions_file, index=False
        )
        
        print(f"\nResults saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
