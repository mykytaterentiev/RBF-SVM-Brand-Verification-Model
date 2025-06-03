"""Model evaluation with focus on imbalanced data metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_auc_score, 
    average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from typing import Dict, Any, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for imbalanced brand verification."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(self, model, X: np.ndarray, y: np.ndarray, 
                sample_weights: np.ndarray, dataset_name: str = "dataset") -> Dict[str, float]:
        """
        Comprehensive evaluation with imbalance-aware metrics.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            sample_weights: Sample weights
            dataset_name: Name for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} set ({len(X)} samples)")
        
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1_score': f1_score(y, y_pred, average='binary'),
            
            # Weighted metrics (account for sample weights)
            'weighted_f1': f1_score(y, y_pred, average='weighted', sample_weight=sample_weights),
            'weighted_precision': precision_score(y, y_pred, average='weighted', sample_weight=sample_weights),
            'weighted_recall': recall_score(y, y_pred, average='weighted', sample_weight=sample_weights),
            
            # Area under curves
            'roc_auc': roc_auc_score(y, y_proba),
            'pr_auc': average_precision_score(y, y_proba),
            
            # Class-specific metrics
            'precision_verified': precision_score(y, y_pred, pos_label=1),
            'recall_verified': recall_score(y, y_pred, pos_label=1),
            'precision_not_verified': precision_score(y, y_pred, pos_label=0),
            'recall_not_verified': recall_score(y, y_pred, pos_label=0),
        }
        
        # Confusion matrix analysis
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        })
        
        # Log key metrics
        logger.info(f"{dataset_name} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        
        return metrics
    
    def detailed_evaluation(self, model, X: np.ndarray, y: np.ndarray, 
                          sample_weights: np.ndarray, 
                          feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate detailed evaluation report.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            sample_weights: Sample weights
            feature_names: List of feature names
            
        Returns:
            Detailed evaluation dictionary
        """
        logger.info("Generating detailed evaluation report")
        
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Basic evaluation
        basic_metrics = self.evaluate(model, X, y, sample_weights, "detailed")
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Brand frequency analysis
        brand_freq_analysis = self._analyze_by_brand_frequency(
            X, y, y_pred, y_proba, sample_weights, feature_names
        )
        
        # Error analysis
        error_analysis = self._analyze_errors(X, y, y_pred, y_proba, feature_names)
        
        # Feature importance (if available)
        feature_importance = {}
        try:
            feature_importance = model.get_feature_importance(feature_names)
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
        
        # Model characteristics
        model_info = model.get_model_info()
        
        detailed_report = {
            'basic_metrics': basic_metrics,
            'classification_report': class_report,
            'brand_frequency_analysis': brand_freq_analysis,
            'error_analysis': error_analysis,
            'feature_importance': feature_importance,
            'model_info': model_info,
            'sample_count': len(X),
            'feature_count': len(feature_names)
        }
        
        return detailed_report
    
    def _analyze_by_brand_frequency(self, X: np.ndarray, y: np.ndarray, 
                                  y_pred: np.ndarray, y_proba: np.ndarray,
                                  sample_weights: np.ndarray, 
                                  feature_names: List[str]) -> Dict[str, Any]:
        """Analyze model performance by brand frequency segments."""
        
        # Find brand_freq feature
        brand_freq_idx = None
        for i, name in enumerate(feature_names):
            if 'brand_freq' in name.lower():
                brand_freq_idx = i
                break
        
        if brand_freq_idx is None:
            return {'error': 'brand_freq feature not found'}
        
        brand_freq = X[:, brand_freq_idx]
        
        # Create frequency segments
        freq_bins = [0, 0.01, 0.1, 0.5, 1.0]
        freq_labels = ['rare', 'uncommon', 'common', 'very_common']
        freq_segments = pd.cut(brand_freq, bins=freq_bins, labels=freq_labels, include_lowest=True)
        
        analysis = {}
        
        for segment in freq_labels:
            mask = freq_segments == segment
            if mask.sum() == 0:
                continue
                
            segment_metrics = {
                'count': int(mask.sum()),
                'accuracy': accuracy_score(y[mask], y_pred[mask]),
                'f1_score': f1_score(y[mask], y_pred[mask]),
                'precision': precision_score(y[mask], y_pred[mask]),
                'recall': recall_score(y[mask], y_pred[mask]),
                'avg_weight': float(sample_weights[mask].mean()),
                'label_distribution': pd.Series(y[mask]).value_counts().to_dict()
            }
            
            analysis[segment] = segment_metrics
        
        return analysis
    
    def _analyze_errors(self, X: np.ndarray, y: np.ndarray, 
                       y_pred: np.ndarray, y_proba: np.ndarray,
                       feature_names: List[str]) -> Dict[str, Any]:
        """Analyze prediction errors."""
        
        # Find different types of errors
        false_positives = (y == 0) & (y_pred == 1)
        false_negatives = (y == 1) & (y_pred == 0)
        correct_predictions = y == y_pred
        
        analysis = {
            'false_positive_count': int(false_positives.sum()),
            'false_negative_count': int(false_negatives.sum()),
            'correct_prediction_count': int(correct_predictions.sum()),
        }
        
        # Analyze confidence of errors
        if false_positives.sum() > 0:
            fp_confidence = y_proba[false_positives]
            analysis['false_positive_confidence'] = {
                'mean': float(fp_confidence.mean()),
                'median': float(np.median(fp_confidence)),
                'std': float(fp_confidence.std())
            }
        
        if false_negatives.sum() > 0:
            fn_confidence = 1 - y_proba[false_negatives]  # Confidence in wrong prediction
            analysis['false_negative_confidence'] = {
                'mean': float(fn_confidence.mean()),
                'median': float(np.median(fn_confidence)),
                'std': float(fn_confidence.std())
            }
        
        # High confidence errors (potentially problematic)
        high_conf_threshold = 0.8
        high_conf_fp = false_positives & (y_proba > high_conf_threshold)
        high_conf_fn = false_negatives & (y_proba < (1 - high_conf_threshold))
        
        analysis['high_confidence_errors'] = {
            'false_positives': int(high_conf_fp.sum()),
            'false_negatives': int(high_conf_fn.sum()),
            'total': int(high_conf_fp.sum() + high_conf_fn.sum())
        }
        
        return analysis
    
    def plot_evaluation_charts(self, model, X: np.ndarray, y: np.ndarray,
                             output_dir: str = "evaluation_plots") -> None:
        """
        Generate evaluation plots.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Verified', 'Verified'],
                   yticklabels=['Not Verified', 'Verified'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_proba)
        ap_score = average_precision_score(y, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP={ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction Confidence Distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_proba[y == 0], bins=30, alpha=0.7, label='Not Verified', color='red')
        plt.hist(y_proba[y == 1], bins=30, alpha=0.7, label='Verified', color='blue')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by True Label')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        correct_mask = y == y_pred
        plt.hist(y_proba[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(y_proba[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Prediction Correctness')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {output_path}")
