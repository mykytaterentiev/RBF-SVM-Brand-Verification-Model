"""
Evaluation module for RBF-SVM Brand Verification Model.

This module provides comprehensive model evaluation including metrics,
confusion matrices, and detailed reporting with emphasis on imbalanced data.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from pathlib import Path
import json

from .config import EvaluationConfig
from .utils import create_directory

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Provides detailed evaluation metrics, visualizations, and reporting
    with special attention to imbalanced classification performance.
    """
    
    def __init__(self, results_dir: Optional[Path] = None) -> None:
        """
        Initialize ModelEvaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        self.evaluation_results: Dict[str, Any] = {}
    
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            sample_weights: Sample weights (optional)
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} dataset")
        
        results = {}
        
        # Basic classification metrics
        results['basic_metrics'] = self._calculate_basic_metrics(y_true, y_pred, sample_weights)
        
        # Detailed classification report
        results['classification_report'] = self._get_classification_report(y_true, y_pred, sample_weights)
        
        # Confusion matrix
        results['confusion_matrix'] = self._calculate_confusion_matrix(y_true, y_pred)
        
        # Probability-based metrics (if probabilities available)
        if y_proba is not None:
            results['probability_metrics'] = self._calculate_probability_metrics(y_true, y_proba)
        
        # Class-specific analysis
        results['class_analysis'] = self._analyze_class_performance(y_true, y_pred, sample_weights)
        
        # Store results
        self.evaluation_results[dataset_name] = results
        
        # Log summary
        self._log_evaluation_summary(results, dataset_name)
        
        return results
    
    def _calculate_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_weights: Sample weights
            
        Returns:
            Dictionary of basic metrics
        """
        metrics = {}
        
        # Calculate metrics for different averaging methods
        for avg_method in EvaluationConfig.AVERAGE_METHODS:
            if avg_method == 'binary':
                # For binary classification
                metrics[f'precision_{avg_method}'] = precision_score(
                    y_true, y_pred, average='binary', sample_weight=sample_weights, zero_division=0
                )
                metrics[f'recall_{avg_method}'] = recall_score(
                    y_true, y_pred, average='binary', sample_weight=sample_weights, zero_division=0
                )
                metrics[f'f1_{avg_method}'] = f1_score(
                    y_true, y_pred, average='binary', sample_weight=sample_weights, zero_division=0
                )
            else:
                # For macro and weighted averaging
                metrics[f'precision_{avg_method}'] = precision_score(
                    y_true, y_pred, average=avg_method, sample_weight=sample_weights, zero_division=0
                )
                metrics[f'recall_{avg_method}'] = recall_score(
                    y_true, y_pred, average=avg_method, sample_weight=sample_weights, zero_division=0
                )
                metrics[f'f1_{avg_method}'] = f1_score(
                    y_true, y_pred, average=avg_method, sample_weight=sample_weights, zero_division=0
                )
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        
        return metrics
    
    def _get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_weights: Sample weights
            
        Returns:
            Classification report as dictionary
        """
        target_names = ['not_verified', 'verified']
        
        report = classification_report(
            y_true, 
            y_pred,
            target_names=target_names,
            sample_weight=sample_weights,
            digits=EvaluationConfig.CLASSIFICATION_REPORT_DIGITS,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    def _calculate_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate and format confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix data
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if specified
        cm_normalized = None
        if EvaluationConfig.CONFUSION_MATRIX_NORMALIZE:
            cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        
        return {
            'matrix': cm.tolist(),
            'matrix_normalized': cm_normalized.tolist() if cm_normalized is not None else None,
            'labels': ['not_verified', 'verified']
        }
    
    def _calculate_probability_metrics(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate probability-based metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of probability metrics
        """
        metrics = {}
        
        # Extract probabilities for positive class
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = None
        
        # Average Precision (PR AUC)
        try:
            metrics['average_precision'] = average_precision_score(y_true, y_proba_pos)
        except ValueError as e:
            logger.warning(f"Could not calculate Average Precision: {e}")
            metrics['average_precision'] = None
        
        return metrics
    
    def _analyze_class_performance(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_weights: Sample weights
            
        Returns:
            Class-specific performance analysis
        """
        analysis = {}
        
        # Class distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        analysis['true_distribution'] = dict(zip(unique_true.astype(int), counts_true.astype(int)))
        analysis['pred_distribution'] = dict(zip(unique_pred.astype(int), counts_pred.astype(int)))
        
        # Per-class metrics
        analysis['per_class_metrics'] = {}
        
        for class_label in [0, 1]:  # not_verified, verified
            class_name = 'not_verified' if class_label == 0 else 'verified'
            
            # True positives, false positives, etc.
            tp = np.sum((y_true == class_label) & (y_pred == class_label))
            fp = np.sum((y_true != class_label) & (y_pred == class_label))
            fn = np.sum((y_true == class_label) & (y_pred != class_label))
            tn = np.sum((y_true != class_label) & (y_pred != class_label))
            
            analysis['per_class_metrics'][class_name] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'support': int(np.sum(y_true == class_label))
            }
        
        return analysis
    
    def _log_evaluation_summary(self, results: Dict[str, Any], dataset_name: str) -> None:
        """
        Log a summary of evaluation results.
        
        Args:
            results: Evaluation results
            dataset_name: Name of the dataset
        """
        logger.info(f"=== {dataset_name.upper()} EVALUATION SUMMARY ===")
        
        # Basic metrics
        basic_metrics = results['basic_metrics']
        logger.info(f"Accuracy: {basic_metrics['accuracy']:.4f}")
        logger.info(f"Precision (binary): {basic_metrics['precision_binary']:.4f}")
        logger.info(f"Recall (binary): {basic_metrics['recall_binary']:.4f}")
        logger.info(f"F1-score (binary): {basic_metrics['f1_binary']:.4f}")
        
        # Probability metrics
        if 'probability_metrics' in results:
            prob_metrics = results['probability_metrics']
            if prob_metrics['roc_auc'] is not None:
                logger.info(f"ROC AUC: {prob_metrics['roc_auc']:.4f}")
            if prob_metrics['average_precision'] is not None:
                logger.info(f"Average Precision: {prob_metrics['average_precision']:.4f}")
        
        # Class distribution
        class_analysis = results['class_analysis']
        logger.info(f"True distribution: {class_analysis['true_distribution']}")
        logger.info(f"Predicted distribution: {class_analysis['pred_distribution']}")
    
    def save_results(self, filepath: Optional[Path] = None) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            if self.results_dir is None:
                logger.warning("No results directory specified. Cannot save results.")
                return
            filepath = self.results_dir / "evaluation_results.json"
        
        # Create directory if needed
        create_directory(filepath.parent)
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = self._convert_to_serializable(self.evaluation_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(key): self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def compare_models(self, dataset_names: List[str]) -> Dict[str, Any]:
        """
        Compare performance across different datasets.
        
        Args:
            dataset_names: List of dataset names to compare
            
        Returns:
            Comparison results
        """
        if not all(name in self.evaluation_results for name in dataset_names):
            missing = [name for name in dataset_names if name not in self.evaluation_results]
            raise ValueError(f"Missing evaluation results for datasets: {missing}")
        
        comparison = {}
        
        # Compare basic metrics
        for metric in ['accuracy', 'precision_binary', 'recall_binary', 'f1_binary']:
            comparison[metric] = {}
            for dataset_name in dataset_names:
                comparison[metric][dataset_name] = self.evaluation_results[dataset_name]['basic_metrics'][metric]
        
        # Compare probability metrics if available
        prob_metrics = ['roc_auc', 'average_precision']
        for metric in prob_metrics:
            comparison[metric] = {}
            for dataset_name in dataset_names:
                if 'probability_metrics' in self.evaluation_results[dataset_name]:
                    comparison[metric][dataset_name] = self.evaluation_results[dataset_name]['probability_metrics'].get(metric)
        
        return comparison
    
    def get_summary_report(self) -> str:
        """
        Generate a text summary report of all evaluations.
        
        Returns:
            Formatted summary report
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report_lines = ["=== MODEL EVALUATION SUMMARY REPORT ===\n"]
        
        for dataset_name, results in self.evaluation_results.items():
            report_lines.append(f"Dataset: {dataset_name.upper()}")
            report_lines.append("-" * 50)
            
            # Basic metrics
            basic_metrics = results['basic_metrics']
            report_lines.append(f"Accuracy:           {basic_metrics['accuracy']:.4f}")
            report_lines.append(f"Precision (binary): {basic_metrics['precision_binary']:.4f}")
            report_lines.append(f"Recall (binary):    {basic_metrics['recall_binary']:.4f}")
            report_lines.append(f"F1-score (binary):  {basic_metrics['f1_binary']:.4f}")
            
            # Probability metrics
            if 'probability_metrics' in results:
                prob_metrics = results['probability_metrics']
                if prob_metrics['roc_auc'] is not None:
                    report_lines.append(f"ROC AUC:           {prob_metrics['roc_auc']:.4f}")
                if prob_metrics['average_precision'] is not None:
                    report_lines.append(f"Average Precision: {prob_metrics['average_precision']:.4f}")
            
            # Class distribution
            class_analysis = results['class_analysis']
            report_lines.append(f"True distribution:  {class_analysis['true_distribution']}")
            report_lines.append(f"Pred distribution:  {class_analysis['pred_distribution']}")
            report_lines.append("")
        
        return "\n".join(report_lines)
