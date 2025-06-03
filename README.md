# RBF-SVM Brand Verification Model

A clean, modular implementation of an RBF-kernel Support Vector Machine for brand verification with focus on long-tail sensitivity and imbalanced data handling.

## Features

- **Long-tail Sensitivity**: Uses calculated sample weights to prioritize rare brands (low brand_freq)
- **Imbalanced Data Handling**: Optimized for highly imbalanced verification datasets (~95% verified)
- **Comprehensive Feature Engineering**: Text features, brand encoding, website analysis
- **Hyperparameter Tuning**: Grid search optimization with F1-score focus
- **Detailed Evaluation**: Brand frequency analysis, error analysis, confidence metrics
- **Modular Design**: Clean separation of concerns for easy maintenance and extension

## Project Structure

```
rbf-svm-vertex/
├── src/rbf_svm/
│   ├── __init__.py
│   ├── data/
│   │   └── loader.py          # Data loading and validation
│   ├── preprocessing/
│   │   └── feature_engineer.py # Feature engineering pipeline
│   ├── models/
│   │   └── rbf_svm.py         # RBF-SVM implementation
│   ├── training/
│   │   └── trainer.py         # Training orchestration
│   └── evaluation/
│       └── evaluator.py       # Model evaluation
├── scripts/
│   ├── train_model.py         # Training script
│   └── predict.py             # Prediction script
├── data/
│   └── 300k.csv              # Dataset
├── models/                    # Saved model artifacts
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model

```bash
# Basic training with hyperparameter tuning
python scripts/train_model.py --tune_hyperparameters

# Custom configuration
python scripts/train_model.py \
    --data_path data/300k.csv \
    --output_dir models \
    --tune_hyperparameters \
    --cv_folds 5 \
    --log_level INFO
```

### 3. Make Predictions

```bash
# Predict on new data
python scripts/predict.py \
    --data_path data/new_data.csv \
    --model_dir models \
    --output_path predictions/results.csv
```

## Key Features

### Sample Weighting Strategy

The model uses `calculated_weights` derived from `brand_freq` to handle:
- **Class Imbalance**: ~95% verified vs ~5% not_verified
- **Brand Frequency Bias**: Prioritize rare brands (long tail) over common brands

```python
# Higher weights for rare brands (low brand_freq)
calculated_weights = 1 + (1 - brand_freq) * scaling_factor
```

### Feature Engineering

1. **Core Features**: `brand_freq`, `website_match`
2. **Text Features**: TF-IDF from `address_norm` (normalized customer input)
3. **Brand Features**: Encoded brand names with fallback to `snowdrop_name`
4. **Website Features**: Domain analysis, HTTPS, TLD patterns
5. **Geographic Features**: Country encoding (limited utility for online transactions)

### Model Configuration

- **Kernel**: RBF (Radial Basis Function)
- **Optimization Metric**: F1-score (better for imbalanced data than accuracy)
- **Hyperparameters**: C (regularization), gamma (kernel parameter), class_weight
- **Cross-Validation**: Stratified K-fold to maintain class balance

## Dataset Schema

| Column | Description | Usage |
|--------|-------------|--------|
| `brand_name` | Internal brand database entry | Feature encoding |
| `brand_freq` | Brand frequency (0-1) | **Critical**: Sample weighting |
| `snowdrop_name` | Fallback brand name | Feature encoding |
| `address_norm` | Normalized customer input | **Important**: Text features |
| `country`, `city` | Geographic data | Limited utility (online) |
| `snowdrop_website` | Official brand website | Website analysis |
| `website_match` | Fuzzy match score | **Important**: Direct feature |
| `label_clean` | Target (verified/not_verified) | **Target variable** |
| `calculated_weights` | Sample weights | **Critical**: Training weights |
| `split` | Predefined train/test split | Split strategy |

## Model Performance

The model is evaluated using imbalance-aware metrics:

- **F1-Score**: Primary optimization target
- **Precision/Recall**: For both classes
- **ROC-AUC**: Overall discriminative ability
- **PR-AUC**: Precision-Recall curve area (better for imbalanced data)
- **Brand Frequency Analysis**: Performance by brand popularity segments

## Advanced Usage

### Programmatic API

```python
from rbf_svm import ModelTrainer, DataLoader, FeatureEngineer

# Initialize components
trainer = ModelTrainer(random_state=42)

# Run complete pipeline
results = trainer.train_pipeline(
    data_path="data/300k.csv",
    output_dir="models",
    tune_hyperparameters=True
)

# Load trained model for inference
model, feature_engineer = trainer.load_trained_model("models")
```

### Custom Feature Engineering

```python
from rbf_svm import FeatureEngineer

fe = FeatureEngineer()
X_train, y_train, weights = fe.fit_transform(train_df)
X_test = fe.transform(test_df)
```

## Model Artifacts

After training, the following artifacts are saved:

- `rbf_svm_model.joblib`: Trained SVM model
- `feature_engineer.joblib`: Fitted feature engineering pipeline
- `training_metadata.json`: Model configuration and metrics
- `evaluation_report.json`: Detailed evaluation results
- `feature_importance.json`: Feature importance analysis

## Important Design Decisions

1. **Sample Weights**: Used during training to handle both class imbalance and brand frequency bias
2. **F1-Score Optimization**: Better than accuracy for imbalanced datasets
3. **Text Processing**: Focus on `address_norm` as normalized customer input (not geographic)
4. **Brand Encoding**: Top brands individually encoded, rare brands grouped as 'other'
5. **Cross-Validation**: Stratified to maintain class distribution across folds

## Monitoring and Evaluation

The model includes comprehensive evaluation:

- **Brand Frequency Segments**: Performance on rare vs common brands
- **Error Analysis**: Confidence analysis of false positives/negatives
- **Feature Importance**: Approximate importance for RBF-SVM
- **Confusion Matrix**: Detailed breakdown of predictions

## Future Extensions

The codebase is designed for easy extension:

- **Continuous Training**: Pipeline ready for incremental learning
- **Feature Store Integration**: Modular feature engineering
- **Model Monitoring**: Evaluation framework for production monitoring
- **A/B Testing**: Model comparison utilities

## Troubleshooting

### Common Issues

1. **Memory Issues**: Large TF-IDF matrices - reduce `max_features` in FeatureEngineer
2. **Slow Training**: Large dataset - reduce hyperparameter grid or use fewer CV folds
3. **Poor Performance**: Check data quality and feature engineering validity

### Logging

All components use Python logging. Set log level for detailed debugging:

```bash
python scripts/train_model.py --log_level DEBUG
```

## License

This project is designed for brand verification use cases with focus on long-tail sensitivity and production readiness.
