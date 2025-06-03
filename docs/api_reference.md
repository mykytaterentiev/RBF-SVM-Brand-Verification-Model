# API Reference

## Core Classes

### ModelTrainer

Main orchestration class for training and prediction.

```python
from rbf_svm.training.trainer import ModelTrainer

trainer = ModelTrainer(random_state=42)
```

#### Methods

**`train_pipeline(data_path, output_dir, tune_hyperparameters=True, cv_folds=5, test_size=0.2)`**

Complete training pipeline from data loading to model saving.

- `data_path` (str): Path to CSV training data
- `output_dir` (str): Directory to save model artifacts
- `tune_hyperparameters` (bool): Enable grid search optimization
- `cv_folds` (int): Cross-validation folds for hyperparameter tuning
- `test_size` (float): Test set proportion if no predefined split

Returns: Dictionary with training results and metrics.

**`predict_on_new_data(data_path, model_dir, output_path=None)`**

Make predictions on new data using trained model.

- `data_path` (str): Path to CSV data for prediction
- `model_dir` (str): Directory containing trained model artifacts
- `output_path` (str, optional): Path to save predictions

Returns: DataFrame with predictions and confidence scores.

### FeatureEngineer

Feature engineering pipeline for brand verification data.

```python
from rbf_svm.preprocessing.feature_engineer import FeatureEngineer

fe = FeatureEngineer()
features, labels, weights = fe.fit_transform(df)
```

#### Methods

**`fit_transform(df)`**

Fit feature engineering pipeline and transform data.

- `df` (DataFrame): Input data with required columns

Returns: Tuple of (features, labels, sample_weights)

**`transform(df)`**

Transform new data using fitted pipeline.

- `df` (DataFrame): Input data

Returns: Transformed feature matrix

**`get_feature_names()`**

Get names of engineered features.

Returns: List of feature names

### RBFSVMClassifier

RBF-kernel SVM with hyperparameter optimization.

```python
from rbf_svm.models.rbf_svm import RBFSVMClassifier

model = RBFSVMClassifier(random_state=42)
```

#### Methods

**`fit(X, y, sample_weight=None, tune_hyperparameters=True, cv_folds=5)`**

Train the model with optional hyperparameter tuning.

**`predict(X)`**

Make binary predictions.

**`predict_proba(X)`**

Get prediction probabilities.

### ModelEvaluator

Comprehensive model evaluation for imbalanced data.

```python
from rbf_svm.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X, y, sample_weights)
```

## Required Data Schema

Your CSV data must contain these columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `brand_name` | str | Internal brand identifier | Yes |
| `brand_freq` | float | Brand frequency (0-1) | Yes |
| `snowdrop_name` | str | Fallback brand name | Yes |
| `address_norm` | str | Normalized customer input | Yes |
| `country` | str | Country code | No |
| `city` | str | City name | No |
| `snowdrop_website` | str | Official brand website | No |
| `website_match` | float | Fuzzy match score (0-1) | Yes |
| `label_clean` | str | Target: 'verified' or 'not_verified' | Yes |
| `calculated_weights` | float | Sample weights for training | Yes |
| `split` | str | Optional: 'TRAIN', 'TEST', 'VALIDATE' | No |

## Environment Variables

Configure the system using environment variables:

```bash
DATA_PATH=data/your_data.csv
MODEL_OUTPUT_DIR=models
TUNE_HYPERPARAMETERS=true
RANDOM_STATE=42
LOG_LEVEL=INFO
```

## Error Handling

The library raises specific exceptions:

- `ValueError`: Invalid input data or configuration
- `FileNotFoundError`: Missing data or model files
- `RuntimeError`: Training or prediction failures

Always wrap model operations in try-catch blocks for production use.
