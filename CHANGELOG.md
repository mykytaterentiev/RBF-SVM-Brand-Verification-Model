# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- RBF-SVM implementation with hyperparameter tuning
- Feature engineering pipeline with long-tail sensitivity
- Comprehensive evaluation metrics for imbalanced data
- Sample weighting for brand frequency bias handling

## [0.1.0] - 2024-01-XX

### Added
- Core RBF-SVM classifier with scikit-learn integration
- Feature engineering for brand verification:
  - Text features from address_norm using TF-IDF
  - Brand name encoding with frequency-based grouping
  - Website analysis features
  - Geographic features
  - Brand frequency derived features
- Data loading and validation with quality checks
- Model training orchestration with complete pipeline
- Comprehensive model evaluation:
  - Imbalance-aware metrics
  - Brand frequency segment analysis
  - Error analysis with confidence assessment
  - Feature importance approximation
- Training and prediction scripts
- Modular architecture for easy extension
- Comprehensive documentation and examples

### Features
- **Long-tail Sensitivity**: Prioritizes rare brands through sample weighting
- **Imbalanced Data Handling**: F1-score optimization and weighted metrics
- **Hyperparameter Tuning**: Grid search with stratified cross-validation
- **Feature Engineering**: 183 engineered features from raw data
- **Production Ready**: Logging, error handling, and artifact management

### Technical Details
- Python 3.8+ support
- Scikit-learn 1.3+ integration
- Comprehensive test coverage
- Type hints throughout
- Pre-commit hooks for code quality
- Security-focused development practices

### Performance
- Handles 250K+ training samples
- 183 engineered features
- Grid search over 48 parameter combinations
- Stratified 5-fold cross-validation
- Memory-efficient feature engineering

### Security
- Data files excluded from version control
- Environment variable configuration
- Input validation and sanitization
- Secure model artifact handling
- Pre-commit security checks

## [0.0.1] - 2024-01-XX

### Added
- Initial project setup
- Basic project structure
- Core dependencies
- Development tooling configuration
