# Contributing to RBF-SVM Brand Verification

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://huggingface.co/your-username/rbf-svm-vertex
cd rbf-svm-vertex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

## Development Workflow

### 1. Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Check linting
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

### 2. Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rbf_svm --cov-report=html

# Run specific test
pytest tests/test_feature_engineer.py
```

### 3. Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Contribution Guidelines

### 1. Issues

Before creating a new issue:
- Search existing issues
- Use issue templates when available
- Provide clear reproduction steps
- Include relevant environment information

### 2. Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Write tests for new functionality
   - Update documentation as needed
   - Follow existing code patterns
   - Ensure all checks pass

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### 3. Commit Message Format

We follow conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/modifications
- `chore:` Maintenance tasks

Example: `feat: add brand frequency analysis to evaluator`

### 4. Code Review Process

1. **Automated checks** must pass
2. **Manual review** by maintainers
3. **Security review** for sensitive changes
4. **Documentation review** if applicable

## Security Considerations

### Data Privacy
- **Never commit data files** to the repository
- **Use synthetic data** for tests when possible
- **Sanitize examples** in documentation

### Code Security
- **Validate all inputs**
- **Avoid hardcoded secrets**
- **Use environment variables** for configuration
- **Keep dependencies updated**

## Testing Guidelines

### 1. Test Structure

```python
# tests/test_feature_engineer.py
import pytest
import pandas as pd
import numpy as np
from rbf_svm.preprocessing.feature_engineer import FeatureEngineer

class TestFeatureEngineer:
    def test_fit_transform_basic(self):
        # Arrange
        df = create_sample_dataframe()
        fe = FeatureEngineer()
        
        # Act
        features, labels, weights = fe.fit_transform(df)
        
        # Assert
        assert features.shape[0] == len(df)
        assert labels.shape[0] == len(df)
        assert weights.shape[0] == len(df)
```

### 2. Test Data

- Use synthetic data for unit tests
- Create fixtures for common test scenarios
- Mock external dependencies
- Test edge cases and error conditions

### 3. Integration Tests

- Test complete pipelines
- Use sample datasets (non-sensitive)
- Verify model training and prediction workflows

## Documentation

### 1. Code Documentation

- **Docstrings**: All public functions and classes
- **Type hints**: For all function parameters and returns
- **Comments**: For complex logic

### 2. User Documentation

- **README.md**: Project overview and quick start
- **API documentation**: Generated from docstrings
- **Examples**: Jupyter notebooks for common use cases

### 3. Developer Documentation

- **Architecture decisions**: Document design choices
- **Performance considerations**: Optimization notes
- **Security measures**: Document security implementations

## Release Process

### 1. Version Bumping

```bash
# Update version in setup.py and pyproject.toml
# Update CHANGELOG.md
# Create release commit
git commit -m "chore: bump version to 0.2.0"
git tag v0.2.0
```

### 2. Release Notes

- Summarize new features
- List bug fixes
- Note breaking changes
- Include migration instructions if needed

## Getting Help

- **General questions**: Open a discussion
- **Bug reports**: Create an issue
- **Security issues**: Follow SECURITY.md guidelines
- **Feature requests**: Open an issue with enhancement label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
