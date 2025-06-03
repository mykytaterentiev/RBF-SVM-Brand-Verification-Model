"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from rbf_svm.preprocessing.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'brand_name': ['spotify', 'amazon', '', 'local_shop', 'spotify'],
        'brand_freq': [1.0, 0.8, 0.2, 0.001, 1.0],
        'snowdrop_name': ['spotify', 'amazon', 'unknown_shop', 'local_shop', 'spotify'],
        'address_norm': ['spotify premium', 'amazon music', 'small shop', 'local business', 'spotify app'],
        'country': ['se', 'lu', 'pl', 'pl', 'se'],
        'city': ['stockholm', 'luxembourg', 'warsaw', 'krakow', 'stockholm'],
        'snowdrop_website': ['https://spotify.com', 'https://amazon.com', '', 'http://local.pl', 'https://spotify.com'],
        'website_match': [0.9, 0.8, 0.0, 0.3, 0.9],
        'label_clean': ['verified', 'verified', 'not_verified', 'verified', 'verified'],
        'calculated_weights': [1.0, 1.2, 10.8, 11.0, 1.0]
    })


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_fit_transform_basic(self, sample_dataframe):
        """Test basic fit_transform functionality."""
        fe = FeatureEngineer()
        features, labels, weights = fe.fit_transform(sample_dataframe)
        
        assert features.shape[0] == len(sample_dataframe)
        assert labels.shape[0] == len(sample_dataframe)
        assert weights.shape[0] == len(sample_dataframe)
        assert fe.is_fitted
        
    def test_transform_after_fit(self, sample_dataframe):
        """Test transform after fitting."""
        fe = FeatureEngineer()
        fe.fit_transform(sample_dataframe)
        
        # Transform new data
        new_features = fe.transform(sample_dataframe)
        assert new_features.shape[1] > 0  # Should have features
        
    def test_transform_without_fit_raises_error(self, sample_dataframe):
        """Test that transform without fit raises error."""
        fe = FeatureEngineer()
        
        with pytest.raises(ValueError, match="must be fitted"):
            fe.transform(sample_dataframe)
            
    def test_label_encoding(self, sample_dataframe):
        """Test label encoding functionality."""
        fe = FeatureEngineer()
        _, labels, _ = fe.fit_transform(sample_dataframe)
        
        # Should be binary encoded
        assert set(labels) <= {0, 1}
        assert labels.sum() == 4  # 4 verified labels
        
    def test_feature_names_generation(self, sample_dataframe):
        """Test feature names generation."""
        fe = FeatureEngineer()
        fe.fit_transform(sample_dataframe)
        
        feature_names = fe.get_feature_names()
        assert len(feature_names) > 0
        assert 'brand_freq' in feature_names
        assert 'website_match' in feature_names
