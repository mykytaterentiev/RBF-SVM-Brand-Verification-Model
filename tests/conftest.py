"""Test configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create sample CSV file for testing."""
    data = {
        'brand_name': ['spotify', 'amazon', '', 'test_brand'] * 25,
        'brand_freq': [1.0, 0.8, 0.2, 0.001] * 25,
        'snowdrop_name': ['spotify', 'amazon', 'unknown', 'test_brand'] * 25,
        'address_norm': ['spotify app', 'amazon store', 'local shop', 'test store'] * 25,
        'country': ['se', 'de', 'pl', 'us'] * 25,
        'city': ['stockholm', 'berlin', 'warsaw', 'new york'] * 25,
        'snowdrop_website': ['https://spotify.com', 'https://amazon.com', '', 'http://test.com'] * 25,
        'website_match': [0.9, 0.8, 0.0, 0.3] * 25,
        'label_clean': ['verified', 'verified', 'not_verified', 'verified'] * 25,
        'calculated_weights': [1.0, 1.2, 10.8, 11.0] * 25,
        'split': ['TRAIN', 'TRAIN', 'TEST', 'VALIDATE'] * 25
    }
    
    df = pd.DataFrame(data)
    csv_path = temp_dir / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
