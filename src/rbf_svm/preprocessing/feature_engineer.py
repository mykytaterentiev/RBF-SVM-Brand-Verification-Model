"""Feature engineering for brand verification with focus on long-tail sensitivity."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for brand verification model."""
    
    def __init__(self):
        """Initialize feature engineering components."""
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.brand_encoder = LabelEncoder()
        self.country_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit feature engineering pipeline and transform data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, labels, sample_weights)
        """
        logger.info("Fitting and transforming features")
        
        # Create feature matrix
        features = self._extract_features(df, fit=True)
        
        # Extract labels
        labels = self._encode_labels(df['label_clean'])
        
        # Extract sample weights
        sample_weights = df['calculated_weights'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        self.is_fitted = True
        logger.info(f"Feature engineering complete. Shape: {features_scaled.shape}")
        
        return features_scaled, labels, sample_weights
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
            
        logger.info("Transforming features")
        
        # Extract features
        features = self._extract_features(df, fit=False)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def _extract_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Extract and engineer features from DataFrame.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers
            
        Returns:
            Feature matrix
        """
        feature_list = []
        
        # 1. Core numeric features
        core_features = [
            'brand_freq',  # Critical: frequency/popularity of brand
            'website_match',  # Fuzzy match score
        ]
        
        for col in core_features:
            if col in df.columns:
                # Handle missing values with median
                values = df[col].fillna(df[col].median()).values
                feature_list.append(values.reshape(-1, 1))
        
        # 2. Brand frequency derived features
        if 'brand_freq' in df.columns:
            brand_freq = df['brand_freq'].fillna(df['brand_freq'].median())
            
            # Long-tail indicator (inverse of brand_freq)
            long_tail_score = 1 - brand_freq
            feature_list.append(long_tail_score.values.reshape(-1, 1))
            
            # Brand popularity category
            brand_categories = pd.cut(brand_freq, 
                                    bins=[0, 0.01, 0.1, 0.5, 1.0], 
                                    labels=['rare', 'uncommon', 'common', 'very_common'])
            if fit:
                brand_cat_encoded = pd.get_dummies(brand_categories, prefix='brand_cat')
            else:
                # Use same categories as during fit
                brand_cat_encoded = pd.get_dummies(brand_categories, prefix='brand_cat')
                # Ensure same columns as during fit
                for col in self.brand_cat_columns:
                    if col not in brand_cat_encoded.columns:
                        brand_cat_encoded[col] = 0
                brand_cat_encoded = brand_cat_encoded[self.brand_cat_columns]
            
            if fit:
                self.brand_cat_columns = brand_cat_encoded.columns.tolist()
            
            feature_list.append(brand_cat_encoded.values)
        
        # 3. Text features from address_norm (normalized name input)
        if 'address_norm' in df.columns:
            text_features = self._extract_text_features(df['address_norm'], fit=fit)
            if text_features is not None:
                feature_list.append(text_features)
        
        # 4. Brand name features (if available)
        if 'brand_name' in df.columns:
            brand_features = self._extract_brand_features(df, fit=fit)
            feature_list.append(brand_features)
        
        # 5. Geographic features (limited utility but included)
        geo_features = self._extract_geo_features(df, fit=fit)
        if geo_features is not None:
            feature_list.append(geo_features)
        
        # 6. Website-related features
        if 'snowdrop_website' in df.columns:
            website_features = self._extract_website_features(df['snowdrop_website'])
            feature_list.append(website_features)
        
        # Concatenate all features
        if feature_list:
            features = np.hstack(feature_list)
        else:
            raise ValueError("No features could be extracted")
            
        return features
    
    def _extract_text_features(self, text_series: pd.Series, fit: bool = False) -> Optional[np.ndarray]:
        """Extract TF-IDF features from address_norm text."""
        # Clean and preprocess text
        cleaned_text = text_series.fillna('').apply(self._clean_text)
        
        if fit:
            # Fit TF-IDF with focus on meaningful terms
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,  # Keep manageable number of features
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=5,  # Ignore very rare terms
                max_df=0.95,  # Ignore very common terms
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_text)
        else:
            if self.tfidf_vectorizer is None:
                return None
            tfidf_features = self.tfidf_vectorizer.transform(cleaned_text)
        
        return tfidf_features.toarray()
    
    def _clean_text(self, text: str) -> str:
        """Clean text for feature extraction."""
        if pd.isna(text):
            return ""
        
        # Remove special characters and normalize
        text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        return text
    
    def _extract_brand_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract features from brand_name."""
        # Brand presence indicator
        has_brand = (~df['brand_name'].isna()).astype(int)
        
        # Use snowdrop_name as fallback
        brand_text = df['brand_name'].fillna(df['snowdrop_name']).fillna('unknown')
        
        if fit:
            # Encode top brands, group rare ones
            brand_counts = brand_text.value_counts()
            top_brands = brand_counts.head(50).index.tolist()  # Top 50 brands
            
            self.top_brands = top_brands
            brand_categorical = brand_text.apply(
                lambda x: x if x in top_brands else 'other'
            )
            
            brand_encoded = pd.get_dummies(brand_categorical, prefix='brand')
            self.brand_columns = brand_encoded.columns.tolist()
        else:
            brand_categorical = brand_text.apply(
                lambda x: x if x in self.top_brands else 'other'
            )
            brand_encoded = pd.get_dummies(brand_categorical, prefix='brand')
            
            # Ensure same columns as during fit
            for col in self.brand_columns:
                if col not in brand_encoded.columns:
                    brand_encoded[col] = 0
            brand_encoded = brand_encoded[self.brand_columns]
        
        # Combine features
        features = np.hstack([
            has_brand.values.reshape(-1, 1),
            brand_encoded.values
        ])
        
        return features
    
    def _extract_geo_features(self, df: pd.DataFrame, fit: bool = False) -> Optional[np.ndarray]:
        """Extract geographic features (limited utility for online transactions)."""
        features = []
        
        # Country encoding (top countries only)
        if 'country' in df.columns:
            countries = df['country'].fillna('unknown')
            
            if fit:
                country_counts = countries.value_counts()
                top_countries = country_counts.head(20).index.tolist()
                self.top_countries = top_countries
                
                country_categorical = countries.apply(
                    lambda x: x if x in top_countries else 'other'
                )
                country_encoded = pd.get_dummies(country_categorical, prefix='country')
                self.country_columns = country_encoded.columns.tolist()
            else:
                country_categorical = countries.apply(
                    lambda x: x if x in self.top_countries else 'other'
                )
                country_encoded = pd.get_dummies(country_categorical, prefix='country')
                
                # Ensure same columns
                for col in self.country_columns:
                    if col not in country_encoded.columns:
                        country_encoded[col] = 0
                country_encoded = country_encoded[self.country_columns]
            
            features.append(country_encoded.values)
        
        if features:
            return np.hstack(features)
        return None
    
    def _extract_website_features(self, website_series: pd.Series) -> np.ndarray:
        """Extract features from website URLs."""
        # Has website indicator
        has_website = (~website_series.isna()).astype(int)
        
        # Extract domain features
        domains = website_series.fillna('').apply(self._extract_domain)
        
        # Common TLD indicator
        common_tlds = ['.com', '.org', '.net', '.gov', '.edu']
        has_common_tld = domains.apply(
            lambda x: int(any(tld in x for tld in common_tlds))
        )
        
        # HTTPS indicator
        has_https = website_series.fillna('').apply(
            lambda x: int(x.startswith('https://'))
        )
        
        features = np.column_stack([
            has_website.values,
            has_common_tld.values,
            has_https.values
        ])
        
        return features
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if pd.isna(url) or not url:
            return ""
        
        # Simple domain extraction
        url = str(url).lower()
        if '://' in url:
            url = url.split('://', 1)[1]
        if '/' in url:
            url = url.split('/')[0]
        
        return url
    
    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Encode string labels to numeric."""
        # Map labels to binary
        label_map = {'verified': 1, 'not_verified': 0}
        
        encoded_labels = labels.map(label_map)
        
        if encoded_labels.isna().any():
            logger.warning("Unknown labels found, treating as not_verified")
            encoded_labels = encoded_labels.fillna(0)
        
        return encoded_labels.values
    
    def get_feature_names(self) -> list:
        """Get names of engineered features."""
        if not self.is_fitted:
            return []
        
        names = ['brand_freq', 'website_match', 'long_tail_score']
        
        # Add brand category names
        if hasattr(self, 'brand_cat_columns'):
            names.extend(self.brand_cat_columns)
        
        # Add TF-IDF feature names
        if self.tfidf_vectorizer is not None:
            tfidf_names = [f"tfidf_{i}" for i in range(len(self.tfidf_vectorizer.get_feature_names_out()))]
            names.extend(tfidf_names)
        
        # Add other feature names
        names.extend(['has_brand'])
        if hasattr(self, 'brand_columns'):
            names.extend(self.brand_columns)
        
        if hasattr(self, 'country_columns'):
            names.extend(self.country_columns)
        
        names.extend(['has_website', 'has_common_tld', 'has_https'])
        
        return names
