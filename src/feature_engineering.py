"""
Feature engineering module for RBF-SVM Brand Verification Model.

This module handles text processing, categorical encoding, and feature creation
with emphasis on capturing brand similarity and text patterns.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import hstack, csr_matrix

from .config import DataConfig, FeatureConfig
from .utils import clean_text

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for brand verification dataset.
    
    Handles text processing, categorical encoding, and feature creation
    with focus on brand name similarity and text patterns.
    """
    
    def __init__(self) -> None:
        """Initialize FeatureEngineer with configured parameters."""
        self.tfidf_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.is_fitted: bool = False
        
        # Store feature names for later reference
        self.feature_names: List[str] = []
        self.text_feature_names: Dict[str, List[str]] = {}
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, np.ndarray, List[str]]:
        """
        Fit feature engineering pipeline and transform data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, target_array, feature_names)
        """
        logger.info("Fitting and transforming features")
        
        # Process text features
        df_processed = self._process_text_features(df)
        
        # Extract features
        features = self._extract_all_features(df_processed, fit=True)
        
        # Extract target
        target = self._extract_target(df)
        
        self.is_fitted = True
        logger.info(f"Feature engineering completed. Feature shape: {features.shape}")
        
        return features, target, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, np.ndarray]:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, target_array)
            
        Raises:
            ValueError: If pipeline hasn't been fitted
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        logger.info("Transforming features using fitted pipeline")
        
        # Process text features
        df_processed = self._process_text_features(df)
        
        # Extract features
        features = self._extract_all_features(df_processed, fit=False)
        
        # Extract target
        target = self._extract_target(df)
        
        return features, target
    
    def _process_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process text features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed text features
        """
        df_processed = df.copy()
        
        # Process brand name - use fallback if needed
        brand_text = df_processed[DataConfig.BRAND_NAME_COL].fillna(
            df_processed[DataConfig.SNOWDROP_NAME_COL]
        ).fillna('')
        
        df_processed['brand_name_processed'] = brand_text.apply(clean_text)
        
        # Process address norm (merchant name)
        df_processed['address_norm_processed'] = df_processed[DataConfig.ADDRESS_NORM_COL].apply(clean_text)
        
        # Create combined text features
        df_processed['combined_text'] = (
            df_processed['brand_name_processed'] + ' ' + 
            df_processed['address_norm_processed']
        ).str.strip()
        
        return df_processed
    
    def _extract_text_features(self, df: pd.DataFrame, fit: bool = False) -> csr_matrix:
        """
        Extract TF-IDF features from text columns.
        
        Args:
            df: DataFrame with processed text
            fit: Whether to fit the vectorizers
            
        Returns:
            Sparse matrix of text features
        """
        text_features = []
        
        for feature_name in FeatureConfig.TEXT_FEATURES:
            if feature_name not in df.columns:
                logger.warning(f"Text feature {feature_name} not found in DataFrame")
                continue
            
            # Get text data
            text_data = df[feature_name].fillna('').astype(str)
            
            if fit:
                # Initialize and fit vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=FeatureConfig.MAX_FEATURES_TFIDF,
                    min_df=FeatureConfig.MIN_DF_TFIDF,
                    max_df=FeatureConfig.MAX_DF_TFIDF,
                    ngram_range=FeatureConfig.NGRAM_RANGE,
                    stop_words='english',
                    lowercase=True,
                    strip_accents='unicode'
                )
                
                features = vectorizer.fit_transform(text_data)
                self.tfidf_vectorizers[feature_name] = vectorizer
                
                # Store feature names
                feature_names = [f"{feature_name}_tfidf_{i}" for i in range(features.shape[1])]
                self.text_feature_names[feature_name] = feature_names
                
            else:
                # Transform using fitted vectorizer
                if feature_name not in self.tfidf_vectorizers:
                    logger.error(f"Vectorizer for {feature_name} not fitted")
                    continue
                
                vectorizer = self.tfidf_vectorizers[feature_name]
                features = vectorizer.transform(text_data)
            
            text_features.append(features)
            logger.info(f"Extracted {features.shape[1]} TF-IDF features from {feature_name}")
        
        if text_features:
            return hstack(text_features)
        else:
            # Return empty sparse matrix if no text features
            return csr_matrix((len(df), 0))
    
    def _extract_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> csr_matrix:
        """
        Extract one-hot encoded categorical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders
            
        Returns:
            Sparse matrix of categorical features
        """
        categorical_features = []
        
        for feature_name in FeatureConfig.CATEGORICAL_FEATURES:
            if feature_name not in df.columns:
                logger.warning(f"Categorical feature {feature_name} not found in DataFrame")
                continue
            
            # Handle missing values
            data = df[feature_name].fillna('unknown').astype(str)
            
            if fit:
                # Fit encoder
                encoder = OneHotEncoder(
                    sparse_output=True,
                    handle_unknown='ignore',
                    drop='first'  # Drop first category to avoid multicollinearity
                )
                
                features = encoder.fit_transform(data.values.reshape(-1, 1))
                self.onehot_encoders[feature_name] = encoder
                
                # Store feature names
                try:
                    categories = encoder.categories_[0]
                    if len(categories) > 1:  # Check if categories were dropped
                        feature_names = [f"{feature_name}_{cat}" for cat in categories[1:]]
                    else:
                        feature_names = []
                except:
                    feature_names = [f"{feature_name}_{i}" for i in range(features.shape[1])]
                
            else:
                # Transform using fitted encoder
                if feature_name not in self.onehot_encoders:
                    logger.error(f"Encoder for {feature_name} not fitted")
                    continue
                
                encoder = self.onehot_encoders[feature_name]
                features = encoder.transform(data.values.reshape(-1, 1))
                feature_names = []  # Names already stored
            
            if features.shape[1] > 0:
                categorical_features.append(features)
                if fit:
                    self.feature_names.extend(feature_names)
                
                logger.info(f"Extracted {features.shape[1]} categorical features from {feature_name}")
        
        if categorical_features:
            return hstack(categorical_features)
        else:
            return csr_matrix((len(df), 0))
    
    def _extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of numerical features
        """
        numerical_data = []
        numerical_names = []
        
        for feature_name in FeatureConfig.NUMERICAL_FEATURES:
            if feature_name not in df.columns:
                logger.warning(f"Numerical feature {feature_name} not found in DataFrame")
                continue
            
            # Extract and handle missing values
            values = pd.to_numeric(df[feature_name], errors='coerce').fillna(0.0)
            numerical_data.append(values.values)
            numerical_names.append(feature_name)
        
        if numerical_data:
            features = np.column_stack(numerical_data)
            if not hasattr(self, '_numerical_names_stored'):
                self.feature_names.extend(numerical_names)
                self._numerical_names_stored = True
            
            logger.info(f"Extracted {len(numerical_names)} numerical features")
            return features
        else:
            return np.empty((len(df), 0))
    
    def _create_similarity_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create brand similarity features.
        
        Args:
            df: DataFrame with processed text
            
        Returns:
            Array of similarity features
        """
        similarity_features = []
        similarity_names = []
        
        # Website match score (already exists)
        if DataConfig.WEBSITE_MATCH_COL in df.columns:
            website_match = pd.to_numeric(df[DataConfig.WEBSITE_MATCH_COL], errors='coerce').fillna(0.0)
            similarity_features.append(website_match.values)
            similarity_names.append('website_match_score')
        
        # Brand frequency (inverse of rarity)
        if DataConfig.BRAND_FREQ_COL in df.columns:
            brand_freq = pd.to_numeric(df[DataConfig.BRAND_FREQ_COL], errors='coerce').fillna(0.0)
            similarity_features.append(brand_freq.values)
            similarity_names.append('brand_frequency')
            
            # Log brand frequency for emphasis on long-tail
            log_brand_freq = np.log1p(brand_freq.values)
            similarity_features.append(log_brand_freq)
            similarity_names.append('log_brand_frequency')
        
        # Text length features
        if 'brand_name_processed' in df.columns:
            brand_length = df['brand_name_processed'].str.len().fillna(0)
            similarity_features.append(brand_length.values)
            similarity_names.append('brand_name_length')
        
        if 'address_norm_processed' in df.columns:
            address_length = df['address_norm_processed'].str.len().fillna(0)
            similarity_features.append(address_length.values)
            similarity_names.append('address_norm_length')
        
        if similarity_features:
            features = np.column_stack(similarity_features)
            if not hasattr(self, '_similarity_names_stored'):
                self.feature_names.extend(similarity_names)
                self._similarity_names_stored = True
            
            logger.info(f"Created {len(similarity_names)} similarity features")
            return features
        else:
            return np.empty((len(df), 0))
    
    def _extract_all_features(self, df: pd.DataFrame, fit: bool = False) -> csr_matrix:
        """
        Extract all feature types and combine them.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the feature extractors
            
        Returns:
            Combined sparse feature matrix
        """
        all_features = []
        
        # Reset feature names if fitting
        if fit:
            self.feature_names = []
            self._numerical_names_stored = False
            self._similarity_names_stored = False
        
        # Extract text features
        text_features = self._extract_text_features(df, fit=fit)
        if text_features.shape[1] > 0:
            all_features.append(text_features)
            if fit:
                # Add text feature names
                for feature_name in FeatureConfig.TEXT_FEATURES:
                    if feature_name in self.text_feature_names:
                        self.feature_names.extend(self.text_feature_names[feature_name])
        
        # Extract categorical features
        categorical_features = self._extract_categorical_features(df, fit=fit)
        if categorical_features.shape[1] > 0:
            all_features.append(categorical_features)
        
        # Extract numerical features
        numerical_features = self._extract_numerical_features(df)
        if numerical_features.shape[1] > 0:
            all_features.append(csr_matrix(numerical_features))
        
        # Extract similarity features
        similarity_features = self._create_similarity_features(df)
        if similarity_features.shape[1] > 0:
            all_features.append(csr_matrix(similarity_features))
        
        # Combine all features
        if all_features:
            combined_features = hstack(all_features)
        else:
            combined_features = csr_matrix((len(df), 0))
        
        logger.info(f"Combined feature matrix shape: {combined_features.shape}")
        return combined_features
    
    def _extract_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and encode target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Binary encoded target array
        """
        target_series = df[DataConfig.TARGET_COL]
        
        # Convert to binary (1 for verified, 0 for not_verified)
        target_binary = (target_series == DataConfig.VERIFIED_LABEL).astype(int)
        
        logger.info(f"Target distribution: {np.bincount(target_binary)}")
        return target_binary.values
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get feature names for importance analysis.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
