"""Generate dummy data for testing RBF-SVM brand verification model."""

import argparse
import pandas as pd
import numpy as np
import random
import sys
from pathlib import Path
import logging
from typing import List

logger = logging.getLogger(__name__)


class DummyDataGenerator:
    """Generate realistic dummy data for brand verification."""
    
    def __init__(self, random_state: int = 42):
        """Initialize with random state for reproducibility."""
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Realistic brand names (mix of common and rare)
        self.brands = {
            # Very common brands (high frequency)
            'spotify': 1.0,
            'amazon': 0.8,
            'google': 0.7,
            'facebook': 0.6,
            'netflix': 0.5,
            'paypal': 0.4,
            'ryanair': 0.3,
            
            # Common brands (medium frequency)
            'lidl': 0.15,
            'shell': 0.12,
            'biedronka': 0.1,
            'orlen': 0.08,
            'żabka': 0.07,
            'carrefour': 0.06,
            'inpost': 0.05,
            'dino': 0.04,
            'netto': 0.03,
            'circle k': 0.02,
            
            # Uncommon brands (low frequency)
            'kiwi': 0.015,
            'stokrotka': 0.012,
            'lewiatan': 0.01,
            'deichmann': 0.008,
            'mediamarkt': 0.006,
            'pepco': 0.005,
            'edeka': 0.004,
            'rewe': 0.003,
            'tedi': 0.002,
            'uniqlo': 0.001,
            
            # Rare/specialized brands (very low frequency)
            'groszek': 0.0007,
            'świat prasy': 0.0005,
            'apteka hygieia': 0.0003,
            'piekarnia pod telegrafem': 0.0002,
            'tabak polska': 0.0001,
        }
        
        # Countries with realistic distribution
        self.countries = ['pl', 'de', 'dk', 'se', 'ie', 'no', 'nl', 'be', 'fr', 'es', 'it', 'gr', 'grc', 'cz', 'lu', 'gb']
        
        # Cities by country
        self.cities = {
            'pl': ['warsaw', 'krakow', 'gdansk', 'wroclaw', 'poznan', 'lodz', 'szczecin', 'lublin'],
            'de': ['berlin', 'munich', 'hamburg', 'cologne', 'frankfurt', 'stuttgart', 'düsseldorf'],
            'dk': ['copenhagen', 'copenhagen s', 'aarhus', 'odense', 'aalborg'],
            'se': ['stockholm', 'gothenburg', 'malmö', 'uppsala', 'linköping'],
            'ie': ['dublin', 'cork', 'galway', 'limerick', 'waterford'],
            'no': ['oslo', 'bergen', 'stavanger', 'trondheim', 'drammen'],
            'nl': ['amsterdam', 'rotterdam', 'the hague', 'utrecht', 'eindhoven'],
            'be': ['brussels', 'antwerp', 'ghent', 'charleroi', 'liège'],
            'fr': ['paris', 'lyon', 'marseille', 'toulouse', 'nice'],
            'es': ['madrid', 'barcelona', 'valencia', 'seville', 'bilbao'],
            'it': ['rome', 'milan', 'naples', 'turin', 'palermo'],
            'gr': ['athens', 'thessaloniki', 'patras', 'heraklion'],
            'grc': ['athens', 'thessaloniki', 'patras', 'heraklion'],
            'cz': ['prague', 'brno', 'ostrava', 'plzen'],
            'lu': ['luxembourg', 'esch-sur-alzette'],
            'gb': ['london', 'manchester', 'birmingham', 'glasgow', 'liverpool']
        }
    
    def generate_dataset(self, n_samples: int = 300000) -> pd.DataFrame:
        """Generate complete dummy dataset."""
        logger.info(f"Generating {n_samples} dummy samples")
        
        data = []
        
        for i in range(n_samples):
            # Select brand with frequency-based probability
            brand_name, brand_freq = self._select_brand()
            
            # Generate other fields
            snowdrop_name = self._generate_snowdrop_name(brand_name)
            address_norm = self._generate_address_norm(brand_name, snowdrop_name)
            country = np.random.choice(self.countries, p=self._get_country_probabilities())
            city = self._get_city_for_country(country)
            snowdrop_website = self._generate_website(brand_name, brand_freq)
            website_match = self._generate_website_match(brand_name, snowdrop_name, snowdrop_website)
            label_clean = self._generate_label(brand_freq, website_match)
            calculated_weights = self._generate_calculated_weights(brand_freq, label_clean)
            split = self._generate_split(i, n_samples)
            
            data.append({
                'brand_name': brand_name,
                'brand_freq': brand_freq,
                'snowdrop_name': snowdrop_name,
                'address_norm': address_norm,
                'country': country,
                'city': city,
                'snowdrop_website': snowdrop_website,
                'website_match': website_match,
                'label_clean': label_clean,
                'calculated_weights': calculated_weights,
                'split': split
            })
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1}/{n_samples} samples")
        
        df = pd.DataFrame(data)
        logger.info(f"Dataset generation complete. Shape: {df.shape}")
        
        return df
    
    def _select_brand(self) -> tuple:
        """Select brand name based on frequency distribution."""
        if np.random.random() < 0.1:  # 10% chance of empty brand (not_verified cases)
            return "", np.random.uniform(0.2, 0.22)  # Default weight for empty brands
        
        # Weight brands by frequency for realistic distribution
        brands = list(self.brands.keys())
        weights = list(self.brands.values())
        
        # Adjust weights to create realistic long-tail distribution
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        brand = np.random.choice(brands, p=weights)
        freq = self.brands[brand] + np.random.normal(0, 0.01)  # Add small noise
        freq = max(0, min(1, freq))  # Clamp to [0, 1]
        
        return brand, freq
    
    def _generate_snowdrop_name(self, brand_name: str) -> str:
        """Generate snowdrop_name (fallback brand name)."""
        if not brand_name:
            # Generate random business names for unverified entries
            prefixes = ['piekarnia', 'stacja paliw', 'sklep', 'restaurant', 'cafe', 'market']
            suffixes = ['centrum', 'express', 'plus', 'mini', 'super', 'local']
            return f"{np.random.choice(prefixes)} {np.random.choice(suffixes)}"
        
        return brand_name
    
    def _generate_address_norm(self, brand_name: str, snowdrop_name: str) -> str:
        """Generate realistic address_norm (customer input)."""
        base_name = brand_name if brand_name else snowdrop_name
        
        # Add realistic variations/corruptions
        variations = [
            base_name,  # Exact match
            base_name.upper(),  # All caps
            base_name.lower(),  # All lowercase
            f"{base_name} sp",  # With company suffix
            f"{base_name} polska",  # With country
            f"{base_name} {self._random_string(2, 4)}",  # With random suffix
            f"{base_name.replace(' ', '')}",  # No spaces
        ]
        
        # For 'too good to go', add special patterns
        if 'too good to go' in base_name.lower():
            variations.extend([
                f"toogoodt {self._random_string(2, 6)}",
                f"toogoodt {self._random_string(2, 4)} {self._random_string(2, 4)}",
            ])
        
        return np.random.choice(variations)
    
    def _random_string(self, min_len: int, max_len: int) -> str:
        """Generate random string of letters."""
        length = np.random.randint(min_len, max_len + 1)
        return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
    
    def _get_country_probabilities(self) -> List[float]:
        """Get realistic country distribution."""
        # Poland dominates, followed by other EU countries
        base_probs = [0.6, 0.15, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.015, 0.015, 0.01, 0.01, 0.005, 0.005, 0.005, 0.003]
        
        # Ensure we have probabilities for all countries
        num_countries = len(self.countries)
        if len(base_probs) > num_countries:
            probs = base_probs[:num_countries]
        else:
            probs = base_probs + [0.001] * (num_countries - len(base_probs))
        
        # Normalize to ensure they sum to 1
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return probs.tolist()
    
    def _get_city_for_country(self, country: str) -> str:
        """Get random city for country."""
        if country in self.cities:
            return np.random.choice(self.cities[country])
        return "unknown"
    
    def _generate_website(self, brand_name: str, brand_freq: float) -> str:
        """Generate realistic website URL."""
        if not brand_name or np.random.random() < 0.1:  # 10% chance of no website
            return ""
        
        # Higher frequency brands more likely to have websites
        if np.random.random() > brand_freq + 0.1:
            return ""
        
        # Generate realistic domains
        domain_name = brand_name.lower().replace(' ', '').replace('ł', 'l').replace('ż', 'z')
        
        tlds = ['.com', '.pl', '.de', '.dk', '.se', '.ie', '.no', '.nl', '.be', '.fr', '.es', '.it', '.gr']
        protocol = 'https://' if np.random.random() < 0.8 else 'http://'
        subdomain = 'www.' if np.random.random() < 0.7 else ''
        
        return f"{protocol}{subdomain}{domain_name}{np.random.choice(tlds)}"
    
    def _generate_website_match(self, brand_name: str, snowdrop_name: str, website: str) -> float:
        """Generate realistic website match score."""
        if not website or not brand_name:
            return 0.0
        
        # Higher match scores for exact brand matches
        if brand_name.lower() in website.lower():
            return np.random.uniform(0.7, 1.0)
        elif any(word in website.lower() for word in brand_name.lower().split()):
            return np.random.uniform(0.3, 0.7)
        else:
            return np.random.uniform(0.0, 0.3)
    
    def _generate_label(self, brand_freq: float, website_match: float) -> str:
        """Generate label based on brand frequency and website match."""
        # Higher frequency brands and better website matches more likely to be verified
        # But maintain overall 95% verified rate
        base_prob = 0.95
        
        # Slight adjustments based on brand frequency and website match
        freq_adjustment = (brand_freq - 0.5) * 0.02  # Small adjustment based on frequency
        match_adjustment = (website_match - 0.5) * 0.01  # Small adjustment based on match
        
        verification_prob = base_prob + freq_adjustment + match_adjustment
        verification_prob = max(0.85, min(0.98, verification_prob))  # Clamp between 85% and 98%
        
        if np.random.random() < verification_prob:
            return 'verified'
        else:
            return 'not_verified'
    
    def _generate_calculated_weights(self, brand_freq: float, label: str) -> float:
        """Generate calculated weights based on brand frequency."""
        # Inverse relationship: lower frequency = higher weight
        base_weight = 1 + (1 - brand_freq) * 10
        
        # Add some noise
        noise = np.random.normal(0, 0.1)
        weight = base_weight + noise
        
        return max(1.0, weight)  # Minimum weight of 1.0
    
    def _generate_split(self, index: int, total_samples: int) -> str:
        """Generate train/test/validate split."""
        # 70% train, 15% validate, 15% test
        if index < total_samples * 0.7:
            return 'TRAIN'
        elif index < total_samples * 0.85:
            return 'VALIDATE'
        else:
            return 'TEST'


def main():
    """Main function to generate dummy data."""
    parser = argparse.ArgumentParser(description="Generate dummy data for RBF-SVM brand verification")
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=300000,
        help="Number of samples to generate (default: 300000)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/dummy_300k.csv",
        help="Output path for generated data (default: data/dummy_300k.csv)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        generator = DummyDataGenerator(random_state=args.random_state)
        df = generator.generate_dataset(n_samples=args.n_samples)
        
        # Save to CSV
        logger.info(f"Saving dataset to {output_path}")
        df.to_csv(output_path, index=False)
        
        # Print statistics
        print("\n" + "="*60)
        print("DUMMY DATA GENERATION COMPLETED")
        print("="*60)
        print(f"Generated samples: {len(df):,}")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        print("\nDataset Statistics:")
        print(f"  Label distribution: {df['label_clean'].value_counts().to_dict()}")
        print(f"  Brand frequency stats:")
        print(f"    Mean: {df['brand_freq'].mean():.4f}")
        print(f"    Median: {df['brand_freq'].median():.4f}")
        print(f"    Min: {df['brand_freq'].min():.4f}")
        print(f"    Max: {df['brand_freq'].max():.4f}")
        
        print(f"  Split distribution: {df['split'].value_counts().to_dict()}")
        print(f"  Top brands: {df['brand_name'].value_counts().head().to_dict()}")
        print(f"  Countries: {df['country'].value_counts().head().to_dict()}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        print(f"\nERROR: Data generation failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
