"""
Generate dummy data for testing the RBF-SVM Brand Verification Model.

This script creates a synthetic dataset with realistic brand verification data
including text features, numerical features, and proper class imbalance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_brand_names() -> List[str]:
    """Generate realistic brand names."""
    prefixes = ['Tech', 'Digital', 'Smart', 'Global', 'Prime', 'Elite', 'Pro', 'Max', 'Ultra', 'Super']
    suffixes = ['Solutions', 'Systems', 'Corp', 'Inc', 'Ltd', 'Group', 'Works', 'Lab', 'Hub', 'Zone']
    industries = ['Fashion', 'Food', 'Auto', 'Home', 'Health', 'Beauty', 'Travel', 'Games', 'Books', 'Music']
    
    brands = []
    
    # Popular brands (high frequency)
    popular_brands = [
        'Nike', 'Apple', 'Google', 'Amazon', 'Microsoft', 'Samsung', 'Sony', 'Dell', 'HP', 'Intel',
        'McDonald\'s', 'Coca-Cola', 'Pepsi', 'Starbucks', 'Walmart', 'Target', 'BMW', 'Toyota',
        'Mercedes', 'Ford', 'Adidas', 'Puma', 'Zara', 'H&M', 'Netflix', 'Spotify', 'Adobe', 'Oracle'
    ]
    
    # Add popular brands multiple times (they appear frequently)
    for brand in popular_brands:
        frequency = np.random.randint(10, 50)  # High frequency
        brands.extend([brand] * frequency)
    
    # Generate medium frequency brands
    for _ in range(50):
        brand = f"{random.choice(prefixes)} {random.choice(industries)}"
        frequency = np.random.randint(3, 15)  # Medium frequency
        brands.extend([brand] * frequency)
    
    # Generate long-tail brands (low frequency)
    for _ in range(200):
        brand = f"{random.choice(prefixes)}{random.choice(suffixes)}"
        frequency = np.random.randint(1, 3)  # Low frequency (long-tail)
        brands.extend([brand] * frequency)
    
    # Add some completely unique brands (frequency = 1)
    for i in range(100):
        brand = f"UniqueShop{i:03d}"
        brands.append(brand)
    
    return brands

def generate_merchant_names(brand_names: List[str]) -> List[str]:
    """Generate merchant names based on brand names with some variations."""
    merchant_names = []
    
    variations = ['', ' Store', ' Shop', ' Outlet', ' Official', ' Direct', ' Online', ' Market']
    noise_words = ['The', 'Best', 'Premium', 'Quality', 'Top', 'Original', 'Authentic']
    
    for brand in brand_names:
        if np.random.random() < 0.7:  # 70% exact match
            merchant_name = brand + random.choice(variations)
        else:  # 30% with some noise
            if np.random.random() < 0.5:
                merchant_name = f"{random.choice(noise_words)} {brand}"
            else:
                merchant_name = brand.replace(' ', random.choice(['', '-', '_']))
        
        merchant_names.append(merchant_name)
    
    return merchant_names

def generate_addresses(n: int) -> List[str]:
    """Generate normalized address strings."""
    streets = ['Main St', 'Oak Ave', 'Park Rd', 'First St', 'Second Ave', 'Market St', 'Hill Rd']
    numbers = range(1, 9999)
    
    addresses = []
    for _ in range(n):
        if np.random.random() < 0.1:  # 10% empty addresses
            addresses.append('')
        else:
            num = random.choice(numbers)
            street = random.choice(streets)
            addresses.append(f"{num} {street}")
    
    return addresses

def generate_countries_and_cities(n: int) -> Tuple[List[str], List[str]]:
    """Generate countries and corresponding cities."""
    locations = {
        'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia'],
        'UK': ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Liverpool', 'Bristol'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa', 'Edmonton'],
        'Germany': ['Berlin', 'Munich', 'Hamburg', 'Cologne', 'Frankfurt', 'Stuttgart'],
        'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes'],
        'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Gold Coast']
    }
    
    countries = []
    cities = []
    
    for _ in range(n):
        country = random.choice(list(locations.keys()))
        city = random.choice(locations[country])
        countries.append(country)
        cities.append(city)
    
    return countries, cities

def generate_websites(brand_names: List[str]) -> List[str]:
    """Generate website URLs for brands."""
    tlds = ['.com', '.net', '.org', '.co.uk', '.de', '.fr']
    
    websites = []
    for brand in brand_names:
        if np.random.random() < 0.2:  # 20% no website
            websites.append('')
        else:
            # Clean brand name for URL
            clean_brand = brand.lower().replace(' ', '').replace('\'', '')
            tld = random.choice(tlds)
            websites.append(f"www.{clean_brand}{tld}")
    
    return websites

def calculate_brand_frequencies(brand_names: List[str]) -> List[int]:
    """Calculate frequency of each brand in the dataset."""
    brand_counts = {}
    for brand in brand_names:
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    return [brand_counts[brand] for brand in brand_names]

def generate_website_match_scores(brand_names: List[str], websites: List[str]) -> List[float]:
    """Generate website match scores based on brand-website similarity."""
    scores = []
    
    for brand, website in zip(brand_names, websites):
        if not website:  # No website
            scores.append(0.0)
        else:
            # Calculate similarity score
            clean_brand = brand.lower().replace(' ', '').replace('\'', '')
            clean_website = website.replace('www.', '').replace('.com', '').replace('.net', '').replace('.org', '')
            
            if clean_brand in clean_website:
                score = np.random.uniform(0.8, 1.0)  # High match
            elif any(word in clean_website for word in clean_brand.split() if len(word) > 2):
                score = np.random.uniform(0.4, 0.8)  # Medium match
            else:
                score = np.random.uniform(0.0, 0.4)  # Low match
            
            scores.append(round(score, 3))
    
    return scores

def generate_targets_and_weights(brand_frequencies: List[int], website_scores: List[float]) -> Tuple[List[str], List[float]]:
    """Generate verification targets and calculated weights."""
    targets = []
    weights = []
    
    for freq, web_score in zip(brand_frequencies, website_scores):
        # Higher frequency and website match increase verification probability
        base_prob = 0.3  # Base verification rate
        freq_boost = min(freq / 50.0, 0.4)  # Frequency contribution
        web_boost = web_score * 0.3  # Website match contribution
        
        verification_prob = base_prob + freq_boost + web_boost
        verification_prob = min(verification_prob, 0.9)  # Cap at 90%
        
        is_verified = np.random.random() < verification_prob
        targets.append('verified' if is_verified else 'not_verified')
        
        # Calculate weights (emphasize long-tail brands)
        if freq == 1:  # Unique brands get highest weight
            weight = np.random.uniform(2.0, 3.0)
        elif freq <= 5:  # Low frequency brands
            weight = np.random.uniform(1.5, 2.5)
        elif freq <= 20:  # Medium frequency brands
            weight = np.random.uniform(1.0, 1.8)
        else:  # High frequency brands
            weight = np.random.uniform(0.8, 1.2)
        
        weights.append(round(weight, 3))
    
    return targets, weights

def generate_train_test_split(n: int) -> List[str]:
    """Generate train/test/validate split."""
    # 70% train, 15% validate, 15% test
    splits = ['TRAIN'] * int(0.7 * n) + ['VALIDATE'] * int(0.15 * n) + ['TEST'] * int(0.15 * n)
    
    # Handle remainder
    while len(splits) < n:
        splits.append('TRAIN')
    
    # Shuffle the splits
    random.shuffle(splits)
    return splits[:n]

def main():
    """Generate the dummy dataset."""
    print("Generating dummy brand verification dataset...")
    
    # Generate brand names (this creates the base distribution)
    brand_names = generate_brand_names()
    
    # Shuffle to randomize order
    random.shuffle(brand_names)
    
    # Take first 1000 for our dataset
    n_samples = 1000
    brand_names = brand_names[:n_samples]
    
    print(f"Generating {n_samples} samples...")
    
    # Generate other features
    merchant_names = generate_merchant_names(brand_names)
    addresses = generate_addresses(n_samples)
    countries, cities = generate_countries_and_cities(n_samples)
    websites = generate_websites(brand_names)
    
    # Calculate derived features
    brand_frequencies = calculate_brand_frequencies(brand_names)
    website_scores = generate_website_match_scores(brand_names, websites)
    targets, weights = generate_targets_and_weights(brand_frequencies, website_scores)
    splits = generate_train_test_split(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'brand_name': brand_names,
        'brand_freq': brand_frequencies,
        'snowdrop_name': merchant_names,  # Merchant name from Snowdrop system
        'address_norm': addresses,
        'country': countries,
        'city': cities,
        'website': websites,
        'website_match': website_scores,
        'target': targets,
        'calculated_weights': weights,
        'split': splits
    })
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['address_norm', 'website', 'city'])
        df.at[idx, col] = np.nan
    
    # Save to CSV
    output_path = Path('data/dummy_data_1k.csv')
    output_path.parent.mkdir(exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:")
    print(df['target'].value_counts())
    print(f"Split distribution:")
    print(df['split'].value_counts())
    print(f"Brand frequency stats:")
    print(df['brand_freq'].describe())
    print(f"Sample of data:")
    print(df.head())

if __name__ == "__main__":
    main()
