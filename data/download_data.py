import os
import sys
import numpy as np
import pandas as pd
import lightkurve as lk
import warnings

# Suppress lightkurve and astropy warnings to keep output clean
warnings.filterwarnings("ignore")

def download_target_lightcurve(target_name, output_path):
    print(f"Searching for '{target_name}'...")
    try:
        # Search for long cadence light curves
        search_result = lk.search_lightcurve(target_name, cadence='long')
        
        if len(search_result) == 0:
            # Fallback: search without cadence restriction
            search_result = lk.search_lightcurve(target_name)
            
        if len(search_result) == 0:
            print(f"  [ERROR] No light curve products found for {target_name}.")
            return False, 0
        
        print(f"  Found {len(search_result)} data products. Downloading the first one...")
        
        # Download the first light curve in the search results
        lc = search_result[0].download()
        
        if lc is None:
            print(f"  [ERROR] Download returned None for {target_name}.")
            return False, 0
            
        # Extract time, flux, and flux_err values
        time_vals = np.array(lc.time.value, dtype=float)
        flux_vals = np.array(lc.flux.value, dtype=float)
        flux_err_vals = np.array(lc.flux_err.value, dtype=float)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_vals,
            'flux': flux_vals,
            'flux_err': flux_err_vals
        })
        
        # Drop rows with missing values
        df_cleaned = df.dropna()
        num_points = len(df_cleaned)
        
        if num_points == 0:
            print(f"  [ERROR] No valid data points left after cleaning for {target_name}.")
            return False, 0
            
        # Save to CSV
        df_cleaned.to_csv(output_path, index=False)
        print(f"  [SUCCESS] Saved {num_points} data points to {output_path}")
        return True, num_points
        
    except Exception as e:
        print(f"  [ERROR] Failed to download or process {target_name}: {e}")
        return False, 0

def main():
    print("Starting exoplanet host and control stars data download...")
    
    # Define directories
    raw_dir = os.path.join("data", "raw_lightcurves")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Define targets
    positive_targets = {
        'Kepler-452': 'kepler452_positive.csv',
        'Kepler-10': 'kepler10_positive.csv',
        'Kepler-8': 'kepler8_positive.csv',
        'TRAPPIST-1': 'trappist1_positive.csv',
        'Kepler-90': 'kepler90_positive.csv'
    }
    
    negative_targets = {
        'KIC 3733346': 'star1_negative.csv',
        'KIC 5088536': 'star2_negative.csv',
        'KIC 8462852': 'star3_negative.csv',
        'KIC 1026474': 'star4_negative.csv',
        'KIC 12009504': 'star5_negative.csv'
    }
    
    summary = []
    downloaded_count = 0
    
    # Process positive targets
    print("\n=== Downloading Confirmed Exoplanet Hosts (Positive Examples) ===")
    for target, filename in positive_targets.items():
        out_path = os.path.join(raw_dir, filename)
        success, points = download_target_lightcurve(target, out_path)
        if success:
            downloaded_count += 1
            summary.append({'target': target, 'class': 'Positive', 'points': points, 'saved_file': filename})
            
    # Process negative targets
    print("\n=== Downloading Control Stars (Negative Examples) ===")
    for target, filename in negative_targets.items():
        out_path = os.path.join(raw_dir, filename)
        success, points = download_target_lightcurve(target, out_path)
        if success:
            downloaded_count += 1
            summary.append({'target': target, 'class': 'Negative', 'points': points, 'saved_file': filename})
            
    # Print summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total stars successfully downloaded: {downloaded_count}/{len(positive_targets) + len(negative_targets)}")
    print(f"{'Target Star':<15} | {'Class':<10} | {'Data Points':<12} | {'Filename':<25}")
    print("-" * 70)
    for item in summary:
        print(f"{item['target']:<15} | {item['class']:<10} | {item['points']:<12} | {item['saved_file']:<25}")
    print("="*50)
    
if __name__ == "__main__":
    main()
