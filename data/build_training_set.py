import os
import sys
import time
import random
import numpy as np
import pandas as pd
import lightkurve as lk
import warnings

# Suppress lightkurve and astropy warnings to keep output clean
warnings.filterwarnings("ignore")

# For reproducibility
random.seed(42)
np.random.seed(42)

def main():
    print("--- Exoplanet AI Hackathon: Training Set Builder ---")
    
    # 1. Download the NASA Exoplanet Archive KOI cumulative table
    tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,koi_disposition+from+cumulative&format=csv"
    print("Downloading KOI cumulative table from NASA Exoplanet Archive...")
    try:
        df_koi = pd.read_csv(tap_url)
        print(f"Successfully loaded {len(df_koi)} KOI entries.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to download KOI table: {e}")
        sys.exit(1)
        
    # 2. Filter & Aggregate by kepid
    # A single star (kepid) can have multiple planetary candidates (KOIs).
    # Group by kepid and categorize:
    # - CONFIRMED (1): If the star has at least one confirmed exoplanet.
    # - FALSE POSITIVE (0): If the star has NO confirmed exoplanets and at least one false positive.
    kepid_groups = df_koi.groupby('kepid')['koi_disposition'].apply(list).to_dict()
    
    confirmed_kepids = []
    fp_kepids = []
    
    for kepid, dispositions in kepid_groups.items():
        if 'CONFIRMED' in dispositions:
            confirmed_kepids.append(kepid)
        elif 'FALSE POSITIVE' in dispositions:
            # Must have at least one false positive and zero confirmed planets
            fp_kepids.append(kepid)
            
    print(f"Found unique stars: {len(confirmed_kepids)} Confirmed, {len(fp_kepids)} False Positive.")
    
    # 3. Sample 200 from each category
    sample_size = 200
    if len(confirmed_kepids) < sample_size or len(fp_kepids) < sample_size:
        print(f"[ERROR] Insufficient unique stars to sample {sample_size} of each.")
        sys.exit(1)
        
    sampled_confirmed = random.sample(confirmed_kepids, sample_size)
    sampled_fp = random.sample(fp_kepids, sample_size)
    
    # Combine samples with labels (confirmed = 1, false_positive = 0)
    targets = []
    for kepid in sampled_confirmed:
        targets.append((kepid, 1, 'confirmed'))
    for kepid in sampled_fp:
        targets.append((kepid, 0, 'falsepositive'))
        
    # Shuffle targets to mix positive and negative cases
    random.shuffle(targets)
    
    # Create raw data folder if it doesn't exist
    raw_dir = os.path.join("data", "raw_lightcurves")
    os.makedirs(raw_dir, exist_ok=True)
    
    successful_downloads = []
    
    print(f"\nStarting download of {len(targets)} light curves with 1-2s delay between requests...")
    
    for idx, (kepid, label, label_str) in enumerate(targets, 1):
        filename = f"{kepid}_{label_str}.csv"
        filepath = os.path.join(raw_dir, filename)
        
        success = False
        num_points = 0
        
        try:
            # Search for long cadence light curves for this KIC ID
            search_result = lk.search_lightcurve(f"KIC {kepid}", cadence='long')
            
            if len(search_result) > 0:
                # Download the first light curve
                lc = search_result[0].download()
                
                if lc is not None:
                    time_vals = np.array(lc.time.value, dtype=float)
                    flux_vals = np.array(lc.flux.value, dtype=float)
                    flux_err_vals = np.array(lc.flux_err.value, dtype=float)
                    
                    df_lc = pd.DataFrame({
                        'time': time_vals,
                        'flux': flux_vals,
                        'flux_err': flux_err_vals
                    }).dropna()
                    
                    num_points = len(df_lc)
                    if num_points > 0:
                        df_lc.to_csv(filepath, index=False)
                        success = True
                    else:
                        print(f"  [WARN] Star KIC {kepid} had no data points after cleaning.")
            else:
                print(f"  [WARN] No long-cadence light curve found for KIC {kepid}.")
                
        except Exception as e:
            # Log failure but continue
            print(f"  [ERROR] Failed downloading KIC {kepid}: {e}")
            
        if success:
            successful_downloads.append({
                'kepid': kepid,
                'label': label,
                'filename': filename,
                'data_points': num_points
            })
            
        # Print progress every 20 stars
        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(targets)} processed ({len(successful_downloads)} successful)...")
            
        # 1-2 seconds delay to be polite to NASA servers
        time.sleep(random.uniform(1.0, 2.0))
        
    # Write labels.csv
    labels_df = pd.DataFrame(successful_downloads)
    labels_path = os.path.join("data", "labels.csv")
    labels_df.to_csv(labels_path, index=False)
    
    print("\n" + "="*50)
    print("BUILD TRAINING SET COMPLETED")
    print("="*50)
    print(f"Requested:          {len(targets)}")
    print(f"Downloaded:         {len(successful_downloads)}")
    print(f"Saved labels to:    {labels_path}")
    print("="*50)

if __name__ == "__main__":
    main()
