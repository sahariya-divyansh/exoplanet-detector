import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings to keep stdout clean
warnings.filterwarnings("ignore")

def preprocess_lightcurve(filepath):
    """
    Load a raw light curve, apply sigma clipping, flatten the trend, 
    normalize to 0-1, and resample to a fixed length of 2000 points.
    """
    df = pd.read_csv(filepath)
    raw_time = df['time'].values
    raw_flux = df['flux'].values
    
    # 1. Sigma Clipping (Outlier Removal) - remove points beyond 3 standard deviations
    median_flux = np.median(raw_flux)
    std_flux = np.std(raw_flux)
    outlier_mask = np.abs(raw_flux - median_flux) < 3 * std_flux
    
    time_clean = raw_time[outlier_mask]
    flux_clean = raw_flux[outlier_mask]
    
    if len(flux_clean) < 10:
        raise ValueError(f"Too few data points ({len(flux_clean)}) after outlier removal.")
        
    # 2. Flattening (Savitzky-Golay filter to remove long-term trends)
    # Window size should be ~401, but adjusted down if array is shorter
    window_len = 401
    if len(flux_clean) <= window_len:
        window_len = len(flux_clean) // 2 * 2 - 1  # Largest odd number smaller than len
        if window_len < 3:
            window_len = 3
            
    trend = savgol_filter(flux_clean, window_length=window_len, polyorder=2)
    flux_flattened = flux_clean / trend
    
    # 3. Normalization (Min-Max scale to 0-1 range)
    f_min = np.min(flux_flattened)
    f_max = np.max(flux_flattened)
    flux_normalized = (flux_flattened - f_min) / (f_max - f_min + 1e-8)
    
    # 4. Interpolation & Resampling to a fixed length of 2000 points
    new_time = np.linspace(time_clean.min(), time_clean.max(), 2000)
    flux_resampled = np.interp(new_time, time_clean, flux_normalized)
    
    # Clip any potential out-of-bounds values from interpolation
    flux_resampled = np.clip(flux_resampled, 0.0, 1.0)
    
    return raw_time, raw_flux, time_clean, flux_clean, new_time, flux_resampled

def save_comparison_plot(raw_time, raw_flux, clean_time, clean_flux, resampled_time, resampled_flux, title, output_path):
    """
    Generate and save a before/after visualization of the preprocessing steps.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.suptitle(title, fontsize=13, fontweight='bold', color='#2c3e50')
    
    # Raw light curve plot (with outliers highlighted)
    ax1.plot(raw_time, raw_flux, color='#bdc3c7', alpha=0.8, label='Raw Flux')
    # Find outliers
    outliers_mask = ~np.isin(raw_time, clean_time)
    if np.any(outliers_mask):
        ax1.scatter(raw_time[outliers_mask], raw_flux[outliers_mask], color='#e74c3c', s=15, label='Outliers (3σ)')
    ax1.set_title('Original Light Curve (with Outliers)', fontsize=10, fontweight='semibold')
    ax1.set_ylabel('Raw Flux (e-/s)', fontsize=9)
    ax1.legend(loc='upper right')
    
    # Processed light curve plot
    ax2.plot(resampled_time, resampled_flux, color='#2980b9', linewidth=1.2, label='Preprocessed & Resampled')
    ax2.set_title('Processed & Flattened Light Curve (2000 Uniform Points)', fontsize=10, fontweight='semibold')
    ax2.set_xlabel('Time (Days)', fontsize=9)
    ax2.set_ylabel('Normalized Flux (0-1)', fontsize=9)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    print("--- Exoplanet AI Hackathon: Data Preprocessing Pipeline ---")
    
    # Define paths
    labels_file = os.path.join("data", "labels.csv")
    raw_dir = os.path.join("data", "raw_lightcurves")
    processed_dir = os.path.join("data", "processed")
    plots_dir = os.path.join(processed_dir, "sample_plots")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(labels_file):
        print(f"[ERROR] Labels file not found: {labels_file}. Please run download scripts first.")
        sys.exit(1)
        
    # Read labels
    df_labels = pd.read_csv(labels_file)
    print(f"Loaded {len(df_labels)} targets from labels.csv.")
    
    processed_data = []
    processed_labels = []
    
    # Count of stars we want to save plots for
    plot_indices = [0, len(df_labels) // 2, len(df_labels) - 1] # First, middle, and last star
    plots_saved = 0
    
    print("\nPreprocessing light curves...")
    for idx, row in df_labels.iterrows():
        kepid = row['kepid']
        label = row['label']
        filename = row['filename']
        filepath = os.path.join(raw_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  [WARN] Light curve file not found: {filepath}. Skipping.")
            continue
            
        try:
            # Preprocess
            raw_time, raw_flux, clean_time, clean_flux, resampled_time, resampled_flux = preprocess_lightcurve(filepath)
            
            processed_data.append(resampled_flux)
            processed_labels.append(label)
            
            # Save 3 sample plots as requested
            if idx in plot_indices and plots_saved < 3:
                label_name = "Confirmed" if label == 1 else "False Positive"
                plot_filename = f"sample_{kepid}_{label_name}.png"
                plot_path = os.path.join(plots_dir, plot_filename)
                
                title = f"Preprocessing Pipeline Visualization: KIC {kepid} ({label_name})"
                save_comparison_plot(
                    raw_time, raw_flux, clean_time, clean_flux, 
                    resampled_time, resampled_flux, title, plot_path
                )
                print(f"  [SAVED PLOT] Saved diagnostic plot to {plot_path}")
                plots_saved += 1
                
        except Exception as e:
            print(f"  [ERROR] Failed to preprocess KIC {kepid}: {e}")
            
    # Convert lists to NumPy arrays
    X = np.array(processed_data)
    y = np.array(processed_labels)
    
    # Save the complete dataset
    all_data_path = os.path.join("data", "processed_lightcurves.npy")
    all_labels_path = os.path.join("data", "processed_labels.npy")
    np.save(all_data_path, X)
    np.save(all_labels_path, y)
    print(f"\nSaved complete dataset:")
    print(f"  Light curves: {all_data_path} (Shape: {X.shape})")
    print(f"  Labels:       {all_labels_path} (Shape: {y.shape})")
    
    # Train-val-test split (70% train, 15% val, 15% test)
    # Stratify by labels to ensure balanced splits in all sets
    print("\nSplitting dataset into train (70%), validation (15%), and test (15%)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    # Save splits
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    # Calculate statistics
    num_confirmed = np.sum(y == 1)
    num_fp = np.sum(y == 0)
    
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Total stars processed successfully: {len(X)}")
    print(f"Confirmed Exoplanet Hosts (1):      {num_confirmed}")
    print(f"False Positives / Controls (0):     {num_fp}")
    print("-" * 50)
    print("Dataset Split Shapes:")
    print(f"  Train Set:      X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Validation Set: X_val   {X_val.shape}, y_val   {y_val.shape}")
    print(f"  Test Set:       X_test  {X_test.shape}, y_test  {y_test.shape}")
    print("="*50)

if __name__ == "__main__":
    main()
