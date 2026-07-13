import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import tensorflow as tf
from astropy.timeseries import LombScargle
import gradio as gr

# Add project root to sys.path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from data.preprocess_data import preprocess_lightcurve

# Load the trained models using absolute paths
print("Loading trained models...")
denoiser_path = os.path.join(project_root, "models", "saved_models", "denoiser.keras")
classifier_path = os.path.join(project_root, "models", "saved_models", "classifier.keras")

if not os.path.exists(denoiser_path) or not os.path.exists(classifier_path):
    print(f"[ERROR] Models not found. Ensure train_denoiser.py and train_classifier.py have been executed.")
    sys.exit(1)

denoiser = tf.keras.models.load_model(denoiser_path)
classifier = tf.keras.models.load_model(classifier_path)
print("Models loaded successfully!")

def detect_exoplanet(star_name):
    # Strip whitespace
    star_name = star_name.strip()
    if not star_name:
        return "Please enter a valid star name.", None
        
    temp_csv_path = f"temp_{star_name.replace(' ', '_')}.csv"
    
    try:
        # Search for long cadence light curves
        print(f"Searching light curve for {star_name}...")
        search_result = lk.search_lightcurve(star_name, cadence='long')
        if len(search_result) == 0:
            search_result = lk.search_lightcurve(star_name)
            
        if len(search_result) == 0:
            return f"Error: No light curve products found for target '{star_name}' on NASA servers.", None
            
        print("Downloading light curve...")
        lc = search_result[0].download()
        if lc is None:
            return f"Error: Download returned None for target '{star_name}'. Try another star.", None
            
        # Extract columns
        time_vals = np.array(lc.time.value, dtype=float)
        flux_vals = np.array(lc.flux.value, dtype=float)
        flux_err_vals = np.array(lc.flux_err.value, dtype=float)
        
        # Save to temporary CSV for preprocessing pipeline compatibility
        df = pd.DataFrame({
            'time': time_vals,
            'flux': flux_vals,
            'flux_err': flux_err_vals
        }).dropna()
        
        if len(df) == 0:
            return f"Error: No valid data points found in light curve for '{star_name}'.", None
            
        df.to_csv(temp_csv_path, index=False)
        
        # Preprocess using preprocess_lightcurve function
        print("Preprocessing light curve...")
        raw_time, raw_flux, clean_time, clean_flux, resampled_time, resampled_flux = preprocess_lightcurve(temp_csv_path)
        
        # Cleanup temp file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            
        # Denoiser expects input shape (1, 2000, 1)
        print("Denoising...")
        flux_denoised = denoiser.predict(resampled_flux[np.newaxis, ..., np.newaxis], verbose=0).flatten()
        
        # Classifier expects input shape (1, 2000, 1)
        # Shift denoised flux by subtracting 0.5 (centering) to match training preprocess
        flux_denoised_centered = flux_denoised - 0.5
        print("Classifying...")
        prob = classifier.predict(flux_denoised_centered[np.newaxis, ..., np.newaxis], verbose=0).flatten()[0]
        
        # Prediction label at tuned threshold (0.50)
        threshold = 0.50
        is_planet = prob >= threshold
        
        if is_planet:
            confidence = prob * 100
            result_text = f"🪐 PLANET DETECTED\nConfidence: {confidence:.2f}% (Probability: {prob:.4f})"
        else:
            confidence = (1.0 - prob) * 100
            result_text = f"✨ No planet detected\nConfidence: {confidence:.2f}% (Probability: {prob:.4f})"
            
        # Build matplotlib 3-panel plot
        print("Plotting figures...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Panel 1: Raw light curve
        ax1.plot(clean_time, clean_flux, color='#bdc3c7', alpha=0.8, linewidth=0.8, label='Raw Flux (cleaned)')
        ax1.set_title(f"Raw Light Curve for {star_name}", fontsize=11, fontweight='semibold')
        ax1.set_ylabel("Flux (e-/s)")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        
        # Panel 2: Denoised light curve (0-1 Normalized)
        # Plot both the normalized resampled input and the denoised reconstruction
        ax2.plot(resampled_time, resampled_flux, color='#e74c3c', alpha=0.5, linewidth=0.8, label='Normalized Input')
        ax2.plot(resampled_time, flux_denoised, color='#2980b9', linewidth=1.5, label='Denoised Signal')
        ax2.set_title("Denoised Light Curve (Normalized 0-1 scale)", fontsize=11, fontweight='semibold')
        ax2.set_ylabel("Normalized Flux")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        
        # Panel 3: Phase-folded view if a period can be estimated
        try:
            ls = LombScargle(clean_time, clean_flux)
            frequency, power = ls.autopower(minimum_frequency=1/20.0, maximum_frequency=1/0.5)
            best_frequency = frequency[np.argmax(power)]
            best_period = 1.0 / best_frequency
            
            phase = (clean_time % best_period) / best_period
            
            ax3.scatter(phase, clean_flux, color='#8e44ad', s=3, alpha=0.4, label='Data points')
            # Add a smoothed trend line on phase-folded plot for visual aid
            # Sort phase and plot running average
            sort_idx = np.argsort(phase)
            phase_sorted = phase[sort_idx]
            flux_sorted = clean_flux[sort_idx]
            # Simple rolling mean using pandas
            smoothed_flux = pd.Series(flux_sorted).rolling(window=max(2, len(flux_sorted)//20), center=True).mean().values
            ax3.plot(phase_sorted, smoothed_flux, color='#2c3e50', linewidth=1.5, label='Smoothed Trend')
            
            ax3.set_title(f"Phase-Folded Light Curve (Estimated Period: {best_period:.3f} days)", fontsize=11, fontweight='semibold')
            ax3.set_xlabel("Phase")
            ax3.set_ylabel("Flux (e-/s)")
            ax3.grid(True, linestyle='--', alpha=0.5)
            ax3.legend()
        except Exception as pe:
            ax3.text(0.5, 0.5, f"Could not estimate period for phase folding:\n{pe}", ha='center', va='center', fontsize=10, color='gray')
            ax3.set_title("Phase-Folded Light Curve", fontsize=11, fontweight='semibold')
            
        plt.tight_layout()
        
        return result_text, fig
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        print(f"[ERROR] failed: {e}")
        return f"Error executing exoplanet detection pipeline: {e}", None

# Create Gradio App with custom layout and theme
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 🪐 Exoplanet Transit Detection AI Pipeline
        ### Built for **Bharatiya Antariksh Hackathon 2026**
        This tool implements an end-to-end deep learning pipeline to detect exoplanets from live Kepler space telescope data.
        1. **Live Download**: Fetches high-precision raw light curves from NASA servers via *lightkurve*.
        2. **Denoising**: Passes the preprocessed curves through a **1D Convolutional Autoencoder** to filter out stellar noise and cosmic rays.
        3. **Classification**: Evaluates the denoised curve using a **1D CNN Binary Classifier** to calculate planet transit probabilities.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Star Name", 
                placeholder="Try: Kepler-452, Kepler-10, TRAPPIST-1",
                info="Type the catalog name of a star observed by the Kepler mission (e.g. Kepler-452, Kepler-10, TRAPPIST-1, or KIC IDs)."
            )
            
            submit_btn = gr.Button("Analyze Live Data", variant="primary")
            
            # Predefined examples
            gr.Examples(
                examples=["Kepler-452", "Kepler-10", "KIC 3733346"],
                inputs=input_text,
                label="Example Stars (Click to Test)"
            )
            
        with gr.Column(scale=2):
            output_text = gr.Label(label="Detection Prediction")
            output_plot = gr.Plot(label="Diagnostic Plots (Raw, Denoised, and Phase-folded)")
            
    submit_btn.click(
        fn=detect_exoplanet, 
        inputs=input_text, 
        outputs=[output_text, output_plot]
    )

if __name__ == "__main__":
    # Launch with public share link active
    demo.launch(share=True)
