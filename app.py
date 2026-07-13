import os
import sys

# Hugging Face Spaces runs this root-level app.py from the project root.
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Set non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import lightkurve as lk
import tensorflow as tf
from astropy.timeseries import LombScargle
import gradio as gr

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
        
    safe_star_name = star_name.replace(' ', '_')
    temp_csv_path = os.path.join(project_root, f"temp_{safe_star_name}.csv")
    plot_path = os.path.join(project_root, f"temp_plot_{safe_star_name}.png")
    
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
            result_text = f"PLANET DETECTED\nConfidence: {confidence:.2f}% (Probability: {prob:.4f})"
        else:
            confidence = (1.0 - prob) * 100
            result_text = f"No planet detected\nConfidence: {confidence:.2f}% (Probability: {prob:.4f})"
            
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
            sort_idx = np.argsort(phase)
            phase_sorted = phase[sort_idx]
            flux_sorted = clean_flux[sort_idx]
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
        
        # Save figure to file to avoid JavaScript canvas errors in Gradio gr.Plot component
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        return result_text, plot_path
        
    except Exception as e:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        print(f"[ERROR] failed: {e}")
        return f"Error executing exoplanet detection pipeline: {e}", None

# HTML header for dark mode and typography
head_html = """
<script>
// Force Gradio into Dark Mode to enable high-contrast light text on dark background
document.documentElement.classList.add('dark');
</script>
<style>
/* Font import for premium typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
</style>
"""

# Custom CSS for full screen and glassmorphism styling
custom_css = """
body {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 30rem),
        linear-gradient(135deg, #020617 0%, #111827 52%, #030712 100%) !important;
}

/* Full screen layout container - overrides max-width restrictions */
.gradio-container, .gradio-container > div, .gradio-container > div > div {
    max-width: none !important;
    width: 100% !important;
}

.gradio-container {
    background: transparent !important;
    margin: 0 !important;
    padding: 2rem !important;
    position: relative !important;
    z-index: 1 !important;
}

footer, .built-with, [data-testid="footer"] {
    display: none !important;
}

/* Glassmorphism styling for all card elements */
.block, .gr-box, .gr-input, .gr-button, .gr-form, .gr-panel, .gr-card, .gr-label, .gr-plot, .examples {
    border-radius: 16px !important;
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Force text colors inside inputs, outputs, labels, and lists to be white */
.block label span, .block span, .block div, .block p, .block input, .block textarea, .block li, .block ol, .block h1, .block h2, .block h3, .block label {
    color: #ffffff !important;
}

/* Fix specific colors for form labels and descriptions */
.block .info, .block .label-content {
    color: #cccccc !important;
}

/* Text fields inside the glass blocks */
input[type="text"], textarea {
    background: rgba(0, 0, 0, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* Placeholder color styling */
input::placeholder, textarea::placeholder {
    color: #aaaaaa !important;
}

/* Center main header without blue glow */
#main-header {
    text-align: center !important;
    margin-bottom: 2.5rem !important;
    color: #ffffff !important;
}

h1, h2, h3 {
    text-align: center !important;
    color: #ffffff !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-top: 0.5rem !important;
    color: #ffffff !important;
}

/* Style main CTA button */
button.primary {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.7) !important;
    transform: translateY(-1.5px) !important;
}

/* Examples buttons container styling and hover effects */
.examples button, .gr-button {
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #e2e8f0 !important;
    transition: all 0.2s ease !important;
}

.examples button:hover, .gr-button:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important;
}

/* Image container formatting for plot output */
.gr-plot, [data-testid="plot-container"], div.svelte-10zdzc3, img.svelte-mv3sg8 {
    border-radius: 16px !important;
    overflow: hidden !important;
}
"""

with gr.Blocks(head=head_html, css=custom_css, theme=gr.themes.Default(primary_hue="blue", secondary_hue="indigo")) as demo:
    # Title and description text
    gr.Markdown(
        """
        # Exoplanet Transit Detection AI Pipeline
        
        This tool implements an end-to-end deep learning pipeline to detect exoplanets from live Kepler space telescope data.
        
        1. Live Download: Fetches high-precision raw light curves from NASA servers via lightkurve.
        2. Denoising: Passes the preprocessed curves through a 1D Convolutional Autoencoder to filter out stellar noise and cosmic rays.
        3. Classification: Evaluates the denoised curve using a 1D CNN Binary Classifier to calculate planet transit probabilities.
        """,
        elem_id="main-header"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Star Name", 
                placeholder="Try: Kepler-452, Kepler-10, TRAPPIST-1",
                info="Type the catalog name of a star observed by the Kepler mission (e.g. Kepler-452, Kepler-10, TRAPPIST-1, or KIC IDs)."
            )
            
            submit_btn = gr.Button("Analyze Live Data", variant="primary")
            
            # Example stars
            gr.Examples(
                examples=["Kepler-452", "Kepler-10", "KIC 3733346"],
                inputs=input_text,
                label="Example Stars (Click to Test)"
            )
            
        with gr.Column(scale=2):
            output_text = gr.Label(label="Detection Prediction")
            output_plot = gr.Image(type="filepath", label="Diagnostic Plots (Raw, Denoised, and Phase-folded)")
            
    submit_btn.click(
        fn=detect_exoplanet, 
        inputs=input_text, 
        outputs=[output_text, output_plot]
    )

if __name__ == "__main__":
    demo.launch()
