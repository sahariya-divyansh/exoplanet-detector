import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import warnings

# Suppress Keras warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NoisySequence(Sequence):
    """
    On-the-fly noisy data generator for Keras training.
    """
    def __init__(self, X, batch_size=32, shuffle=True):
        self.X = X
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_clean = self.X[batch_indices]
        
        # Clone clean curves to add noise
        X_noisy = np.copy(X_clean)
        
        for i in range(len(X_noisy)):
            # 1. Add Gaussian noise (std between 0.01 and 0.05)
            noise_std = np.random.uniform(0.01, 0.05)
            gaussian_noise = np.random.normal(0, noise_std, X_noisy[i].shape)
            X_noisy[i] += gaussian_noise
            
            # 2. Inject cosmic ray spikes (~5% probability per point)
            spike_mask = np.random.random(X_noisy[i].shape) < 0.05
            spikes = np.random.choice([-1.0, 1.0], size=X_noisy[i].shape) * np.random.uniform(0.3, 0.8, size=X_noisy[i].shape)
            X_noisy[i][spike_mask] += spikes[spike_mask]
            
        # Clip back to [0.0, 1.0] since outputs are sigmoid bounded
        X_noisy = np.clip(X_noisy, 0.0, 1.0)
        
        # Add channel dimension (N, 2000, 1) required for Conv1D
        return X_noisy[..., np.newaxis], X_clean[..., np.newaxis]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_denoiser_autoencoder():
    """
    1D Convolutional Autoencoder for light curve denoising.
    """
    inputs = layers.Input(shape=(2000, 1))
    
    # Encoder
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    # Decoder
    x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    
    # Output (Sigmoid activation to force output into [0.0, 1.0])
    outputs = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, outputs, name="Convolutional_Denoising_Autoencoder")
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    print("--- Exoplanet AI Hackathon: Denoising Autoencoder Training ---")
    
    # Load dataset splits
    train_path = os.path.join("data", "processed", "X_train.npy")
    val_path = os.path.join("data", "processed", "X_val.npy")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[ERROR] Processed data files not found. Please run preprocess_data.py first.")
        sys.exit(1)
        
    X_train = np.load(train_path)
    X_val = np.load(val_path)
    
    print(f"Loaded datasets:")
    print(f"  Training curves:   {X_train.shape}")
    print(f"  Validation curves: {X_val.shape}")
    
    # Initialize generators
    train_seq = NoisySequence(X_train, batch_size=32, shuffle=True)
    val_seq = NoisySequence(X_val, batch_size=32, shuffle=False)
    
    # Build model
    model = build_denoiser_autoencoder()
    model.summary()
    
    # Set callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    # Train
    epochs = 50
    print(f"\nTraining model for up to {epochs} epochs...")
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Print final validation loss
    final_val_loss = model.evaluate(val_seq, verbose=0)
    print("\n" + "="*50)
    print(f"TRAINING COMPLETE. Best Validation MSE Loss: {final_val_loss:.6f}")
    print("="*50)
    
    # Plot & save loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='#e74c3c', linewidth=1.5)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#3498db', linewidth=1.5)
    plt.title('Denoising Autoencoder Training History', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    history_plot_path = os.path.join("models", "denoiser_training_history.png")
    plt.savefig(history_plot_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {history_plot_path}")
    
    # Plot noisy input vs. denoised output vs. clean target for 3 random validation samples
    print("\nGenerating sample denoising comparison plots on validation set...")
    val_indices = np.random.choice(len(X_val), size=3, replace=False)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i, idx in enumerate(val_indices):
        clean_lc = X_val[idx]
        
        # Construct noisy version (using the same logic as generator)
        noise_std = np.random.uniform(0.01, 0.05)
        noisy_lc = clean_lc + np.random.normal(0, noise_std, clean_lc.shape)
        spike_mask = np.random.random(noisy_lc.shape) < 0.05
        spikes = np.random.choice([-1.0, 1.0], size=noisy_lc.shape) * np.random.uniform(0.3, 0.8, size=noisy_lc.shape)
        noisy_lc[spike_mask] += spikes[spike_mask]
        noisy_lc = np.clip(noisy_lc, 0.0, 1.0)
        
        # Predict denoised version
        denoised_lc = model.predict(noisy_lc[np.newaxis, ..., np.newaxis], verbose=0)[0, :, 0]
        
        # Plot
        ax = axes[i]
        ax.plot(noisy_lc, color='#bdc3c7', alpha=0.6, label='Noisy Input (Gaussian + Spikes)')
        ax.plot(clean_lc, color='#2ecc71', alpha=0.8, linewidth=1.5, label='Original Clean Target')
        ax.plot(denoised_lc, color='#2980b9', linewidth=1.8, label='Denoised Reconstruction')
        ax.set_title(f"Validation Sample #{idx}", fontsize=10, fontweight='semibold')
        ax.set_ylabel('Normalized Flux (0-1)')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel('Time Bin (Index)')
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    results_plot_path = os.path.join("models", "denoiser_sample_results.png")
    plt.savefig(results_plot_path, dpi=150)
    plt.close()
    print(f"Saved sample results comparison to {results_plot_path}")
    
    # Save the trained model
    saved_model_dir = os.path.join("models", "saved_models")
    os.makedirs(saved_model_dir, exist_ok=True)
    model_path = os.path.join(saved_model_dir, "denoiser.keras")
    model.save(model_path)
    print(f"\nModel saved successfully as: {model_path}")

if __name__ == "__main__":
    main()
