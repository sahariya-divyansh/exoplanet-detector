import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_transformer_classifier():
    """
    Builds a Transformer-based binary classifier for exoplanet detection.
    """
    inputs = layers.Input(shape=(2000, 1))
    
    # Conv1D first to generate local feature embeddings
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # Downsample sequence from 2000 to 250 to speed up MultiHeadAttention computation
    x = layers.MaxPooling1D(pool_size=8, padding='same')(x)
    
    # Transformer Block 1
    # MultiHeadAttention + Residual Connection + LayerNormalization
    attn_1 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_1])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Transformer Block 2
    # MultiHeadAttention + Residual Connection + LayerNormalization
    attn_2 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_2])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global Pooling to summarize sequence
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense Layers for classification with Dropout for regularization
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    # Output Layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="Transformer_Classifier")
    
    # Track accuracy, precision, and recall
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model

def main():
    print("--- Exoplanet AI Hackathon: Transformer Classifier Training ---")
    
    # 1. Load splits
    processed_dir = os.path.join("data", "processed")
    X_train_path = os.path.join(processed_dir, "X_train.npy")
    y_train_path = os.path.join(processed_dir, "y_train.npy")
    X_val_path = os.path.join(processed_dir, "X_val.npy")
    y_val_path = os.path.join(processed_dir, "y_val.npy")
    X_test_path = os.path.join(processed_dir, "X_test.npy")
    y_test_path = os.path.join(processed_dir, "y_test.npy")
    
    for path in [X_train_path, y_train_path, X_val_path, y_val_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Required dataset file not found: {path}. Run preprocess_data.py first.")
            sys.exit(1)
            
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    # 2. Load trained denoiser
    denoiser_path = os.path.join("models", "saved_models", "denoiser.keras")
    if not os.path.exists(denoiser_path):
        print(f"[ERROR] Trained denoiser model not found at {denoiser_path}. Run train_denoiser.py first.")
        sys.exit(1)
        
    print("Loading trained denoiser model...")
    denoiser = tf.keras.models.load_model(denoiser_path)
    
    # Denoise all sets before classification training
    print("Denoising training, validation, and test sets...")
    X_train_denoised = denoiser.predict(X_train[..., np.newaxis], batch_size=32, verbose=0)
    X_val_denoised = denoiser.predict(X_val[..., np.newaxis], batch_size=32, verbose=0)
    X_test_denoised = denoiser.predict(X_test[..., np.newaxis], batch_size=32, verbose=0)
    
    print(f"Denoised data shapes:")
    print(f"  Train: {X_train_denoised.shape}")
    print(f"  Val:   {X_val_denoised.shape}")
    print(f"  Test:  {X_test_denoised.shape}")
    
    # 3. Build Classifier
    classifier = build_transformer_classifier()
    classifier.summary()
    
    # Set callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1)
    ]
    
    # 4. Train model
    epochs = 60
    batch_size = 16
    print(f"\nTraining Transformer classifier for up to {epochs} epochs...")
    history = classifier.fit(
        X_train_denoised, y_train,
        validation_data=(X_val_denoised, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluate on test set
    print("\nEvaluating model on the held-out test set...")
    test_eval = classifier.evaluate(X_test_denoised, y_test, batch_size=batch_size, verbose=0)
    loss = test_eval[0]
    accuracy = test_eval[1]
    precision = test_eval[2]
    recall = test_eval[3]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print("\n" + "="*50)
    print("TEST EVALUATION PERFORMANCE")
    print("="*50)
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print("="*50)
    
    # Confusion Matrix
    y_pred = classifier.predict(X_test_denoised, batch_size=batch_size, verbose=0)
    y_pred_bin = (y_pred >= 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_test, y_pred_bin)
    print("\nConfusion Matrix:")
    print(f"  True Negatives (False Positives): {cm[0, 0]}")
    print(f"  False Positives (Type I Error):   {cm[0, 1]}")
    print(f"  False Negatives (Type II Error):  {cm[1, 0]}")
    print(f"  True Positives (Confirmed):       {cm[1, 1]}")
    
    # Save Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['False Positive (0)', 'Confirmed (1)'],
           yticklabels=['False Positive (0)', 'Confirmed (1)'],
           title='Classifier Confusion Matrix (Test Set)',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black",
                    fontweight='bold', fontsize=12)
    fig.tight_layout()
    cm_path = os.path.join("models", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix plot to {cm_path}")
    
    # 6. Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(history.history['loss'], label='Train Loss', color='#e74c3c', linewidth=1.5)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='#3498db', linewidth=1.5)
    ax1.set_title('Classifier Loss History', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Crossentropy Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Accuracy curves
    ax2.plot(history.history['accuracy'], label='Train Acc', color='#2ecc71', linewidth=1.5)
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color='#9b59b6', linewidth=1.5)
    ax2.set_title('Classifier Accuracy History', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    hist_path = os.path.join("models", "classifier_training_history.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {hist_path}")
    
    # 7. Save model
    saved_model_dir = os.path.join("models", "saved_models")
    os.makedirs(saved_model_dir, exist_ok=True)
    classifier_path = os.path.join(saved_model_dir, "classifier.keras")
    classifier.save(classifier_path)
    print(f"Classifier saved successfully as: {classifier_path}")

if __name__ == "__main__":
    main()
