import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_cnn_classifier():
    """
    Builds a simple 1D CNN binary classifier for exoplanet detection.
    Includes L2 regularization (0.001) and increased dropout (0.4) to reduce overfitting.
    """
    inputs = layers.Input(shape=(2000, 1))
    
    # Conv1D(16) -> MaxPool -> Conv1D(32) -> MaxPool -> Conv1D(64) -> GlobalAveragePooling1D
    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense(32) -> Dropout(0.4) -> Dense(1, sigmoid) with L2 regularization
    x = layers.Dense(
        32, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(
        1, 
        activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    
    model = Model(inputs, outputs, name="CNN_Classifier")
    
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    # Lower learning rate (0.0005)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model

def augment_light_curve(x):
    """
    Data augmentation helper: Time shifts (roll), tiny Gaussian noise, 
    and slight amplitude rescaling.
    x shape: (2000, 1) or (2000,)
    """
    x_aug = np.copy(x)
    
    # 1. Time shift (roll) by a small random number of indices (e.g. -150 to 150)
    shift = np.random.randint(-150, 150)
    x_aug = np.roll(x_aug, shift, axis=0)
    
    # 2. Add small random Gaussian noise
    noise_std = np.random.uniform(0.001, 0.005)
    noise = np.random.normal(0, noise_std, x_aug.shape)
    x_aug += noise
    
    # 3. Rescale amplitude deviation from median slightly
    median = np.median(x_aug)
    scale = np.random.uniform(0.95, 1.05)
    x_aug = (x_aug - median) * scale + median
    
    # Clip back to [0.0, 1.0]
    x_aug = np.clip(x_aug, 0.0, 1.0)
    return x_aug

def main():
    print("--- Exoplanet AI Hackathon: CNN Classifier Training (v2) ---")
    
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
    
    # STEP 1: Sanity Check Data Alignment
    print("\n" + "="*50)
    print("STEP 1: SANITY CHECK DATA ALIGNMENT")
    print("="*50)
    mean_label_1 = np.mean(X_train[y_train == 1])
    mean_label_0 = np.mean(X_train[y_train == 0])
    min_label_1 = np.min(X_train[y_train == 1])
    min_label_0 = np.min(X_train[y_train == 0])
    
    print(f"y_train shape:                     {y_train.shape}")
    print(f"y_train label distribution (1/0):  {(y_train == 1).sum()} / {(y_train == 0).sum()}")
    print(f"y_train first 20 values:           {y_train[:20].tolist()}")
    print(f"Confirmed stars (1) - Mean Flux:   {mean_label_1:.6f} | Min Flux: {min_label_1:.6f}")
    print(f"Control stars (0)   - Mean Flux:   {mean_label_0:.6f} | Min Flux: {min_label_0:.6f}")
    print(f"Mean Flux Difference (0 - 1):      {mean_label_0 - mean_label_1:.6f}")
    print(f"Min Flux Difference (0 - 1):       {mean_label_0 - min_label_1:.6f}")
    print("="*50 + "\n")
    
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
    
    # STEP 3: Data Augmentation (5x training variety)
    print("Performing 5x data augmentation on training set...")
    X_train_augmented = []
    y_train_augmented = []
    
    for i in range(len(X_train_denoised)):
        # 1. Original denoised curve
        X_train_augmented.append(X_train_denoised[i])
        y_train_augmented.append(y_train[i])
        # 2. Augmented copy 1
        X_train_augmented.append(augment_light_curve(X_train_denoised[i]))
        y_train_augmented.append(y_train[i])
        # 3. Augmented copy 2
        X_train_augmented.append(augment_light_curve(X_train_denoised[i]))
        y_train_augmented.append(y_train[i])
        # 4. Augmented copy 3
        X_train_augmented.append(augment_light_curve(X_train_denoised[i]))
        y_train_augmented.append(y_train[i])
        # 5. Augmented copy 4
        X_train_augmented.append(augment_light_curve(X_train_denoised[i]))
        y_train_augmented.append(y_train[i])
        
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    
    # Center all dataset splits (subtract 0.5)
    print("Centering datasets around 0.0...")
    X_train_final = X_train_augmented - 0.5
    X_val_final = X_val_denoised - 0.5
    X_test_final = X_test_denoised - 0.5
    
    print(f"Data shapes after denoising, 5x augmentation, and centering:")
    print(f"  Train Set: {X_train_final.shape} | Labels: {y_train_augmented.shape}")
    print(f"  Val Set:   {X_val_final.shape} | Labels: {y_val.shape}")
    print(f"  Test Set:  {X_test_final.shape} | Labels: {y_test.shape}")
    
    # STEP 2: Build CNN Classifier
    classifier = build_cnn_classifier()
    classifier.summary()
    
    # Compute class weights (balanced)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_augmented), y=y_train_augmented)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Computed class weights: {class_weight_dict}")
    
    # Set callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1)
    ]
    
    # Train model (epochs=60, batch_size=8)
    epochs = 60
    batch_size = 8
    print(f"\nTraining CNN classifier for up to {epochs} epochs...")
    history = classifier.fit(
        X_train_final, y_train_augmented,
        validation_data=(X_val_final, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # STEP 1: Decision Threshold Tuning on Validation Set
    print("\nTuning decision threshold on validation set...")
    y_val_pred_probs = classifier.predict(X_val_final, batch_size=32, verbose=0).flatten()
    
    thresholds = np.arange(0.3, 0.71, 0.05)
    best_threshold = 0.5
    best_val_f1 = -1
    
    print("\n" + "="*60)
    print("VALIDATION SET DECISION THRESHOLD TUNING")
    print("="*60)
    print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 50)
    
    for th in thresholds:
        y_val_pred_bin = (y_val_pred_probs >= th).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_val_pred_bin, average='binary', zero_division=0)
        print(f"{th:<12.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = th
            
    print("="*60)
    print(f"Optimal Decision Threshold: {best_threshold:.2f} (Validation F1 Score: {best_val_f1:.4f})")
    print("="*60 + "\n")
    
    # STEP 4: Evaluate on test set using the NEW optimal threshold
    print("\nEvaluating model on the held-out test set with optimal threshold...")
    y_pred = classifier.predict(X_test_final, batch_size=32, verbose=0)
    y_pred_probs = y_pred.flatten()
    
    # Classify based on optimal threshold
    y_pred_bin = (y_pred_probs >= best_threshold).astype(int)
    
    # Calculate test metrics using optimal threshold
    test_acc = accuracy_score(y_test, y_pred_bin)
    test_prec = precision_score(y_test, y_pred_bin, zero_division=0)
    test_recall = recall_score(y_test, y_pred_bin, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_bin, zero_division=0)
    
    # STEP 2: Compute Test ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    
    print("\n" + "="*50)
    print("STEP 4: PREDICTED PROBABILITY DISTRIBUTION ON TEST SET")
    print("="*50)
    print(f"  Min probability:  {np.min(y_pred_probs):.6f}")
    print(f"  Max probability:  {np.max(y_pred_probs):.6f}")
    print(f"  Mean probability: {np.mean(y_pred_probs):.6f}")
    print(f"  Std deviation:    {np.std(y_pred_probs):.6f}")
    deciles = np.percentile(y_pred_probs, np.arange(10, 100, 10))
    print(f"  Deciles (10th to 90th percentile): {[round(d, 4) for d in deciles]}")
    print("="*50)
    
    print("\n" + "="*50)
    print(f"TEST EVALUATION PERFORMANCE (Threshold = {best_threshold:.2f})")
    print("="*50)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test ROC-AUC:   {roc_auc:.4f}")
    print("="*50)
    
    # Confusion Matrix
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
           title=f'CNN Classifier Confusion Matrix (Threshold = {best_threshold:.2f})',
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
    
    # STEP 2: Save ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_path = os.path.join("models", "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve plot to {roc_path}")
    
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
