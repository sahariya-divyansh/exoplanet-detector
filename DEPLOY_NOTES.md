# Hugging Face Spaces Deployment Notes

## Manual Model Upload Required

The trained Keras model files are binary artifacts and are excluded from Git by `.gitignore`, so they were not pushed to GitHub.

Before the Hugging Face Space can run successfully, upload these local files into the Space with the same relative paths:

```text
models/saved_models/denoiser.keras
models/saved_models/classifier.keras
```

Exact local paths on this machine:

```text
C:\Users\user\OneDrive\Desktop\AI exoplanet detection\models\saved_models\denoiser.keras
C:\Users\user\OneDrive\Desktop\AI exoplanet detection\models\saved_models\classifier.keras
```

Expected layout inside the Hugging Face Space:

```text
.
|-- app.py
|-- requirements.txt
|-- data/
|   |-- preprocess_data.py
|-- models/
|   |-- saved_models/
|       |-- denoiser.keras
|       |-- classifier.keras
```

If these files are missing, `app.py` will stop at startup with a "Models not found" error.
