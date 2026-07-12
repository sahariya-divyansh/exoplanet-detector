import sys
import lightkurve as lk
import tensorflow as tf
import sklearn
import matplotlib
import pandas as pd
import numpy as np
import gradio as gr

print("--- AI Exoplanet Detection Project Environment Setup Verification ---")
print(f"Python Version:       {sys.version.split()[0]}")
print(f"numpy version:        {np.__version__}")
print(f"pandas version:       {pd.__version__}")
print(f"matplotlib version:   {matplotlib.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"tensorflow version:   {tf.__version__}")
print(f"lightkurve version:   {lk.__version__}")
print(f"gradio version:       {gr.__version__}")
print("---------------------------------------------------------------------")
print("All packages successfully installed and imported!")
