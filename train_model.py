
"""
Optional training/evaluation script.

The Streamlit app already trains and evaluates the models at startup.
This script is provided so the project has a conventional ML-development
entry point for experiments and future model versioning.
"""

from pathlib import Path
import pandas as pd
from app import load_data, train_models

if __name__ == "__main__":
    df = load_data()
    bundle = train_models(df)
    print("\nGlucoGuard AI model comparison")
    print("-" * 45)
    for name, score in bundle["cv_scores"].items():
        print(f"{name:24s} CV ROC-AUC: {score:.4f}")
    print(f"\nSelected model: {bundle['best_name']}")
    print(f"Holdout ROC-AUC: {bundle['metrics']['roc_auc']:.4f}")
