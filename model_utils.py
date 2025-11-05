#!/usr/bin/env python3
# Model loading utilities for Self-Healing Classification project.
import os
from joblib import load as joblib_load

def load_backup_model(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "backup_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Backup model not found at {path}. Run `python train_backup_model.py` first."
        )
    return joblib_load(path)

def load_transformer_model(transformer_dir=None):
    try:
        from transformers import pipeline
        import torch
    except Exception as e:
        raise RuntimeError(
            "Transformers or torch are not installed. Install them to use transformer models."
        ) from e

    if transformer_dir is None:
        transformer_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "transformer"
        )

    if not os.path.exists(transformer_dir):
        raise FileNotFoundError(
            f"Transformer model directory not found at {transformer_dir}. Place your fine-tuned model there."
        )

    pipe = pipeline(
        "text-classification",
        model=transformer_dir,
        tokenizer=transformer_dir,
        device=0 if torch.cuda.is_available() else -1,
    )
    return pipe

def safe_load_model(prefer_transformer=False):
    if prefer_transformer:
        try:
            return load_transformer_model(), "transformer"
        except Exception as e:
            print(f"[model_utils] Transformer load failed: {e}. Falling back to backup model.")
    model = load_backup_model()
    return model, "backup"
