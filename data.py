"""
Utilidades para cargar y preparar datos
"""
from typing import Optional
from datasets import load_dataset


def load_gsm8k(val_rows: Optional[int] = None):
    """
    Carga el dataset GSM8K y lo divide en train/val/test.
    
    Args:
        val_rows: Número de filas para validación (None = min(200, len(train)))
    
    Returns:
        Tuple de (ds_train, ds_val, ds_test)
    """
    print("Loading GSM8K…")
    ds_train_full = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")

    if val_rows is None:
        val_rows = min(200, len(ds_train_full))
    else:
        val_rows = min(val_rows, len(ds_train_full))

    ds_val = ds_train_full.select(range(val_rows))
    ds_train = ds_train_full.select(range(val_rows, len(ds_train_full)))

    print(f"Splits: {len(ds_train)} train | {len(ds_val)} val | {len(ds_test)} test")
    return ds_train, ds_val, ds_test

