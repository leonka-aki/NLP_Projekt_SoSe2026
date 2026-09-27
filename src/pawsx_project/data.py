from __future__ import annotations
from typing import Tuple
import pandas as pd
from pawsx_project.config import (
    DATEN_VERARBEITET_PFAD,
    HF_DATENSATZ_NAME,
    HF_DATENSATZ_SPRACHE,
    ID_SPALTE,
    LABEL_SPALTE,
    TEXT_SPALTE_1,
    TEXT_SPALTE_2,
)

_SPLITS = ("train", "validation", "test")

def _cache_datei(split: str):
    return DATEN_VERARBEITET_PFAD / f"{split}.parquet"

def _hf_split_zu_dataframe(hf_split, split: str) -> pd.DataFrame:
    df = hf_split.to_pandas()
    df = df.rename(
        columns={
            "sentence1": TEXT_SPALTE_1,
            "sentence2": TEXT_SPALTE_2,
            "label": LABEL_SPALTE,
        }
    )
    df[ID_SPALTE] = [f"{split}_{i}" for i in range(len(df))]
    df = df[[ID_SPALTE, TEXT_SPALTE_1, TEXT_SPALTE_2, LABEL_SPALTE]]
    return df.reset_index(drop=True)

def _lade_von_huggingface() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    rohdaten = load_dataset(HF_DATENSATZ_NAME, HF_DATENSATZ_SPRACHE)
    return tuple(_hf_split_zu_dataframe(rohdaten[split], split) for split in _SPLITS)

def lade_datensatz(
    neu_laden: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Laedt train/validation/test des deutschen PAWS-X-Datensatzes.

    Ergebnisse werden nach dem ersten Laden lokal als Parquet zwischengespeichert
    (data/processed/), damit das Notebook nicht bei jedem Lauf neu von
    Hugging Face herunterladen muss. Mit neu_laden=True wird der Cache ignoriert.
    """
    cache_vorhanden = all(_cache_datei(split).exists() for split in _SPLITS)

    if cache_vorhanden and not neu_laden:
        return tuple(pd.read_parquet(_cache_datei(split)) for split in _SPLITS)

    train_df, val_df, test_df = _lade_von_huggingface()

    DATEN_VERARBEITET_PFAD.mkdir(parents=True, exist_ok=True)
    for split, df in zip(_SPLITS, (train_df, val_df, test_df)):
        df.to_parquet(_cache_datei(split), index=False)

    return train_df, val_df, test_df