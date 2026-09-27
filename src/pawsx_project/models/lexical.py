"""Modell 1: lexikalische Baseline auf Basis von Wortüberlappung."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pawsx_project.config import LABEL_SPALTE, TEXT_SPALTE_1, TEXT_SPALTE_2
from pawsx_project.models.base import BasisModell
from pawsx_project.preprocessing import bereinige_text_tfidf

def _tokenisiere(text: str) -> set:
    return set(bereinige_text_tfidf(text).split())

def berechne_merkmale(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet lexikalische Ähnlichkeitsmerkmale für jedes Satzpaar in df.

    jaccard_aehnlichkeit ist die Entscheidungsgrundlage von JaccardModell.
    Die beiden weiteren Merkmale werden nicht für die Klassifikation benutzt.
    """
    tokens1 = df[TEXT_SPALTE_1].map(_tokenisiere)
    tokens2 = df[TEXT_SPALTE_2].map(_tokenisiere)

    schnittmenge = [len(a & b) for a, b in zip(tokens1, tokens2)]
    vereinigung = [len(a | b) for a, b in zip(tokens1, tokens2)]
    kleinere_menge = [min(len(a), len(b)) for a, b in zip(tokens1, tokens2)]

    jaccard = [s / u if u > 0 else 0.0 for s, u in zip(schnittmenge, vereinigung)]
    anteil_gemeinsam = [s / k if k > 0 else 0.0 for s, k in zip(schnittmenge, kleinere_menge)]
    laengendifferenz = [abs(len(a) - len(b)) for a, b in zip(tokens1, tokens2)]

    return pd.DataFrame(
        {
            "jaccard_aehnlichkeit": jaccard,
            "anteil_gemeinsamer_woerter": anteil_gemeinsam,
            "laengendifferenz": laengendifferenz,
        },
        index=df.index,
    )

class JaccardModell(BasisModell):
    """Klassifiziert allein über einen Schwellenwert auf der Jaccard-Ähnlichkeit
    der (bereinigten) Wortmengen beider Sätze. Der Schwellenwert wird auf dem
    Validierungsdatensatz per F1-Score bestimmt.
    """

    def __init__(self) -> None:
        self._schwelle: float = 0.5

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "JaccardModell":
        if val_df is None:
            raise ValueError("JaccardModell benoetigt val_df zur Schwellenwertbestimmung.")

        jaccard = berechne_merkmale(val_df)["jaccard_aehnlichkeit"].to_numpy()
        y_true = val_df[LABEL_SPALTE].to_numpy()

        beste_schwelle, bester_f1 = 0.5, -1.0
        for schwelle in np.unique(jaccard):
            y_pred = (jaccard >= schwelle).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > bester_f1:
                bester_f1, beste_schwelle = score, schwelle

        self._schwelle = float(beste_schwelle)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        jaccard = berechne_merkmale(df)["jaccard_aehnlichkeit"].to_numpy()
        return (jaccard >= self._schwelle).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return berechne_merkmale(df)["jaccard_aehnlichkeit"].to_numpy()

    @property
    def name(self) -> str:
        return "Wortüberlappung (Jaccard)"
