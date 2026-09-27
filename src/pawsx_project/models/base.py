from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd


class BasisModell(ABC):
    """Gemeinsame Schnittstelle aller Paraphrasenerkennungs-Modelle.

    Jedes Modell (Wortueberlappung, TF-IDF, Sentence Embeddings,
    Cross-Encoder) implementiert diese Schnittstelle, damit Evaluation und
    Fehleranalyse modellunabhaengig geschrieben werden koennen.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "BasisModell":
        """Trainiert das Modell. val_df dient z.B. der Schwellenwert-/Hyperparameterwahl."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Gibt binaere Vorhersagen (0/1) fuer jede Zeile von df zurueck."""

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Gibt Wahrscheinlichkeiten fuer Klasse 1 zurueck, falls verfuegbar."""
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Anzeigename des Modells, u.a. fuer Ergebnistabellen und Dateinamen."""
