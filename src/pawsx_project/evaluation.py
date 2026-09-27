from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pawsx_project.config import ERGEBNIS_PREDICTIONS_PFAD, LABEL_SPALTE
from pawsx_project.models.base import BasisModell

def evaluiere_modell(
    modell: BasisModell,
    test_df: pd.DataFrame,
    speichere_predictions: bool = True,
) -> dict:
    """Wertet ein trainiertes Modell auf test_df aus und liefert Standardmetriken.

    Bei speichere_predictions=True werden zusaetzlich die Vorhersagen unter
    results/predictions/<modellname>.csv abgelegt
    """
    y_true = test_df[LABEL_SPALTE].to_numpy()
    y_pred = modell.predict(test_df)
    y_proba = modell.predict_proba(test_df)

    metriken = _berechne_metriken(modell.name, y_true, y_pred)

    if speichere_predictions:
        _speichere_predictions(modell.name, test_df, y_pred, y_proba)

    return metriken

def evaluiere_scores(
    modell_name: str, y_true: np.ndarray, y_proba: np.ndarray, schwelle: float = 0.5
) -> dict:
    """Berechnet Standardmetriken direkt aus Wahrscheinlichkeiten statt aus einem
    Basismodell. Wird fuer Vorhersagen gebraucht, die nicht ueber ein einzelnes
    Modell.predict() laufen - z.B. die test-time-symmetrisierten Cross-Encoder-
    Scores
    """
    y_pred = (y_proba >= schwelle).astype(int)
    return _berechne_metriken(modell_name, y_true, y_pred)

def _berechne_metriken(modell_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Modell": modell_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

def _speichere_predictions(
    modell_name: str,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> None:
    ERGEBNIS_PREDICTIONS_PFAD.mkdir(parents=True, exist_ok=True)

    predictions_df = test_df.copy()
    predictions_df["prediction"] = y_pred
    predictions_df["probability"] = y_proba if y_proba is not None else np.nan
    predictions_df["correct"] = predictions_df[LABEL_SPALTE] == predictions_df["prediction"]

    dateiname = _dateisicherer_name(modell_name) + ".csv"
    predictions_df.to_csv(ERGEBNIS_PREDICTIONS_PFAD / dateiname, index=False)

def _dateisicherer_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus")
