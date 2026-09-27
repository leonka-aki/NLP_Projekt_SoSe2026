from __future__ import annotations
import time
import pandas as pd
from pawsx_project.config import KLASSEN, TEXT_SPALTE_1, TEXT_SPALTE_2
from pawsx_project.evaluation import evaluiere_modell
from pawsx_project.models.base import BasisModell

def fuehre_experiment_durch(
    modell: BasisModell,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Trainiert modell und evaluiert es auf test_df. Identisch fuer jedes Modell
    """
    start = time.perf_counter()
    modell.fit(train_df, val_df)
    gemessene_zeit = time.perf_counter() - start

    ergebnis = evaluiere_modell(modell, test_df)
    trainingsdauer = getattr(modell, "trainingsdauer_sekunden", None)
    ergebnis["Trainingszeit (s)"] = trainingsdauer if trainingsdauer is not None else gemessene_zeit
    return ergebnis

def werte_satzpaar_aus(
    modelle: dict[str, BasisModell], satz1: str, satz2: str
) -> pd.DataFrame:
    """Wendet mehrere bereits trainierte Modelle auf ein einzelnes, frei waehlbares
    Satzpaar an - damit lassen sich alle Modelle interaktiv mit eigenen Beispielen
    testen ohne den Testdatensatz neu zu laden
    """
    paar_df = pd.DataFrame({TEXT_SPALTE_1: [satz1], TEXT_SPALTE_2: [satz2]})

    zeilen = []
    for name, modell in modelle.items():
        vorhersage = int(modell.predict(paar_df)[0])
        proba = modell.predict_proba(paar_df)
        zeilen.append(
            {
                "Modell": name,
                "Vorhersage": KLASSEN[vorhersage],
                "Score": float(proba[0]) if proba is not None else float("nan"),
            }
        )
    return pd.DataFrame(zeilen)
