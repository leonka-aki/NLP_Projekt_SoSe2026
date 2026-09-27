"""Modell 2/3: TF-IDF-Repräsentationen + klassisches Machine Learning."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from pawsx_project.config import LABEL_SPALTE, RANDOM_STATE, TEXT_SPALTE_1, TEXT_SPALTE_2
from pawsx_project.models.base import BasisModell
from pawsx_project.preprocessing import bereinige_text_tfidf

class _TfidfKlassifikationsModell(BasisModell):
    """Gemeinsame Basis für TF-IDF-basierte Klassifikatoren.

    Beide Sätze werden mit demselben, auf train_df gefitteten TF-IDF-Vektorisierer
    in numerische Repräsentationen umgewandelt. Merkmalsvektor pro Satzpaar:
    [TF-IDF(Satz1), TF-IDF(Satz2), Kosinusähnlichkeit(Satz1, Satz2)]. Unterklassen
    unterscheiden sich nur im verwendeten Klassifikator.
    """
    def __init__(self, klassifikator, anzeigename: str, max_features: int = 20_000):
        self._klassifikator = klassifikator
        self._anzeigename = anzeigename
        self._vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    def fit(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> "_TfidfKlassifikationsModell":
        alle_saetze = pd.concat([train_df[TEXT_SPALTE_1], train_df[TEXT_SPALTE_2]]).map(
            bereinige_text_tfidf
        )
        self._vectorizer.fit(alle_saetze)

        X = self._baue_merkmale(train_df)
        y = train_df[LABEL_SPALTE].to_numpy()
        self._klassifikator.fit(X, y)
        return self

    def _baue_merkmale(self, df: pd.DataFrame) -> sp.csr_matrix:
        satz1 = df[TEXT_SPALTE_1].map(bereinige_text_tfidf)
        satz2 = df[TEXT_SPALTE_2].map(bereinige_text_tfidf)

        vec1 = self._vectorizer.transform(satz1)
        vec2 = self._vectorizer.transform(satz2)

        # TfidfVectorizer normalisiert Vektoren standardmaessig auf L2-Norm,
        # dadurch entspricht das Skalarprodukt der Kosinusaehnlichkeit.
        kosinus = np.asarray(vec1.multiply(vec2).sum(axis=1)).reshape(-1, 1)

        return sp.hstack([vec1, vec2, sp.csr_matrix(kosinus)], format="csr")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._baue_merkmale(df)
        return self._klassifikator.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        X = self._baue_merkmale(df)
        if hasattr(self._klassifikator, "predict_proba"):
            return self._klassifikator.predict_proba(X)[:, 1]
        if hasattr(self._klassifikator, "decision_function"):
            return self._klassifikator.decision_function(X)
        return None

    @property
    def name(self) -> str:
        return self._anzeigename

class TfidfLogistischeRegression(_TfidfKlassifikationsModell):
    def __init__(self, max_features: int = 20_000):
        super().__init__(
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            "TF-IDF + Logistische Regression",
            max_features=max_features,
        )

class TfidfLinearSVM(_TfidfKlassifikationsModell):
    def __init__(self, max_features: int = 20_000):
        super().__init__(
            LinearSVC(random_state=RANDOM_STATE),
            "TF-IDF + Linear SVM",
            max_features=max_features,
        )
