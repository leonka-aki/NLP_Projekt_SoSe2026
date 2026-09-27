"""Modell 4: Sentence Embeddings + einfacher Klassifikator."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pawsx_project.config import (
    LABEL_SPALTE,
    RANDOM_STATE,
    SENTENCE_TRANSFORMER_MODELL,
    TEXT_SPALTE_1,
    TEXT_SPALTE_2,
)
from pawsx_project.models.base import BasisModell


class SentenceTransformerModell(BasisModell):
    """Nutzt ein vortrainiertes Sentence-Transformer-Modell, um beide Sätze in
    semantische Embeddings umzuwandeln. Merkmalsvektor pro Satzpaar:
    [Kosinusähnlichkeit, |Embedding1 - Embedding2|, Embedding1 * Embedding2],
    anschließend klassifiziert durch Logistische Regression.
    """
    def __init__(self, modellname: str = SENTENCE_TRANSFORMER_MODELL, encoder=None):
        if encoder is not None:
            self._encoder = encoder
        else:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(modellname)

        self._klassifikator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    def _encode(self, texte) -> np.ndarray:
        return self._encoder.encode(
            list(texte), batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )

    def _baue_merkmale(self, df: pd.DataFrame) -> np.ndarray:
        alle_texte = list(df[TEXT_SPALTE_1]) + list(df[TEXT_SPALTE_2])
        embeddings = np.asarray(self._encode(alle_texte))

        n = len(df)
        emb1, emb2 = embeddings[:n], embeddings[n:]

        # Embeddings sind L2-normalisiert -> Skalarprodukt = Kosinusaehnlichkeit
        kosinus = np.sum(emb1 * emb2, axis=1, keepdims=True)
        differenz = np.abs(emb1 - emb2)
        produkt = emb1 * emb2

        return np.hstack([kosinus, differenz, produkt])

    def fit(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> "SentenceTransformerModell":
        X = self._baue_merkmale(train_df)
        y = train_df[LABEL_SPALTE].to_numpy()
        self._klassifikator.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._baue_merkmale(df)
        return self._klassifikator.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        X = self._baue_merkmale(df)
        return self._klassifikator.predict_proba(X)[:, 1]

    @property
    def name(self) -> str:
        return "Sentence Transformer + Logistische Regression"
