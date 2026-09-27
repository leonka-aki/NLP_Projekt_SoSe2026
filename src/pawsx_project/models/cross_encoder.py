"""Modell 5: Transformer Cross-Encoder (gemeinsames Feintuning auf Satzpaaren)."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pawsx_project.config import (
    CROSS_ENCODER_BASISMODELL,
    ERGEBNIS_MODELLE_PFAD,
    LABEL_SPALTE,
    RANDOM_STATE,
    TEXT_SPALTE_1,
    TEXT_SPALTE_2,
)
from pawsx_project.models.base import BasisModell

_TRAININGSDAUER_DATEI = "trainingsdauer_sekunden.txt"

class CrossEncoderModell(BasisModell):
    """Feintuning eines vortrainierten deutschen BERT-Modells auf Satzpaaren
    (Eingabe konzeptionell "Satz1 [SEP] Satz2"), direkt für binäre
    Paraphrasenklassifikation.

    Im Gegensatz zu Modell 4 (Sentence Embeddings, beide Sätze getrennt gepoolt)
    sieht der Cross-Encoder beide Sätze gemeinsam in einem Forward-Pass und kann
    dadurch die Wortreihenfolge und Interaktionen zwischen den Sätzen modellieren.
    """

    def __init__(
        self,
        modellname: str = CROSS_ENCODER_BASISMODELL,
        epochen: float = 2.0,
        batch_size: int = 16,
        lernrate: float = 2e-5,
        max_length: int = 128,
        schwelle: float = 0.5,
        anzeigename: Optional[str] = None,
        checkpoint_pfad: Optional[Union[str, Path]] = None,
    ):
        self._modellname = modellname
        self._epochen = epochen
        self._batch_size = batch_size
        self._lernrate = lernrate
        self._max_length = max_length
        self._schwelle = schwelle
        self._anzeigename = anzeigename
        self._checkpoint_pfad = Path(checkpoint_pfad) if checkpoint_pfad is not None else None
        self._modell = None  # type: Optional["CrossEncoder"]
        self._trainingsdauer: Optional[float] = None

    def _baue_dataset(self, df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "satz1": df[TEXT_SPALTE_1].tolist(),
                "satz2": df[TEXT_SPALTE_2].tolist(),
                "label": df[LABEL_SPALTE].astype(int).tolist(),
            }
        )

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "CrossEncoderModell":
        from sentence_transformers.cross_encoder import CrossEncoder

        if self._checkpoint_pfad is not None and (self._checkpoint_pfad / "config.json").exists():
            self._modell = CrossEncoder(
                str(self._checkpoint_pfad), num_labels=1, max_length=self._max_length
            )
            self._trainingsdauer = self._lade_trainingsdauer()
            return self

        from sentence_transformers.cross_encoder import (
            CrossEncoderTrainer,
            CrossEncoderTrainingArguments,
        )
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

        self._modell = CrossEncoder(self._modellname, num_labels=1, max_length=self._max_length)

        train_dataset = self._baue_dataset(train_df)
        eval_dataset = self._baue_dataset(val_df) if val_df is not None else None

        args = CrossEncoderTrainingArguments(
            output_dir=str(ERGEBNIS_MODELLE_PFAD / "cross_encoder_checkpoints"),
            num_train_epochs=self._epochen,
            per_device_train_batch_size=self._batch_size,
            per_device_eval_batch_size=self._batch_size,
            learning_rate=self._lernrate,
            warmup_steps=100,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="no",
            logging_steps=200,
            seed=RANDOM_STATE,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = CrossEncoderTrainer(
            model=self._modell,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=BinaryCrossEntropyLoss(self._modell),
        )
        start = time.perf_counter()
        trainer.train()
        self._trainingsdauer = time.perf_counter() - start

        if self._checkpoint_pfad is not None:
            self._checkpoint_pfad.mkdir(parents=True, exist_ok=True)
            self._modell.save_pretrained(str(self._checkpoint_pfad), create_model_card=False)
            self._speichere_trainingsdauer()

        return self

    def _lade_trainingsdauer(self) -> Optional[float]:
        if self._checkpoint_pfad is None:
            return None
        datei = self._checkpoint_pfad / _TRAININGSDAUER_DATEI
        return float(datei.read_text()) if datei.exists() else None

    def _speichere_trainingsdauer(self) -> None:
        if self._checkpoint_pfad is not None and self._trainingsdauer is not None:
            (self._checkpoint_pfad / _TRAININGSDAUER_DATEI).write_text(f"{self._trainingsdauer:.1f}")

    def _scores(self, df: pd.DataFrame) -> np.ndarray:
        paare = list(zip(df[TEXT_SPALTE_1], df[TEXT_SPALTE_2]))
        return np.asarray(
            self._modell.predict(paare, batch_size=self._batch_size, show_progress_bar=False)
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self._scores(df) >= self._schwelle).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        return self._scores(df)

    @property
    def name(self) -> str:
        return self._anzeigename or "Transformer Cross-Encoder"

    @property
    def trainingsdauer_sekunden(self) -> Optional[float]:
        """Dauer des eigentlichen Feintunings in Sekunden.
        """
        return self._trainingsdauer


def predict_proba_beide_richtungen(
    modell: CrossEncoderModell, df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Scores in normaler und in vertauschter Satzreihenfolge.
    """
    vertauscht = df.rename(columns={TEXT_SPALTE_1: TEXT_SPALTE_2, TEXT_SPALTE_2: TEXT_SPALTE_1})
    return modell.predict_proba(df), modell.predict_proba(vertauscht)
