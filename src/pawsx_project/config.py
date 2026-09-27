from pathlib import Path

PROJEKT_PFAD = Path(__file__).resolve().parents[2]

DATEN_PFAD = PROJEKT_PFAD / "data"
DATEN_VERARBEITET_PFAD = DATEN_PFAD / "processed"

ERGEBNIS_PFAD = PROJEKT_PFAD / "results"
ERGEBNIS_METRIKEN_PFAD = ERGEBNIS_PFAD / "metrics"
ERGEBNIS_PREDICTIONS_PFAD = ERGEBNIS_PFAD / "predictions"
ERGEBNIS_FIGUREN_PFAD = ERGEBNIS_PFAD / "figures"
ERGEBNIS_MODELLE_PFAD = ERGEBNIS_PFAD / "models"

RANDOM_STATE = 42

# Datensatz: deutscher Split von PAWS-X (https://huggingface.co/datasets/google-research-datasets/paws-x)
HF_DATENSATZ_NAME = "google-research-datasets/paws-x"
HF_DATENSATZ_SPRACHE = "de"

ID_SPALTE = "id"
TEXT_SPALTE_1 = "sentence1"
TEXT_SPALTE_2 = "sentence2"
LABEL_SPALTE = "label"

KLASSEN = {
    0: "Keine Paraphrase",
    1: "Paraphrase",
}

SENTENCE_TRANSFORMER_MODELL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CROSS_ENCODER_BASISMODELL = "google-bert/bert-base-german-cased"
CROSS_ENCODER_BASISMODELL_STARK = "xlm-roberta-base"
