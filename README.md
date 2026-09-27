# Lexikalische Ähnlichkeit vs. semantisches Verständnis

NLP-Projekt: Vergleich klassischer NLP-Verfahren und
Transformer-Modelle bei der Erkennung deutscher Paraphrasen auf dem
[PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x)-Datensatz (Split `de`).

## Forschungsfrage

In welchem Maß verbessern kontextbasierte Sprachrepräsentationen die Erkennung
deutscher Paraphrasen gegenüber klassischen, auf lexikalischer Ähnlichkeit
basierenden NLP-Verfahren?

## Untersuchte Modelle

1. Wortüberlappung (Jaccard-Ähnlichkeit) — lexikalische Baseline
2. TF-IDF + Logistische Regression
3. TF-IDF + Linear SVM
4. Sentence Embeddings (Sentence-Transformer + Klassifikator)
5. Transformer Cross-Encoder (gemeinsames Feintuning auf Satzpaaren, `bert-base-german-cased`)
6. Transformer Cross-Encoder mit stärkerem, mehrsprachigem Backbone (`xlm-roberta-base`)

## Zusammenfassung der Ergebnisse

| Modell | Accuracy | F1 | Trainingszeit |
| --- | ---: | ---: | ---: |
| Wortüberlappung (Jaccard) | 46.0% | 0.623 | <1s |
| TF-IDF + Linear SVM | 56.1% | 0.425 | ~8s |
| TF-IDF + Logistische Regression | 57.9% | 0.371 | ~7s |
| Sentence Transformer + Logistische Regression | 58.3% | 0.352 | ~3min |
| Transformer Cross-Encoder (BERT) | 85.4% | 0.840 | ~15min |
| Cross-Encoder (BERT, symmetrisiert) | 86.0% | 0.846 | - |
| Transformer Cross-Encoder (XLM-RoBERTa) | 86.6% | 0.853 | ~3h |
| Cross-Encoder (XLM-R, symmetrisiert) | 87.1% | 0.858 | - |

Das stärkere, mehrsprachige XLM-RoBERTa-Backbone verbessert das Ergebnis nochmals, benötigt dafür
aber ca. 3 Stunden statt ~15 Minuten Trainingszeit (2.55× mehr Parameter, vermutlich
VRAM-limitiert auf der verwendeten 6GB-Laptop-GPU) und löst laut Fehleranalyse im Notebook
überwiegend dieselben Fälle wie das BERT-Modell — der Mehraufwand steht in keinem proportionalen
Verhältnis zum Genauigkeitsgewinn. Die Mittelung der Vorhersage über beide
Satzreihenfolgen  verbessert **beide** Cross-Encoder messbar und
praktisch kostenlos (keine zusätzliche Trainingszeit). Cross-GPU-Nichtdeterminismus bedeutet,
dass wiederholte Trainingsläufe leicht abweichende Werte liefern können.

## Hinweis zur Datenqualität

PAWS-X-DE ist maschinell aus dem englischen PAWS-Datensatz übersetzt. Einige
scheinbare Modellfehler in der Fehleranalyse können daher auf
Übersetzungsartefakte statt auf tatsächliche Modellschwächen zurückgehen.

## Projektstruktur

```text
src/pawsx_project/
    config.py              Zentrale Pfade, Konstanten, Modellnamen
    data.py                Laden/Cachen von PAWS-X-DE
    preprocessing.py       Textbereinigung (getrennt für TF-IDF vs. Transformer)
    evaluation.py          Einheitliche Metriken + Predictions-Export
    pipeline.py            fuehre_experiment_durch(): fit + evaluate + Trainingszeit in einem
                           Schritt; werte_satzpaar_aus(): alle Modelle auf ein einzelnes,
                           frei waehlbares Satzpaar anwenden
    models/
        base.py                  Gemeinsame Modell-Schnittstelle (BasisModell)
        lexical.py               Modell 1: Jaccard-Ähnlichkeit (Wortüberlappung)
        tfidf.py                 Modell 2/3: TF-IDF + Logistische Regression / Linear SVM
        sentence_transformer.py  Modell 4: Sentence Embeddings + Logistische Regression
        cross_encoder.py         Modell 5/6: Cross-Encoder-Feintuning (BERT bzw. XLM-RoBERTa)

notebooks/                Theorie, Stand der Technik, Implementierung,
                          Fehleranalyse, eigene Erweiterung, Fazit, Ausblick, Quellen
results/
    metrics/              Modellvergleichstabelle
    predictions/          Predictions je Modell (für Fehleranalyse)
    figures/              Diagramme
```



Leon Kaufhold
Matrikelnummer: 30525414