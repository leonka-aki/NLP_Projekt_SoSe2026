from __future__ import annotations
import re

_WHITESPACE_PATTERN = re.compile(r"\s+")
_NICHT_WORT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)

def bereinige_text_basis(text: str) -> str:
    """Minimale Bereinigung: Whitespace normalisieren, Raender trimmen."""
    text = text.strip()
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text

def bereinige_text_tfidf(text: str) -> str:
    """Zusaetzliche Bereinigung fuer lexikalische/TF-IDF-Modelle."""
    text = bereinige_text_basis(text)
    text = text.lower()
    text = _NICHT_WORT_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
