"""Conservative language identification for radiology reports (stdlib only).

Script-based for Cyrillic/Greek, stop-word based for Latin-script languages. It deliberately returns
"unknown" when the evidence is weak or two languages are nearly tied, because a wrong guess would let
an English/Spanish rule set mislabel a report it cannot actually read.
"""

from __future__ import annotations

import re
import unicodedata

SUPPORTED_LANGUAGES = frozenset({"en", "es"})
MIN_STOPWORD_HITS = 3
MIN_MARGIN = 1.5  # winner must have at least this many times the runner-up's hits

_STOPWORDS = {
    "en": "the and there is are of with no without in at to for from was were this that which noted seen "
          "tear meniscus menisci ligament effusion findings impression intact cartilage knee unremarkable fracture",
    "es": "el la los las de del y con sin hay en se es son una un por para que observa observan "
          "rotura menisco meniscos rodilla ligamento ligamentos hallazgos derrame cartilago artrosis impresion conclusion tecnica",
    "tr": "ve bir ile yok izlenmektedir izlenmistir olarak dir normal gorunumde uzerinde yoktur",
    "de": "der die das und kein keine ohne mit ist sind eine einer im zeigt",
    "nl": "het een geen van met zijn de en is te op",
    "fr": "le les des pas avec une est et dans sans du au",
    "hr": "nema je se na bez uredan uredna vidljiv vidljive i u",
    "pt": "sem com uma nao os do da dos das em",
    "it": "il della senza con non una dei delle nel",
}
_WORD_SETS = {lang: frozenset(words.split()) for lang, words in _STOPWORDS.items()}
_TOKEN = re.compile(r"[a-z]+")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def detect_language(text: str) -> str:
    """Return an ISO-like code ("en", "es", "tr", ...), "cyrillic", "greek", or "unknown"."""
    if not text or not text.strip():
        return "unknown"
    cyrillic = sum("Ѐ" <= ch <= "ӿ" for ch in text)
    greek = sum("Ͱ" <= ch <= "Ͽ" for ch in text)
    letters = sum(ch.isalpha() for ch in text) or 1
    if cyrillic / letters > 0.2:
        return "cyrillic"
    if greek / letters > 0.2:
        return "greek"
    tokens = _TOKEN.findall(_fold(text))
    scores = {lang: sum(token in words for token in tokens) for lang, words in _WORD_SETS.items()}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, best_hits), (_, runner_hits) = ranked[0], ranked[1]
    if best_hits < MIN_STOPWORD_HITS or (runner_hits and best_hits < MIN_MARGIN * runner_hits):
        return "unknown"
    return best


def is_supported(language: str) -> bool:
    return language in SUPPORTED_LANGUAGES
