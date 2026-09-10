"""Normalisation du libellé bancaire et des références (§3.1).

Pipeline déterministe et versionné :
1. Majuscules, suppression des accents (NFKD).
2. Tout caractère non alphanumérique → espace.
3. Compression des espaces multiples.
4. Extraction de deux vues : `label_tokens` (tokens alphabétiques) et
   `label_numbers` (tokens numériques + variantes sans zéros de tête et
   sans préfixe alphabétique).
"""

from __future__ import annotations

import dataclasses
import re
import string
import unicodedata
from typing import Callable, Optional

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _alnum_upper_ascii(s: Optional[str]) -> str:
    if not s:
        return ""
    upper = unicodedata.normalize("NFKD", s.upper())
    ascii_only = "".join(ch for ch in upper if not unicodedata.combining(ch))
    cleaned = _NON_ALNUM_RE.sub(" ", ascii_only)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def _number_variants(token: str) -> set[str]:
    """`FA0012345` → `{FA0012345, 0012345, 12345}` (cf. §3.1)."""
    variants = {token}
    stripped_prefix = token.lstrip(string.ascii_uppercase)
    if stripped_prefix and stripped_prefix != token:
        variants.add(stripped_prefix)
    base = stripped_prefix if stripped_prefix else token
    variants.add(base.lstrip("0") or "0")
    return variants


@dataclasses.dataclass(frozen=True)
class NormalizedLabel:
    raw: str
    normalized: str
    label_tokens: tuple[str, ...]
    label_numbers: frozenset[str]


def normalize_label(label: Optional[str]) -> NormalizedLabel:
    """Normalise un libellé bancaire brut en vues tokens / numbers."""
    cleaned = _alnum_upper_ascii(label)
    tokens = cleaned.split(" ") if cleaned else []
    label_tokens = tuple(t for t in tokens if t.isalpha())
    label_numbers: set[str] = set()
    for t in tokens:
        if any(c.isdigit() for c in t):
            label_numbers |= _number_variants(t)
    return NormalizedLabel(
        raw=label or "",
        normalized=cleaned,
        label_tokens=label_tokens,
        label_numbers=frozenset(label_numbers),
    )


def normalize_reference(reference: Optional[str]) -> frozenset[str]:
    """Normalise une référence facture en le même espace de variantes que
    `label_numbers`, pour un test d'appartenance direct (K2, §4)."""
    cleaned = _alnum_upper_ascii(reference)
    variants: set[str] = set()
    for t in (cleaned.split(" ") if cleaned else []):
        if any(c.isdigit() for c in t):
            variants |= _number_variants(t)
    return frozenset(variants)


def normalize_reference_primary(reference: Optional[str]) -> str:
    """Version brute (sans retrait de préfixe/zéros) de la référence, une
    fois nettoyée — le token le moins ambigu, utilisé pour distinguer un
    match fort d'un match faible (ex. baseline de la Phase 3)."""
    cleaned = _alnum_upper_ascii(reference)
    tokens = cleaned.split(" ") if cleaned else []
    return tokens[0] if tokens else ""


def longest_common_suffix_len(a: str, b: str) -> int:
    """Longueur du plus long suffixe commun à `a` et `b`."""
    n = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        n += 1
    return n


def best_ngram_similarity(
    name_tokens: tuple[str, ...],
    label_tokens: tuple[str, ...],
    sim_fn: Callable[[str, str], float],
) -> float:
    """Similarité maximale entre `name_tokens` (en bloc) et un n-gramme de
    `label_tokens` de même longueur (fenêtre glissante). `sim_fn` doit
    retourner un score dans un intervalle croissant avec la similarité."""
    if not name_tokens or not label_tokens:
        return 0.0
    name_str = " ".join(name_tokens)
    k = len(name_tokens)
    if len(label_tokens) <= k:
        return sim_fn(name_str, " ".join(label_tokens))
    best = 0.0
    for i in range(len(label_tokens) - k + 1):
        window = " ".join(label_tokens[i : i + k])
        ratio = sim_fn(name_str, window)
        if ratio > best:
            best = ratio
    return best
