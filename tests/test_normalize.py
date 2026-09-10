"""Tests de la normalisation de libellé / référence (§3.1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalize import normalize_label, normalize_reference  # noqa: E402


def test_reference_variants_example_from_spec() -> None:
    result = normalize_label("FA0012345")
    assert result.label_numbers == frozenset({"FA0012345", "0012345", "12345"})


def test_pure_numeric_reference_has_no_duplicate_variants() -> None:
    result = normalize_label("12345")
    assert result.label_numbers == frozenset({"12345"})


def test_accents_and_case_are_normalized() -> None:
    result = normalize_label("Règlement Société Générale")
    assert "É" not in result.normalized
    assert result.normalized == "REGLEMENT SOCIETE GENERALE"


def test_non_alnum_replaced_by_space_and_compressed() -> None:
    result = normalize_label("VIR//FA-0012345,,,DUPONT")
    assert "  " not in result.normalized
    assert result.normalized == "VIR FA 0012345 DUPONT"


def test_empty_and_none_label() -> None:
    assert normalize_label("").label_tokens == ()
    assert normalize_label("").label_numbers == frozenset()
    assert normalize_label(None).normalized == ""


def test_label_tokens_only_contains_alpha_tokens() -> None:
    result = normalize_label("VIR FA0012345 DUPONT 42")
    assert result.label_tokens == ("VIR", "DUPONT")
    assert "FA0012345" not in result.label_tokens
    assert "42" in result.label_numbers


def test_normalize_reference_matches_label_numbers_variant() -> None:
    ref_variants = normalize_reference("FA0012345")
    label = normalize_label("VIR DUPONT 12345")
    assert ref_variants & label.label_numbers


def test_normalize_reference_no_match_for_unrelated_numbers() -> None:
    ref_variants = normalize_reference("FA0012345")
    label = normalize_label("VIR DUPONT 99999")
    assert not (ref_variants & label.label_numbers)
