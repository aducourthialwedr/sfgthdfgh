"""Tests de la baseline par règles (Phase 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import baseline_match  # noqa: E402
from src.blocking import build_static_lookups  # noqa: E402
from src.events import Event, build_journal  # noqa: E402
from src.generator import GeneratorParams, generate  # noqa: E402
from src.state import LedgerState  # noqa: E402


def _invoice_event(invoice_id, debtor_id, agreement_id, client_reference, creation_date, due_date, initial_amount):
    return dict(
        invoice_id=invoice_id,
        client_reference=client_reference,
        creation_date=pd.Timestamp(creation_date),
        due_date=pd.Timestamp(due_date),
        initial_amount=initial_amount,
        currency="EUR",
        debtor_id=debtor_id,
        agreement_id=agreement_id,
    )


@pytest.fixture()
def two_invoice_state() -> LedgerState:
    """Deux factures du même débiteur : une servira au test de référence,
    l'autre au test de montant (montants distincts pour éviter toute
    ambiguïté croisée)."""
    state = LedgerState()
    events = [
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("DBT001",)), dict(party_id="DBT001")),
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("ASG001",)), dict(party_id="ASG001")),
        Event(
            pd.Timestamp("2024-01-01"),
            "AGREEMENT_CREATED",
            (1, ("AGR001",)),
            dict(agreement_id="AGR001", debtor_id="DBT001", client_id="ASG001", market="SERVICES", product="CLASSIQUE", recourse=True),
        ),
        Event(
            pd.Timestamp("2024-02-01"),
            "INVOICE_CREATED",
            (2, ("INV001",)),
            _invoice_event("INV001", "DBT001", "AGR001", "FA24000001", "2024-02-01", "2024-03-01", 100000),
        ),
        Event(
            pd.Timestamp("2024-02-05"),
            "INVOICE_CREATED",
            (2, ("INV002",)),
            _invoice_event("INV002", "DBT001", "AGR001", "FA24000002", "2024-02-05", "2024-03-05", 250000),
        ),
    ]
    for e in events:
        state.apply(e)
    return state


def _lookups(debtor_iban: str = "IBAN_D1") -> dict:
    return dict(debtor_by_iban={debtor_iban: "DBT001"}, assignor_by_iban={"IBAN_A1": "ASG001"})


def test_rule_a_matches_exact_reference(two_invoice_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT1", value_date=pd.Timestamp("2024-03-02"), amount=999,
        currency="EUR", iban_debtor="IBAN_UNKNOWN", label="VIR FA24000001", bankroll_code="STANDARD",
    )
    result = baseline_match(payment, two_invoice_state, pd.Timestamp("2024-03-02"), **_lookups())
    assert result == "INV001"


@pytest.fixture()
def ambiguous_reference_state() -> LedgerState:
    """Deux factures dont les références complètes diffèrent mais dont la
    variante faible (sans préfixe/zéros) coïncide sur "12345"."""
    state = LedgerState()
    events = [
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("DBT001",)), dict(party_id="DBT001")),
        Event(
            pd.Timestamp("2024-01-01"),
            "AGREEMENT_CREATED",
            (1, ("AGR001",)),
            dict(agreement_id="AGR001", debtor_id="DBT001", client_id="ASG001", market="SERVICES", product="CLASSIQUE", recourse=True),
        ),
        Event(
            pd.Timestamp("2024-02-01"),
            "INVOICE_CREATED",
            (2, ("INV001",)),
            _invoice_event("INV001", "DBT001", "AGR001", "FA0012345", "2024-02-01", "2024-03-01", 100000),
        ),
        Event(
            pd.Timestamp("2024-02-05"),
            "INVOICE_CREATED",
            (2, ("INV010",)),
            _invoice_event("INV010", "DBT001", "AGR001", "AB012345", "2024-02-05", "2024-03-05", 250000),
        ),
    ]
    for e in events:
        state.apply(e)
    return state


def test_rule_a_strong_match_wins_despite_shared_weak_variant(ambiguous_reference_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT2", value_date=pd.Timestamp("2024-03-02"), amount=999,
        currency="EUR", iban_debtor="IBAN_UNKNOWN", label="VIR FA0012345", bankroll_code="STANDARD",
    )
    result = baseline_match(payment, ambiguous_reference_state, pd.Timestamp("2024-03-02"), **_lookups())
    assert result == "INV001"


def test_rule_a_abstains_on_ambiguous_weak_reference(ambiguous_reference_state: LedgerState) -> None:
    # Le libellé ne cite que le numéro court partagé par les deux factures,
    # sans préfixe distinctif : aucune ne l'emporte, la baseline s'abstient.
    payment = dict(
        payment_id="PMT2b", value_date=pd.Timestamp("2024-03-02"), amount=999,
        currency="EUR", iban_debtor="IBAN_UNKNOWN", label="VIR 12345", bankroll_code="STANDARD",
    )
    result = baseline_match(payment, ambiguous_reference_state, pd.Timestamp("2024-03-02"), **_lookups())
    assert result is None


def test_rule_b_matches_exact_amount_on_identified_debtor(two_invoice_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT3", value_date=pd.Timestamp("2024-03-02"), amount=250000,
        currency="EUR", iban_debtor="IBAN_D1", label="VIR SANS REFERENCE", bankroll_code="STANDARD",
    )
    result = baseline_match(payment, two_invoice_state, pd.Timestamp("2024-03-02"), **_lookups())
    assert result == "INV002"


def test_rule_b_abstains_without_debtor_identification(two_invoice_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT4", value_date=pd.Timestamp("2024-03-02"), amount=250000,
        currency="EUR", iban_debtor="IBAN_UNKNOWN", label="VIR SANS REFERENCE", bankroll_code="STANDARD",
    )
    result = baseline_match(payment, two_invoice_state, pd.Timestamp("2024-03-02"), **_lookups())
    assert result is None


def test_rule_a_takes_priority_over_rule_b(two_invoice_state: LedgerState) -> None:
    # Réf pointe vers INV001 (100000) mais le montant du paiement correspond
    # exactement à INV002 (250000) : la règle A doit l'emporter.
    payment = dict(
        payment_id="PMT5", value_date=pd.Timestamp("2024-03-02"), amount=250000,
        currency="EUR", iban_debtor="IBAN_D1", label="VIR FA24000001", bankroll_code="STANDARD",
    )
    result = baseline_match(payment, two_invoice_state, pd.Timestamp("2024-03-02"), **_lookups())
    assert result == "INV001"


def test_no_match_when_nothing_applies(two_invoice_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT6", value_date=pd.Timestamp("2024-03-02"), amount=42,
        currency="EUR", iban_debtor="IBAN_UNKNOWN", label="RIEN DU TOUT", bankroll_code="STANDARD",
    )
    result = baseline_match(payment, two_invoice_state, pd.Timestamp("2024-03-02"), **_lookups())
    assert result is None


# ---------------------------------------------------------------------------
# Métriques bout-en-bout sur données générées
# ---------------------------------------------------------------------------


def test_baseline_precision_is_high_and_coverage_is_reasonable() -> None:
    params = GeneratorParams(n_debtors=40, n_assignors=10, n_invoices=2500, n_months=14, seed=777)
    tables = generate(params)
    journal = build_journal(tables)
    lookups = build_static_lookups(tables)

    ground_truth = tables["ground_truth"]
    true_pairs = set(zip(ground_truth["payment_id"], ground_truth["invoice_id"]))
    matchable = set(ground_truth["payment_id"])

    state = LedgerState()
    predictions: dict[str, str] = {}
    for event in journal:
        if event.type == "PAYMENT_RECEIVED":
            payment = event.data
            inv_id = baseline_match(
                payment,
                state,
                as_of=event.timestamp,
                debtor_by_iban=lookups["debtor_by_iban"],
                assignor_by_iban=lookups["assignor_by_iban"],
            )
            if inv_id is not None:
                predictions[payment["payment_id"]] = inv_id
        state.apply(event)

    n_correct = sum(1 for pid, inv_id in predictions.items() if (pid, inv_id) in true_pairs)
    precision = n_correct / len(predictions) if predictions else 0.0
    coverage = n_correct / len(matchable)

    # La baseline doit être fiable (peu de faux positifs) même si elle ne
    # couvre qu'une partie du volume (pas de résolution de groupe).
    assert precision >= 0.95, f"précision baseline {precision:.2%} trop basse"
    assert 0.4 <= coverage <= 0.95, f"couverture baseline {coverage:.2%} hors plage attendue"
