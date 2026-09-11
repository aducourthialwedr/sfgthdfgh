"""Tests de l'étage A — génération de candidats (§4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking import (  # noqa: E402
    build_static_lookups,
    generate_candidates,
    resolve_iban,
)
from src.events import build_journal  # noqa: E402
from src.generator import GeneratorParams, generate  # noqa: E402
from src.state import LedgerState  # noqa: E402


def _base_lookups(**overrides) -> dict:
    base = dict(
        debtor_by_iban={"IBAN_D1": "DBT001"},
        assignor_by_iban={"IBAN_A1": "ASG001"},
        technical_ibans=frozenset({"IBAN_TECH1"}),
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# resolve_iban (§3.2)
# ---------------------------------------------------------------------------


def test_resolve_iban_debtor_direct() -> None:
    payment = dict(iban_debtor="IBAN_D1")
    route, debtor_id = resolve_iban(payment, _base_lookups())
    assert route == "DEBTOR_DIRECT"
    assert debtor_id == "DBT001"


def test_resolve_iban_assignor() -> None:
    payment = dict(iban_debtor="IBAN_A1")
    route, debtor_id = resolve_iban(payment, _base_lookups())
    assert route == "ASSIGNOR"
    assert debtor_id is None


def test_resolve_iban_technical_account_by_iban_membership() -> None:
    # payment n'a pas de bankroll_code (schéma réel) : le compte technique
    # se reconnaît par appartenance au référentiel `technical_ibans`, pas
    # par un champ sur le paiement.
    payment = dict(iban_debtor="IBAN_TECH1")
    route, debtor_id = resolve_iban(payment, _base_lookups())
    assert route == "TECHNICAL_ACCOUNT"
    assert debtor_id is None


def test_resolve_iban_technical_overrides_iban_match() -> None:
    # Un IBAN peut en théorie être répertorié à la fois comme débiteur et
    # comme compte technique dans des données mal nettoyées : le compte
    # technique l'emporte (vérifié en premier).
    payment = dict(iban_debtor="IBAN_D1")
    lookups = _base_lookups(technical_ibans=frozenset({"IBAN_D1"}))
    route, debtor_id = resolve_iban(payment, lookups)
    assert route == "TECHNICAL_ACCOUNT"
    assert debtor_id is None


def test_resolve_iban_unknown() -> None:
    payment = dict(iban_debtor="IBAN_X")
    route, debtor_id = resolve_iban(payment, _base_lookups())
    assert route == "UNKNOWN"
    assert debtor_id is None


# ---------------------------------------------------------------------------
# generate_candidates sur un scénario construit à la main (une clé à la fois)
# ---------------------------------------------------------------------------


def _make_invoice_event(
    invoice_id: str,
    debtor_id: str,
    agreement_id: str,
    client_reference: str,
    creation_date: str,
    due_date: str,
    initial_amount: int,
) -> dict:
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
def hand_built_state() -> LedgerState:
    from src.events import Event

    state = LedgerState()
    events = [
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("DBT001",)), dict(party_id="DBT001")),
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("ASG001",)), dict(party_id="ASG001")),
        Event(
            pd.Timestamp("2024-01-01"),
            "AGREEMENT_CREATED",
            (1, ("AGR001",)),
            dict(
                agreement_id="AGR001",
                debtor_id="DBT001",
                client_id="ASG001",
                market="SERVICES",
                product="CLASSIQUE",
                recourse=True,
            ),
        ),
        Event(
            pd.Timestamp("2024-02-01"),
            "INVOICE_CREATED",
            (2, ("INV001",)),
            _make_invoice_event("INV001", "DBT001", "AGR001", "FA24000001", "2024-02-01", "2024-03-01", 100000),
        ),
    ]
    for e in events:
        state.apply(e)
    return state


def _lookups() -> dict:
    debtor_name_tokens = {"DBT001": ("SARL", "DUPONT")}
    return dict(
        debtor_by_iban={"IBAN_D1": "DBT001"},
        assignor_by_iban={"IBAN_A1": "ASG001"},
        technical_ibans=frozenset(),
        debtor_name_tokens=debtor_name_tokens,
        debtor_bankroll_code={"DBT001": "STANDARD"},
        name_index={"DUPONT": {"DBT001"}},
    )


def test_k1_finds_debtor_direct_match_in_window(hand_built_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT1",
        value_date=pd.Timestamp("2024-03-05"),
        amount=999999,  # ne matche ni K2 ni K3
        currency="EUR",
        iban_debtor="IBAN_D1",
        label="VIR SANS REFERENCE",
    )
    candidates = generate_candidates(payment, hand_built_state, pd.Timestamp("2024-03-05"), _lookups())
    assert "INV001" in candidates


def test_k1_respects_window(hand_built_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT1",
        value_date=pd.Timestamp("2025-06-01"),  # bien au-delà de +30j après due_date
        amount=999999,
        currency="EUR",
        iban_debtor="IBAN_D1",
        label="VIR SANS REFERENCE",
    )
    candidates = generate_candidates(payment, hand_built_state, pd.Timestamp("2025-06-01"), _lookups())
    assert "INV001" not in candidates


def test_k2_finds_reference_match_without_window(hand_built_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT2",
        value_date=pd.Timestamp("2025-12-25"),  # très loin de due_date, K2 n'a pas de fenêtre
        amount=1,
        currency="EUR",
        iban_debtor="IBAN_UNKNOWN",
        label="VIR FA24000001",
    )
    candidates = generate_candidates(payment, hand_built_state, pd.Timestamp("2025-12-25"), _lookups())
    assert "INV001" in candidates


def test_k3_finds_exact_amount_match(hand_built_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT3",
        value_date=pd.Timestamp("2024-03-10"),
        amount=100000,
        currency="EUR",
        iban_debtor="IBAN_UNKNOWN",
        label="VIR SANS RIEN",
    )
    candidates = generate_candidates(payment, hand_built_state, pd.Timestamp("2024-03-10"), _lookups())
    assert "INV001" in candidates


def test_k4_finds_name_match(hand_built_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT4",
        value_date=pd.Timestamp("2024-03-10"),
        amount=1,
        currency="EUR",
        iban_debtor="IBAN_UNKNOWN",
        label="VIREMENT SARL DUPONT",
    )
    candidates = generate_candidates(payment, hand_built_state, pd.Timestamp("2024-03-10"), _lookups())
    assert "INV001" in candidates


def test_k4_misses_without_shared_index_token() -> None:
    # L'index inversé de K4 ne compare qu'aux débiteurs partageant un token
    # avec le libellé : si "DUPONT" n'est pas indexé, K4 ne le trouve pas
    # (compromis de rappel assumé pour la scalabilité, voir blocking.py).
    lookups = _lookups()
    lookups["name_index"] = {}
    from src.events import Event

    state = LedgerState()
    for e in [
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
            _make_invoice_event("INV001", "DBT001", "AGR001", "FA24000001", "2024-02-01", "2024-03-01", 100000),
        ),
    ]:
        state.apply(e)
    payment = dict(
        payment_id="PMT4B", value_date=pd.Timestamp("2024-03-10"), amount=1,
        currency="EUR", iban_debtor="IBAN_UNKNOWN", label="VIREMENT SARL DUPONT",
    )
    candidates = generate_candidates(payment, state, pd.Timestamp("2024-03-10"), lookups)
    assert "INV001" not in candidates


def test_hard_filter_excludes_wrong_currency(hand_built_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT5",
        value_date=pd.Timestamp("2024-03-10"),
        amount=100000,
        currency="USD",
        iban_debtor="IBAN_UNKNOWN",
        label="VIR",
    )
    candidates = generate_candidates(payment, hand_built_state, pd.Timestamp("2024-03-10"), _lookups())
    assert "INV001" not in candidates


def test_no_candidates_for_fully_unrelated_payment(hand_built_state: LedgerState) -> None:
    payment = dict(
        payment_id="PMT6",
        value_date=pd.Timestamp("2024-03-10"),
        amount=42,
        currency="EUR",
        iban_debtor="IBAN_UNKNOWN",
        label="RIEN A VOIR ICI",
    )
    candidates = generate_candidates(payment, hand_built_state, pd.Timestamp("2024-03-10"), _lookups())
    assert candidates == []


# ---------------------------------------------------------------------------
# Rappel de blocking sur données générées (le test qui compte vraiment)
# ---------------------------------------------------------------------------


def test_blocking_recall_at_least_99_percent() -> None:
    params = GeneratorParams(n_debtors=40, n_assignors=10, n_invoices=2500, n_months=14, seed=555)
    tables = generate(params)
    journal = build_journal(tables)
    lookups = build_static_lookups(tables)

    state = LedgerState()
    candidates_by_payment: dict[str, set[str]] = {}
    for event in journal:
        if event.type == "PAYMENT_RECEIVED":
            payment = event.data
            candidates = generate_candidates(payment, state, event.timestamp, lookups)
            candidates_by_payment[payment["payment_id"]] = set(candidates)
        state.apply(event)

    ground_truth = tables["ground_truth"]
    hits = sum(
        1
        for row in ground_truth.itertuples(index=False)
        if row.invoice_id in candidates_by_payment.get(row.payment_id, set())
    )
    recall = hits / len(ground_truth)
    assert recall >= 0.99, f"rappel de blocking {recall:.4%} < 99%"
