"""Tests de la featurisation (Phase 4, §5.3 — montant/temporel/textuel/identité)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import Event  # noqa: E402
from src.features import featurize  # noqa: E402
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


def _imputation_event(payment_id, invoice_id, status, updated_at, residual_amount):
    return dict(
        payment_id=payment_id,
        invoice_id=invoice_id,
        status=status,
        updated_at=pd.Timestamp(updated_at),
        residual_amount=residual_amount,
    )


def _payment_event_data(payment_id, value_date, amount, iban_debtor, label, bankroll_code="STANDARD"):
    return dict(
        payment_id=payment_id,
        value_date=pd.Timestamp(value_date),
        amount=amount,
        currency="EUR",
        iban_debtor=iban_debtor,
        iban_creditor="FR_COLLECT",
        label=label,
        channel="SEPA",
        payment_type="VIREMENT",
        bankroll_code=bankroll_code,
    )


@pytest.fixture()
def base_state() -> LedgerState:
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
            _invoice_event("INV001", "DBT001", "AGR001", "FA24000001", "2024-02-01", "2024-03-01", 1_000_000),
        ),
    ]
    for e in events:
        state.apply(e)
    return state


def _lookups() -> dict:
    return dict(
        debtor_by_iban={"IBAN_D1": "DBT001"},
        assignor_by_iban={"IBAN_A1": "ASG001"},
        debtor_name_tokens={"DBT001": ("SARL", "DUPONT")},
    )


def _featurize(payment_data, invoice_id, state, as_of):
    invoice = state.get_invoice(invoice_id, as_of=as_of)
    return featurize(payment_data, invoice, state, as_of, **_lookups())


# --- montant -----------------------------------------------------------------


def test_amount_exact_match(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT1", "2024-03-01", 1_000_000, "IBAN_D1", "VIR FA24000001")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["amount_exact_match"] is True
    assert f["amount_diff_abs"] == 0
    assert f["amount_diff_rel"] == 0.0
    assert f["payment_covers_invoice"] is True


def test_amount_typical_discount_flag(base_state: LedgerState) -> None:
    # Facture à 10 000€ : un écart de 2% (200€) est hors de la bande "frais
    # bancaires" (5-40€ absolus), donc sans ambiguïté entre les deux flags.
    p = _payment_event_data("PMT2", "2024-03-01", 980_000, "IBAN_D1", "VIR FA24000001")  # -2%
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["is_typical_discount"] is True
    assert f["is_bank_fee_gap"] is False
    assert f["is_retention_gap"] is False


def test_amount_bank_fee_flag(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT3", "2024-03-01", 1_000_000 - 2000, "IBAN_D1", "VIR FA24000001")  # -20€
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["is_bank_fee_gap"] is True
    assert f["is_typical_discount"] is False


def test_amount_retention_flag(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT4", "2024-03-01", 950_000, "IBAN_D1", "VIR FA24000001")  # -5%
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["is_retention_gap"] is True


def test_amount_ratio_for_overpayment_group() -> None:
    # amount_ratio > 1 quand le paiement dépasse la facture (paiement groupé)
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
            _invoice_event("INV001", "DBT001", "AGR001", "FA24000001", "2024-02-01", "2024-03-01", 50000),
        ),
    ]:
        state.apply(e)
    p = _payment_event_data("PMT5", "2024-03-01", 120000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", state, pd.Timestamp("2024-03-01"))
    assert f["amount_ratio"] == pytest.approx(2.4)
    assert f["payment_covers_invoice"] is True


# --- temporel ------------------------------------------------------------


def test_days_to_due_signed(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT6", "2024-03-10", 100000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-10"))
    assert f["days_to_due"] == 9  # payé 9 jours après due_date (2024-03-01)


def test_is_before_creation() -> None:
    # Structurellement, LedgerState ne peut jamais renvoyer comme candidat
    # une facture dont l'événement INVOICE_CREATED n'a pas encore été
    # appliqué (la garantie même d'absence de fuite) : is_before_creation
    # ne peut donc jamais être vrai pour un candidat obtenu via l'état en
    # fonctionnement normal. On teste ici uniquement le calcul de la
    # feature elle-même, avec une facture construite à la main.
    state = LedgerState()
    for e in [
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("DBT001",)), dict(party_id="DBT001")),
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("ASG001",)), dict(party_id="ASG001")),
        Event(
            pd.Timestamp("2024-01-01"),
            "AGREEMENT_CREATED",
            (1, ("AGR001",)),
            dict(agreement_id="AGR001", debtor_id="DBT001", client_id="ASG001", market="SERVICES", product="CLASSIQUE", recourse=True),
        ),
    ]:
        state.apply(e)

    manual_invoice = dict(
        invoice_id="INVFUTURE",
        debtor_id="DBT001",
        agreement_id="AGR001",
        client_reference="FA24009999",
        creation_date=pd.Timestamp("2024-02-01"),
        due_date=pd.Timestamp("2024-03-01"),
        currency="EUR",
        initial_amount=100000,
        current_amount=100000,
        agreement_active_at_creation=True,
    )
    p = _payment_event_data("PMT7", "2024-01-15", 100000, "IBAN_D1", "VIR")
    f = featurize(p, manual_invoice, state, pd.Timestamp("2024-01-15"), **_lookups())
    assert f["is_before_creation"] is True


def test_zscore_none_without_history(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT8", "2024-03-05", 100000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-05"))
    assert f["days_to_due_zscore"] is None


def test_zscore_computed_from_prior_history() -> None:
    state = LedgerState()
    base_events = [
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("DBT001",)), dict(party_id="DBT001")),
        Event(
            pd.Timestamp("2024-01-01"),
            "AGREEMENT_CREATED",
            (1, ("AGR001",)),
            dict(agreement_id="AGR001", debtor_id="DBT001", client_id="ASG001", market="SERVICES", product="CLASSIQUE", recourse=True),
        ),
    ]
    for e in base_events:
        state.apply(e)

    # Trois factures déjà soldées, délai constant de 10 jours -> historique du
    # couple (DBT001,AGR001). Chaque étape (création, paiement, imputation)
    # est appliquée pour les trois factures avant de passer à la suivante,
    # pour respecter l'ordre chronologique non décroissant exigé par apply().
    inv_ids = [f"INVHIST{i}" for i in range(3)]
    for inv_id in inv_ids:
        state.apply(
            Event(
                pd.Timestamp("2024-01-05"),
                "INVOICE_CREATED",
                (2, (inv_id,)),
                _invoice_event(inv_id, "DBT001", "AGR001", f"FA{inv_id}", "2024-01-05", "2024-01-20", 10000),
            )
        )
    for inv_id in inv_ids:
        state.apply(
            Event(
                pd.Timestamp("2024-01-30"),
                "PAYMENT_RECEIVED",
                (3, (f"PMT{inv_id}",)),
                _payment_event_data(f"PMT{inv_id}", "2024-01-30", 10000, "IBAN_D1", "VIR"),
            )
        )
    for inv_id in inv_ids:
        state.apply(
            Event(
                pd.Timestamp("2024-01-31"),
                "IMPUTATION_APPLIED",
                (4, (f"PMT{inv_id}", inv_id)),
                _imputation_event(f"PMT{inv_id}", inv_id, "FULL", "2024-01-31", 0),
            )
        )

    # Nouvelle facture, payée exactement au délai moyen historique (10j) -> zscore ~ 0
    state.apply(
        Event(
            pd.Timestamp("2024-02-01"),
            "INVOICE_CREATED",
            (2, ("INVNEW",)),
            _invoice_event("INVNEW", "DBT001", "AGR001", "FANEW", "2024-02-01", "2024-02-15", 20000),
        )
    )
    p = _payment_event_data("PMTNEW", "2024-02-25", 20000, "IBAN_D1", "VIR")  # due 2024-02-15 -> +10j
    f = _featurize(p, "INVNEW", state, pd.Timestamp("2024-02-25"))
    assert f["days_to_due_zscore"] == pytest.approx(0.0, abs=1e-9)

    # Un délai différent de l'historique (parfaitement régulier, std=0) est
    # indéfini plutôt qu'infini : on laisse None.
    state.apply(
        Event(
            pd.Timestamp("2024-02-01"),
            "INVOICE_CREATED",
            (2, ("INVNEW2",)),
            _invoice_event("INVNEW2", "DBT001", "AGR001", "FANEW2", "2024-02-01", "2024-02-15", 20000),
        )
    )
    p2 = _payment_event_data("PMTNEW2", "2024-03-01", 20000, "IBAN_D1", "VIR")  # +14j, pas +10j
    f2 = _featurize(p2, "INVNEW2", state, pd.Timestamp("2024-03-01"))
    assert f2["days_to_due_zscore"] is None


# --- textuel ---------------------------------------------------------------


def test_ref_exact_in_label(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT9", "2024-03-01", 1, "IBAN_UNKNOWN", "VIR FA24000001")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["ref_exact_in_label"] is True


def test_ref_partial_in_label_on_suffix_match(base_state: LedgerState) -> None:
    # "24000001" tronqué en tête à "4000001" (7 caractères, >= 5) : pas de
    # variante exacte, mais un suffixe commun suffisant.
    p = _payment_event_data("PMT10", "2024-03-01", 1, "IBAN_UNKNOWN", "VIR 4000001")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["ref_exact_in_label"] is False
    assert f["ref_partial_in_label"] is True


def test_label_has_no_alpha_true_for_numeric_label(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT11", "2024-03-01", 1, "IBAN_UNKNOWN", "24000001")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["label_has_no_alpha"] is True
    assert f["label_length"] == len("24000001")


def test_name_similarity_high_for_matching_name(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT12", "2024-03-01", 1, "IBAN_UNKNOWN", "VIREMENT SARL DUPONT")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["name_jaro_winkler"] > 0.9
    assert f["name_token_set_ratio"] > 0.9


def test_name_similarity_low_for_unrelated_label(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT13", "2024-03-01", 1, "IBAN_UNKNOWN", "XZQPY WVUTS")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["name_jaro_winkler"] < 0.6


# --- identité / structure ----------------------------------------------------


def test_iban_route_and_matches_debtor(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT14", "2024-03-01", 100000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["iban_route"] == "DEBTOR_DIRECT"
    assert f["iban_matches_invoice_debtor"] is True


def test_iban_matches_invoice_debtor_false_when_route_not_direct(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT15", "2024-03-01", 100000, "IBAN_UNKNOWN", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["iban_route"] == "UNKNOWN"
    assert f["iban_matches_invoice_debtor"] is False


def test_same_agreement_true_for_matching_assignor_reversement(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT16", "2024-03-01", 100000, "IBAN_A1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["iban_route"] == "ASSIGNOR"
    assert f["same_agreement"] is True


def test_assignor_active_at_value_date(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT17", "2024-03-01", 100000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["assignor_active_at_value_date"] is True


def test_agreement_active_at_creation_passthrough(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT18", "2024-03-01", 100000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["agreement_active_at_creation"] is True


# --- contexte contrat --------------------------------------------------------


def test_context_features_passthrough(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT19", "2024-03-01", 100000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["market"] == "SERVICES"
    assert f["product"] == "CLASSIQUE"
    assert f["recourse"] is True


# --- comportementale ---------------------------------------------------------


def test_behavioral_features_cold_start(base_state: LedgerState) -> None:
    p = _payment_event_data("PMT20", "2024-03-01", 100000, "IBAN_D1", "VIR")
    f = _featurize(p, "INV001", base_state, pd.Timestamp("2024-03-01"))
    assert f["debtor_payment_count"] == 0
    assert f["debtor_mean_payment_delay"] is None
    assert f["debtor_ref_citation_rate"] is None
    assert f["debtor_open_invoice_count"] == 1  # INV001 lui-même, encore ouvert
    assert f["debtor_open_invoice_amount"] == 1_000_000


def test_behavioral_features_reflect_prior_history() -> None:
    state = LedgerState()
    base_events = [
        Event(pd.Timestamp("2024-01-01"), "PARTY_OPENED", (0, ("DBT001",)), dict(party_id="DBT001")),
        Event(
            pd.Timestamp("2024-01-01"),
            "AGREEMENT_CREATED",
            (1, ("AGR001",)),
            dict(agreement_id="AGR001", debtor_id="DBT001", client_id="ASG001", market="BTP", product="CLASSIQUE", recourse=False),
        ),
    ]
    for e in base_events:
        state.apply(e)

    # Une facture soldée avec référence citée, une autre payée partiellement
    # sans référence -> historique mixte pour tester les taux.
    state.apply(
        Event(
            pd.Timestamp("2024-01-05"),
            "INVOICE_CREATED",
            (2, ("INVA",)),
            _invoice_event("INVA", "DBT001", "AGR001", "FA24000AAA", "2024-01-05", "2024-01-15", 50000),
        )
    )
    state.apply(
        Event(
            pd.Timestamp("2024-01-05"),
            "INVOICE_CREATED",
            (2, ("INVB",)),
            _invoice_event("INVB", "DBT001", "AGR001", "FA24000BBB", "2024-01-05", "2024-01-15", 30000),
        )
    )
    state.apply(
        Event(pd.Timestamp("2024-01-20"), "PAYMENT_RECEIVED", (3, ("PMTA",)), _payment_event_data("PMTA", "2024-01-20", 50000, "IBAN_D1", "VIR FA24000AAA"))
    )
    state.apply(
        Event(pd.Timestamp("2024-01-21"), "IMPUTATION_APPLIED", (4, ("PMTA", "INVA")), _imputation_event("PMTA", "INVA", "FULL", "2024-01-21", 0))
    )
    state.apply(
        Event(pd.Timestamp("2024-01-25"), "PAYMENT_RECEIVED", (3, ("PMTB",)), _payment_event_data("PMTB", "2024-01-25", 10000, "IBAN_D1", "VIR"))
    )
    state.apply(
        Event(pd.Timestamp("2024-01-26"), "IMPUTATION_APPLIED", (4, ("PMTB", "INVB")), _imputation_event("PMTB", "INVB", "PARTIAL", "2024-01-26", 20000))
    )

    p = _payment_event_data("PMTNEW", "2024-02-01", 20000, "IBAN_D1", "VIR")
    f = _featurize(p, "INVB", state, pd.Timestamp("2024-02-01"))

    assert f["debtor_payment_count"] == 2
    assert f["debtor_partial_payment_rate"] == pytest.approx(0.5)
    assert f["debtor_ref_citation_rate"] == pytest.approx(0.5)  # seule PMTA cite la référence
    assert f["debtor_open_invoice_count"] == 1  # INVB encore ouverte (solde 20000)
    assert f["debtor_open_invoice_amount"] == 20000
