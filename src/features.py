"""Featurisation d'une paire (payment, invoice) — §5.3.

Phase 4 : familles montant, temporel, textuel, identité/structure.
Comportementale et contexte contrat arrivent en Phase 6, compétition en
Phase 7 — volontairement absentes ici.

Toute feature qui dépend du temps passe par `LedgerState` avec un `as_of`
explicite (§8.2) : c'est la garantie mécanique d'absence de fuite.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.blocking import resolve_iban
from src.normalize import (
    best_ngram_similarity,
    longest_common_suffix_len,
    normalize_label,
    normalize_reference,
    normalize_reference_primary,
)
from src.state import LedgerState

DISCOUNT_REL_RANGE = (0.005, 0.03)
BANK_FEE_ABS_RANGE_CENTS = (500, 4000)
RETENTION_REL_RANGE = (0.04, 0.06)
REF_PARTIAL_MIN_SUFFIX = 5


def _amount_features(payment: dict, current_amount: int) -> dict:
    amount = int(payment["amount"])
    diff_abs = amount - current_amount
    if current_amount > 0:
        diff_rel = diff_abs / current_amount
        amount_ratio: Optional[float] = amount / current_amount
    else:
        diff_rel = None
        amount_ratio = None

    is_underpayment = diff_abs < 0
    abs_diff_rel = abs(diff_rel) if diff_rel is not None else None

    return dict(
        amount_diff_abs=diff_abs,
        amount_diff_rel=diff_rel,
        amount_exact_match=diff_abs == 0,
        payment_covers_invoice=amount >= current_amount,
        amount_ratio=amount_ratio,
        is_typical_discount=bool(
            is_underpayment
            and abs_diff_rel is not None
            and DISCOUNT_REL_RANGE[0] <= abs_diff_rel <= DISCOUNT_REL_RANGE[1]
        ),
        is_bank_fee_gap=bool(
            is_underpayment
            and BANK_FEE_ABS_RANGE_CENTS[0] <= abs(diff_abs) <= BANK_FEE_ABS_RANGE_CENTS[1]
        ),
        is_retention_gap=bool(
            is_underpayment
            and abs_diff_rel is not None
            and RETENTION_REL_RANGE[0] <= abs_diff_rel <= RETENTION_REL_RANGE[1]
        ),
    )


def _temporal_features(
    payment: dict, invoice: dict, state: LedgerState, as_of: pd.Timestamp
) -> dict:
    value_date = pd.Timestamp(payment["value_date"])
    due_date = pd.Timestamp(invoice["due_date"])
    creation_date = pd.Timestamp(invoice["creation_date"])
    days_to_due = (value_date - due_date).days

    zscore: Optional[float] = None
    stats = state.delay_stats(invoice["debtor_id"], invoice["agreement_id"], as_of=as_of)
    if stats is not None:
        mean, std, n = stats
        if n >= 2:
            if std > 0:
                zscore = (days_to_due - mean) / std
            elif days_to_due == mean:
                # historique parfaitement régulier (std=0) : un délai
                # identique est "parfaitement typique" -> z-score nul.
                # S'il diffère, l'écart est indéfini (division par zéro
                # évitée), on laisse `None` plutôt qu'inventer une valeur.
                zscore = 0.0

    return dict(
        days_to_due=days_to_due,
        days_since_creation=(value_date - creation_date).days,
        is_before_creation=value_date < creation_date,
        days_to_due_zscore=zscore,
    )


def _textual_features(
    payment: dict, invoice: dict, debtor_name_tokens: tuple[str, ...]
) -> dict:
    label = normalize_label(payment.get("label", ""))
    ref_variants = normalize_reference(invoice["client_reference"])
    ref_primary = normalize_reference_primary(invoice["client_reference"])
    invoice_id_variants = normalize_reference(invoice["invoice_id"])

    ref_partial = False
    for token in label.label_numbers:
        if longest_common_suffix_len(ref_primary, token) >= REF_PARTIAL_MIN_SUFFIX:
            ref_partial = True
            break

    name_str = " ".join(debtor_name_tokens)
    name_jaro_winkler = best_ngram_similarity(
        debtor_name_tokens, label.label_tokens, JaroWinkler.normalized_similarity
    )
    name_token_set_ratio = (
        fuzz.token_set_ratio(name_str, label.normalized) / 100.0 if name_str and label.normalized else 0.0
    )

    return dict(
        ref_exact_in_label=bool(ref_variants & label.label_numbers),
        ref_partial_in_label=ref_partial,
        invoice_id_in_label=bool(invoice_id_variants & label.label_numbers),
        name_jaro_winkler=name_jaro_winkler,
        name_token_set_ratio=name_token_set_ratio,
        label_length=len(payment.get("label") or ""),
        label_has_no_alpha=len(label.label_tokens) == 0,
    )


def _identity_features(
    payment: dict,
    invoice: dict,
    state: LedgerState,
    as_of: pd.Timestamp,
    debtor_by_iban: dict[str, str],
    assignor_by_iban: dict[str, str],
) -> dict:
    route, direct_debtor_id = resolve_iban(payment, debtor_by_iban, assignor_by_iban)
    agreement = state.get_agreement(invoice["agreement_id"], as_of=as_of)
    assignor_id = agreement["client_id"] if agreement is not None else None

    same_agreement = False
    if route == "ASSIGNOR":
        iban_assignor_id = assignor_by_iban.get(payment.get("iban_debtor"))
        same_agreement = iban_assignor_id is not None and iban_assignor_id == assignor_id

    assignor_active = (
        state.party_is_active(assignor_id, as_of=as_of) if assignor_id is not None else False
    )

    return dict(
        iban_route=route,
        iban_matches_invoice_debtor=bool(
            route == "DEBTOR_DIRECT" and direct_debtor_id == invoice["debtor_id"]
        ),
        bankroll_code=payment.get("bankroll_code"),
        channel=payment.get("channel"),
        payment_type=payment.get("payment_type"),
        same_agreement=same_agreement,
        assignor_active_at_value_date=assignor_active,
        agreement_active_at_creation=bool(invoice.get("agreement_active_at_creation", False)),
    )


BEHAVIORAL_WINDOW_DAYS = 180  # ~6 mois glissants (§5.3)


def _behavioral_features(invoice: dict, state: LedgerState, as_of: pd.Timestamp) -> dict:
    """Agrégats comportementaux du débiteur, fenêtre glissante strictement
    antérieure à `as_of` — entièrement délégués à `LedgerState`, qui les
    maintient de façon incrémentale (§8.2)."""
    return state.behavioral_stats(
        invoice["debtor_id"], as_of=as_of, window_days=BEHAVIORAL_WINDOW_DAYS
    )


def _context_features(invoice: dict, state: LedgerState, as_of: pd.Timestamp) -> dict:
    """Contexte contrat : `market`, `product`, `recourse` (§5.3). Statiques
    une fois l'agreement créé — `as_of` sert uniquement à respecter
    l'interface de lecture de `LedgerState`, pas à filtrer une évolution."""
    agreement = state.get_agreement(invoice["agreement_id"], as_of=as_of)
    if agreement is None:
        return dict(market=None, product=None, recourse=None)
    return dict(
        market=agreement["market"],
        product=agreement["product"],
        recourse=bool(agreement["recourse"]),
    )


def featurize(
    payment: dict,
    invoice: dict,
    state: LedgerState,
    as_of: pd.Timestamp,
    debtor_by_iban: dict[str, str],
    assignor_by_iban: dict[str, str],
    debtor_name_tokens: dict[str, tuple[str, ...]],
) -> dict:
    """Calcule toutes les features des familles montant, temporel, textuel,
    identité/structure (Phase 4), comportementale et contexte contrat
    (Phase 6) pour la paire `(payment, invoice)`, à `as_of`."""
    features: dict = {}
    features.update(_amount_features(payment, int(invoice["current_amount"])))
    features.update(_temporal_features(payment, invoice, state, as_of))
    features.update(
        _textual_features(payment, invoice, debtor_name_tokens.get(invoice["debtor_id"], ()))
    )
    features.update(
        _identity_features(payment, invoice, state, as_of, debtor_by_iban, assignor_by_iban)
    )
    features.update(_behavioral_features(invoice, state, as_of))
    features.update(_context_features(invoice, state, as_of))
    return features
