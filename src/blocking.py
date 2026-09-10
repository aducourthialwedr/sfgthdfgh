"""Étage A — génération de candidats (§4).

Union des quatre clés de blocking (K1 débiteur, K2 référence, K3 montant,
K4 nom), filtrée par les filtres durs (devise, facture ouverte, agreement
actif à la création). Objectif : rappel proche de 100 %, quitte à générer
beaucoup de faux candidats — l'étage B (scoring) et l'étage D (décision)
se chargent de la précision.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

from src.normalize import best_ngram_similarity, normalize_label
from src.state import LedgerState

# Doit rester aligné avec src.generator.TECHNICAL_BANKROLL.
TECHNICAL_BANKROLL = "TECHNIQUE"

K1_BEFORE_DAYS = 180
K1_AFTER_DAYS = 30
K3_WINDOW_DAYS = 90
K4_WINDOW_DAYS = 90
K4_NAME_SIMILARITY_THRESHOLD = 0.85


def resolve_iban(
    payment: dict,
    debtor_by_iban: dict[str, str],
    assignor_by_iban: dict[str, str],
) -> tuple[str, Optional[str]]:
    """Route un paiement (§3.2) : (catégorie, debtor_id si DEBTOR_DIRECT).

    Le `bankroll_code` arbitre en priorité : un paiement marqué compte
    technique l'est même si son IBAN correspond par ailleurs à un IBAN
    connu. À défaut, lookup IBAN direct sur débiteur puis cédant.
    """
    if payment.get("bankroll_code") == TECHNICAL_BANKROLL:
        return "TECHNICAL_ACCOUNT", None
    iban = payment.get("iban_debtor")
    if iban in debtor_by_iban:
        return "DEBTOR_DIRECT", debtor_by_iban[iban]
    if iban in assignor_by_iban:
        return "ASSIGNOR", None
    return "UNKNOWN", None


def _within_window(
    due_date: pd.Timestamp, value_date: pd.Timestamp, before_days: int, after_days: int
) -> bool:
    return (value_date - pd.Timedelta(days=before_days)) <= due_date <= (
        value_date + pd.Timedelta(days=after_days)
    )


def _name_similarity(name_tokens: tuple[str, ...], label_tokens: tuple[str, ...]) -> float:
    """Similarité max entre le nom (en bloc) et un n-gramme du libellé de
    même longueur en tokens (fenêtre glissante)."""
    return best_ngram_similarity(
        name_tokens, label_tokens, lambda a, b: SequenceMatcher(None, a, b).ratio()
    )


def _passes_hard_filters(invoice: dict, payment: dict) -> bool:
    if invoice["currency"] != payment["currency"]:
        return False
    if invoice["current_amount"] <= 0:
        return False
    if not invoice.get("agreement_active_at_creation", True):
        return False
    return True


def generate_candidates(
    payment: dict,
    state: LedgerState,
    as_of: pd.Timestamp,
    debtor_by_iban: dict[str, str],
    assignor_by_iban: dict[str, str],
    debtor_name_tokens: dict[str, tuple[str, ...]],
) -> list[str]:
    """Union K1∪K2∪K3∪K4, filtrée par les filtres durs. Retourne des
    `invoice_id` triés (déterminisme)."""
    value_date = pd.Timestamp(payment["value_date"])
    label = normalize_label(payment.get("label", ""))
    route, direct_debtor_id = resolve_iban(payment, debtor_by_iban, assignor_by_iban)

    found: dict[str, dict] = {}

    # K1 — débiteur identifié par IBAN
    if route == "DEBTOR_DIRECT" and direct_debtor_id is not None:
        for inv in state.open_invoices(direct_debtor_id, as_of=as_of):
            if _within_window(inv["due_date"], value_date, K1_BEFORE_DAYS, K1_AFTER_DAYS):
                found[inv["invoice_id"]] = inv

    # K2 — référence citée en libellé, sans contrainte temporelle
    if label.label_numbers:
        for inv in state.invoices_by_reference(label.label_numbers, as_of=as_of):
            found[inv["invoice_id"]] = inv

    # K3 — montant exact, ± 90 jours
    for inv in state.invoices_by_amount(int(payment["amount"]), as_of=as_of):
        if _within_window(inv["due_date"], value_date, K3_WINDOW_DAYS, K3_WINDOW_DAYS):
            found[inv["invoice_id"]] = inv

    # K4 — similarité de nom, ± 90 jours
    if label.label_tokens:
        for debtor_id, name_tokens in debtor_name_tokens.items():
            if _name_similarity(name_tokens, label.label_tokens) >= K4_NAME_SIMILARITY_THRESHOLD:
                for inv in state.open_invoices(debtor_id, as_of=as_of):
                    if _within_window(inv["due_date"], value_date, K4_WINDOW_DAYS, K4_WINDOW_DAYS):
                        found[inv["invoice_id"]] = inv

    # K1 bis — débiteur identifié indirectement (K2/K3/K4) plutôt que par IBAN.
    # Cas 1↔n typique : la référence citée en libellé n'identifie qu'UNE des
    # factures groupées ; une fois le débiteur connu par ce biais, ses autres
    # factures ouvertes de la même fenêtre sont des candidats tout aussi
    # légitimes que celles trouvées via K1 (IBAN). Sans ce complément, le
    # rappel global mesuré tombe à 99.1% (1↔n à 97.8%) ; avec, il monte à
    # 99.5% (1↔n à 99.0%).
    inferred_debtor_ids = {inv["debtor_id"] for inv in found.values()}
    inferred_debtor_ids.discard(direct_debtor_id)
    for debtor_id in inferred_debtor_ids:
        for inv in state.open_invoices(debtor_id, as_of=as_of):
            if _within_window(inv["due_date"], value_date, K1_BEFORE_DAYS, K1_AFTER_DAYS):
                found[inv["invoice_id"]] = inv

    candidates = [inv_id for inv_id, inv in found.items() if _passes_hard_filters(inv, payment)]
    return sorted(candidates)


def build_static_lookups(tables: dict[str, pd.DataFrame]) -> dict:
    """Construit les tables statiques (IBAN, tokens de nom) utilisées par
    `generate_candidates`, indépendantes du temps (pas de fuite possible)."""
    debtor_by_iban = dict(zip(tables["debtor"]["iban"], tables["debtor"]["party_id"]))
    assignor_by_iban = dict(zip(tables["assignor"]["iban"], tables["assignor"]["party_id"]))
    debtor_name_tokens = {
        party_id: normalize_label(name).label_tokens
        for party_id, name in zip(tables["debtor"]["party_id"], tables["debtor"]["name"])
    }
    return dict(
        debtor_by_iban=debtor_by_iban,
        assignor_by_iban=assignor_by_iban,
        debtor_name_tokens=debtor_name_tokens,
    )
