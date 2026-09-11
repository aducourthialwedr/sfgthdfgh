"""Baseline déterministe (Phase 3) : référence exacte en libellé, ou montant
exact sur un débiteur identifié sans ambiguïté. Aucun apprentissage — c'est
le plancher auquel tous les modèles ultérieurs (étage B) sont comparés.

Ne tente aucune résolution de groupe (1↔n, n↔1, n↔n) : une seule paire par
paiement, seulement quand une règle s'applique sans ambiguïté. La
combinatoire est le rôle de l'étage C (Phase 8), pas de la baseline.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.blocking import build_static_lookups, resolve_iban
from src.events import build_journal
from src.normalize import normalize_label, normalize_reference_primary
from src.state import LedgerState

AMOUNT_WINDOW_DAYS = 90


def _within_window(due_date: pd.Timestamp, value_date: pd.Timestamp, days: int) -> bool:
    return (value_date - pd.Timedelta(days=days)) <= due_date <= (value_date + pd.Timedelta(days=days))


def baseline_match(
    payment: dict,
    state: LedgerState,
    as_of: pd.Timestamp,
    lookups: dict,
) -> Optional[str]:
    """Propose un unique `invoice_id`, ou `None` si aucune règle ne
    s'applique sans ambiguïté.

    Règle A (référence) : parmi les factures dont une variante de référence
    figure dans le libellé, on préfère les matches "forts" (la référence
    complète, non tronquée, apparaît telle quelle) ; à défaut, un match
    "faible" (variante sans préfixe/zéros) n'est retenu que s'il est unique.

    Règle B (montant) : n'est tentée que si le débiteur est identifié sans
    ambiguïté (IBAN → DEBTOR_DIRECT), et seulement si une unique facture
    ouverte de ce débiteur a le montant exact, dans une fenêtre de
    ±90 jours autour de l'échéance.
    """
    value_date = pd.Timestamp(payment["value_date"])
    label = normalize_label(payment.get("label", ""))

    if label.label_numbers:
        candidates = [
            c
            for c in state.invoices_by_reference(label.label_numbers, as_of=as_of)
            if c["currency"] == payment["currency"]
        ]
        strong = [
            c
            for c in candidates
            if normalize_reference_primary(c["client_reference"]) in label.label_numbers
        ]
        pool = strong if strong else candidates
        if len(pool) == 1:
            return pool[0]["invoice_id"]

    route, debtor_id = resolve_iban(payment, lookups)
    if route == "DEBTOR_DIRECT" and debtor_id is not None:
        matches = [
            inv
            for inv in state.open_invoices(debtor_id, as_of=as_of)
            if inv["current_amount"] == payment["amount"]
            and inv["currency"] == payment["currency"]
            and _within_window(inv["due_date"], value_date, AMOUNT_WINDOW_DAYS)
        ]
        if len(matches) == 1:
            return matches[0]["invoice_id"]

    return None


def compute_baseline_predictions(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Rejoue le journal complet et applique `baseline_match` à chaque
    paiement (même discipline §8.3 : avant `state.apply`). Retourne
    `{payment_id: invoice_id}` pour les seuls paiements où une règle a
    tranché."""
    journal = build_journal(tables)
    lookups = build_static_lookups(tables)

    state = LedgerState()
    predictions: dict[str, str] = {}
    for event in journal:
        if event.type == "PAYMENT_RECEIVED":
            payment = event.data
            invoice_id = baseline_match(payment, state, as_of=event.timestamp, lookups=lookups)
            if invoice_id is not None:
                predictions[payment["payment_id"]] = invoice_id
        state.apply(event)
    return predictions
