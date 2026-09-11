"""Étage A — génération de candidats (§4).

Union des quatre clés de blocking (K1 débiteur, K2 référence, K3 montant,
K4 nom), filtrée par les filtres durs (devise, facture ouverte, agreement
actif à la création). Objectif : rappel proche de 100 %, quitte à générer
beaucoup de faux candidats — l'étage B (scoring) et l'étage D (décision)
se chargent de la précision.

Toutes les fonctions publiques prennent un `lookups: dict` unique, produit
par `build_static_lookups` — évite de faire grossir indéfiniment les
signatures à chaque nouveau référentiel statique nécessaire (IBAN, index de
nom...).
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

from src.normalize import best_ngram_similarity, normalize_label
from src.state import LedgerState

K1_BEFORE_DAYS = 180
K1_AFTER_DAYS = 30
K3_WINDOW_DAYS = 90
K4_WINDOW_DAYS = 90
K4_NAME_SIMILARITY_THRESHOLD = 0.85

# Tokens trop fréquents pour être discriminants dans l'index inversé de K4
# (formes sociales) — les indexer ferait retomber sur une comparaison
# quasi exhaustive dès qu'un libellé contient "SARL".
_NAME_INDEX_STOPWORDS = frozenset(
    {"SARL", "SAS", "SA", "EURL", "SNC", "SASU", "SCI", "SCOP", "EI", "EIRL"}
)
_NAME_INDEX_MIN_TOKEN_LEN = 3
# Élagage dynamique des tokens trop fréquents (cf. build_static_lookups) :
# un bucket est coupé au-delà de max(_NAME_INDEX_MIN_BUCKET,
# n_débiteurs / _NAME_INDEX_MAX_BUCKET_FRACTION).
_NAME_INDEX_MIN_BUCKET = 200
_NAME_INDEX_MAX_BUCKET_FRACTION = 500


def resolve_iban(payment: dict, lookups: dict) -> tuple[str, Optional[str]]:
    """Route un paiement (§3.2) : (catégorie, debtor_id si DEBTOR_DIRECT).

    `payment` n'a pas de `bankroll_code` propre (schéma réel) : le compte
    technique/de liaison se reconnaît par appartenance au référentiel
    `technical_ibans` (IBAN internes connus du factor), pas par un champ
    porté par le paiement. À défaut, lookup IBAN direct sur débiteur puis
    cédant.
    """
    iban = payment.get("iban_debtor")
    if iban in lookups["technical_ibans"]:
        return "TECHNICAL_ACCOUNT", None
    if iban in lookups["debtor_by_iban"]:
        return "DEBTOR_DIRECT", lookups["debtor_by_iban"][iban]
    if iban in lookups["assignor_by_iban"]:
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


def _name_index_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(t for t in tokens if t not in _NAME_INDEX_STOPWORDS and len(t) >= _NAME_INDEX_MIN_TOKEN_LEN)


def generate_candidates(
    payment: dict,
    state: LedgerState,
    as_of: pd.Timestamp,
    lookups: dict,
) -> list[str]:
    """Union K1∪K2∪K3∪K4, filtrée par les filtres durs. Retourne des
    `invoice_id` triés (déterminisme)."""
    value_date = pd.Timestamp(payment["value_date"])
    label = normalize_label(payment.get("label", ""))
    route, direct_debtor_id = resolve_iban(payment, lookups)

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

    # K4 — similarité de nom, ± 90 jours. Comparer le libellé à TOUS les
    # débiteurs serait O(n_paiements × n_débiteurs) — infaisable au-delà de
    # quelques dizaines de milliers de débiteurs (mesuré : des jours de
    # calcul à l'échelle réelle, plusieurs millions de débiteurs). On ne
    # compare qu'aux débiteurs partageant au moins un token significatif
    # avec le libellé, via l'index inversé `name_index` construit une fois
    # pour toutes par `build_static_lookups`.
    name_index: dict[str, set[str]] = lookups["name_index"]
    debtor_name_tokens: dict[str, tuple[str, ...]] = lookups["debtor_name_tokens"]
    candidate_debtor_ids: set[str] = set()
    for token in _name_index_tokens(label.label_tokens):
        candidate_debtor_ids |= name_index.get(token, set())
    for debtor_id in candidate_debtor_ids:
        name_tokens = debtor_name_tokens.get(debtor_id, ())
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
    """Construit les référentiels statiques (IBAN, tokens de nom, index
    inversé K4, bankroll_code par débiteur) utilisés par `generate_candidates`
    / `featurize` / `baseline_match`, indépendants du temps (pas de fuite
    possible)."""
    debtor_by_iban = dict(zip(tables["debtor"]["iban"], tables["debtor"]["party_id"]))
    assignor_by_iban = dict(zip(tables["assignor"]["iban"], tables["assignor"]["party_id"]))
    technical_ibans: frozenset[str] = (
        frozenset(tables["_technical_ibans"]["iban"]) if "_technical_ibans" in tables else frozenset()
    )
    debtor_name_tokens = {
        party_id: normalize_label(name).label_tokens
        for party_id, name in zip(tables["debtor"]["party_id"], tables["debtor"]["name"])
    }
    debtor_bankroll_code = dict(zip(tables["debtor"]["party_id"], tables["debtor"]["bankroll_code"]))

    name_index: dict[str, set[str]] = defaultdict(set)
    for party_id, tokens in debtor_name_tokens.items():
        for token in _name_index_tokens(tokens):
            name_index[token].add(party_id)

    # Élagage des tokens trop fréquents (ex. un mot de secteur partagé par
    # une grosse fraction des débiteurs — "INDUSTRIE", "SERVICES"...) : une
    # liste de stopwords figée ne peut pas anticiper ce qui sera fréquent
    # dans des données réelles, donc on mesure et coupe dynamiquement.
    # Un token présent dans un bucket géant n'est de toute façon pas
    # discriminant pour le blocking ; le laisser ferait dégénérer K4 en
    # comparaison quasi exhaustive à grande échelle.
    max_bucket_size = max(_NAME_INDEX_MIN_BUCKET, len(debtor_name_tokens) // _NAME_INDEX_MAX_BUCKET_FRACTION)
    name_index = {
        token: ids for token, ids in name_index.items() if len(ids) <= max_bucket_size
    }

    return dict(
        debtor_by_iban=debtor_by_iban,
        assignor_by_iban=assignor_by_iban,
        technical_ibans=technical_ibans,
        debtor_name_tokens=debtor_name_tokens,
        debtor_bankroll_code=debtor_bankroll_code,
        name_index=name_index,
    )
