"""Étage C — résolution d'ensembles (§6).

Un score de paire seul ne sait pas dire « ce virement de 12 480 € solde ces
trois factures » : ce module reconstitue les groupes 1↔n (recherche de
sous-ensemble bornée), n↔1 (naturellement via l'état, aucune combinatoire),
n↔n (agrégation heuristique de paiements proches), puis résout les
conflits d'affectation sur l'ensemble du lot.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Optional

import pandas as pd

DEFAULT_MAX_K = 5
DEFAULT_TOL_ABS = 500  # 5 €, en centimes
DEFAULT_TOL_REL = 0.03
DEFAULT_MAX_CANDIDATES_IN = 25
DEFAULT_NODE_BUDGET = 50_000
DEFAULT_MIN_SCORE = 0.05
NN_WINDOW_HOURS = 72


@dataclasses.dataclass(frozen=True)
class Candidate:
    invoice_id: str
    amount: int  # current_amount de la facture au moment de la décision
    score: float  # score calibré de la paire (payment, invoice)


@dataclasses.dataclass(frozen=True)
class SubsetResult:
    invoice_ids: tuple[str, ...]
    total_amount: int
    mean_score: float
    amount_gap: int


def find_subsets(
    payment_amount: int,
    candidates: list[Candidate],
    max_k: int = DEFAULT_MAX_K,
    tol_abs: int = DEFAULT_TOL_ABS,
    tol_rel: float = DEFAULT_TOL_REL,
    max_candidates_in: int = DEFAULT_MAX_CANDIDATES_IN,
    node_budget: int = DEFAULT_NODE_BUDGET,
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[list[SubsetResult], bool]:
    """DFS borné sur les sous-ensembles de factures dont la somme approche
    `payment_amount` (§6.1). Retourne `(résultats triés, budget_dépassé)`.

    Tri des résultats : (moyenne des scores décroissante, écart au montant
    croissant, cardinalité croissante — la parcimonie départage à score égal).
    """
    pool = sorted(
        (c for c in candidates if c.score >= min_score), key=lambda c: -c.score
    )[:max_candidates_in]
    tolerance = max(tol_abs, int(round(payment_amount * tol_rel)))
    upper_bound = payment_amount + tolerance
    lower_bound = payment_amount - tolerance

    results: list[SubsetResult] = []
    state = dict(nodes=0, budget_exceeded=False)

    def dfs(start_idx: int, chosen: list[int], current_sum: int) -> None:
        if state["budget_exceeded"]:
            return
        state["nodes"] += 1
        if state["nodes"] > node_budget:
            state["budget_exceeded"] = True
            return
        if chosen and lower_bound <= current_sum <= upper_bound:
            subset = [pool[i] for i in chosen]
            results.append(
                SubsetResult(
                    invoice_ids=tuple(c.invoice_id for c in subset),
                    total_amount=current_sum,
                    mean_score=sum(c.score for c in subset) / len(subset),
                    amount_gap=abs(current_sum - payment_amount),
                )
            )
        if len(chosen) >= max_k:
            return
        for i in range(start_idx, len(pool)):
            new_sum = current_sum + pool[i].amount
            if new_sum > upper_bound:
                continue
            dfs(i + 1, chosen + [i], new_sum)

    dfs(0, [], 0)
    results.sort(key=lambda r: (-r.mean_score, r.amount_gap, len(r.invoice_ids)))
    return results, state["budget_exceeded"]


def group_payments_for_nn(
    payments: list[dict], window_hours: int = NN_WINDOW_HOURS
) -> list[list[str]]:
    """Regroupe les paiements d'un même débiteur tombant dans une fenêtre
    glissante de `window_hours` (§6.3). Heuristique : fenêtre ancrée sur le
    premier paiement du groupe, pas une segmentation optimale."""
    by_debtor: dict[str, list[dict]] = defaultdict(list)
    for p in payments:
        by_debtor[p["debtor_id"]].append(p)

    groups: list[list[str]] = []
    for plist in by_debtor.values():
        plist = sorted(plist, key=lambda p: pd.Timestamp(p["value_date"]))
        current: list[dict] = []
        anchor: Optional[pd.Timestamp] = None
        for p in plist:
            ts = pd.Timestamp(p["value_date"])
            if anchor is not None and (ts - anchor) > pd.Timedelta(hours=window_hours):
                groups.append([x["payment_id"] for x in current])
                current = []
                anchor = None
            if anchor is None:
                anchor = ts
            current.append(p)
        if current:
            groups.append([x["payment_id"] for x in current])
    return groups


@dataclasses.dataclass(frozen=True)
class GroupProposal:
    payment_ids: tuple[str, ...]
    invoice_ids: tuple[str, ...]
    mean_score: float
    amount_gap: int
    origin: str  # "individual", "nn_aggregate" ou "single_fallback"
    is_full: bool = True  # False : ne clôt pas la facture (n↔1 partiel, écart accepté par le score)


# Le score de repli est une probabilité calibrée (§7.1) : 0.05 signifierait
# « 5 % de chances d'avoir raison », un seuil trop permissif pour proposer
# une affectation. 0.3 exige au moins une préférence nette du modèle.
FALLBACK_MIN_SCORE = 0.3


def propose_groups(
    payments: list[dict],
    candidates_by_payment: dict[str, list[Candidate]],
    fallback_min_score: float = FALLBACK_MIN_SCORE,
    **find_subsets_kwargs,
) -> list[GroupProposal]:
    """Propose un groupe par paiement, en trois passes par ordre de
    préférence décroissante :

    1. Somme exacte individuelle (`find_subsets`) : couvre 1↔1 et 1↔n.
    2. Repli n↔n (§6.3) : agrégation des paiements du même débiteur dans une
       fenêtre de 72h, pour les paiements encore non résolus après (1) —
       tentée *avant* le repli candidat unique, sinon ce dernier absorbe
       tout et le repli n↔n ne s'exécute jamais.
    3. Repli "candidat unique", en dernier recours : le meilleur candidat au
       score le plus élevé, sans exiger que son montant couvre exactement le
       paiement. Couvre n↔1 (paiement partiel : le solde restant, pas
       encore soldé, attend le paiement suivant — §6.2) et les écarts
       métier acceptés par l'étage B (escompte, frais, retenue de garantie :
       `is_typical_discount` etc. ont déjà appris ces écarts, l'étage C n'a
       pas à les re-filtrer sur le montant). Ces propositions ne clôturent
       pas la facture (`is_full=False`) : une autre proposition pourra
       encore la cibler plus tard (l'échéancier n↔1 continue).

    Ne tranche pas les conflits entre propositions — c'est le rôle de
    `resolve_conflicts`.
    """
    payments_by_id = {p["payment_id"]: p for p in payments}
    proposals: list[GroupProposal] = []
    resolved: set[str] = set()

    # Passe 1 — somme exacte individuelle (1↔1, 1↔n)
    for payment_id, payment in payments_by_id.items():
        cands = candidates_by_payment.get(payment_id, [])
        results, _ = find_subsets(int(payment["amount"]), cands, **find_subsets_kwargs)
        if results:
            best = results[0]
            proposals.append(
                GroupProposal(
                    payment_ids=(payment_id,),
                    invoice_ids=best.invoice_ids,
                    mean_score=best.mean_score,
                    amount_gap=best.amount_gap,
                    origin="individual",
                    is_full=True,
                )
            )
            resolved.add(payment_id)

    # Passe 2 — agrégats n↔n (§6.3), sur ce qui reste
    unresolved = [p for p in payments if p["payment_id"] not in resolved]
    for window in group_payments_for_nn(unresolved):
        if len(window) < 2:
            continue
        aggregate_amount = sum(int(payments_by_id[pid]["amount"]) for pid in window)
        pooled: dict[str, Candidate] = {}
        for pid in window:
            for c in candidates_by_payment.get(pid, []):
                if c.invoice_id not in pooled or c.score > pooled[c.invoice_id].score:
                    pooled[c.invoice_id] = c
        results, _ = find_subsets(aggregate_amount, list(pooled.values()), **find_subsets_kwargs)
        if results:
            best = results[0]
            proposals.append(
                GroupProposal(
                    payment_ids=tuple(window),
                    invoice_ids=best.invoice_ids,
                    mean_score=best.mean_score,
                    amount_gap=best.amount_gap,
                    origin="nn_aggregate",
                    is_full=True,
                )
            )
            resolved.update(window)

    # Passe 3 — candidat unique, dernier recours (n↔1 partiel, écarts métier)
    for payment_id, payment in payments_by_id.items():
        if payment_id in resolved:
            continue
        cands = candidates_by_payment.get(payment_id, [])
        if not cands:
            continue
        best_candidate = max(cands, key=lambda c: c.score)
        if best_candidate.score >= fallback_min_score:
            proposals.append(
                GroupProposal(
                    payment_ids=(payment_id,),
                    invoice_ids=(best_candidate.invoice_id,),
                    mean_score=best_candidate.score,
                    amount_gap=abs(best_candidate.amount - int(payment["amount"])),
                    origin="single_fallback",
                    is_full=False,
                )
            )
            resolved.add(payment_id)

    return proposals


def resolve_conflicts(proposals: list[GroupProposal]) -> tuple[list[GroupProposal], list[GroupProposal]]:
    """Glouton sur les scores décroissants (§6.4) : un paiement ne peut être
    affecté qu'une fois.

    Une facture ne peut être *totalement* soldée qu'une fois : deux
    propositions `is_full=True` sur la même facture sont en conflit (une
    seule peut être la clôture). Une proposition `is_full=False` (reliquat
    n↔1, §6.2) ne bloque jamais et n'est jamais bloquée par une conflit de
    facture : plusieurs paiements partiels successifs sur la même facture,
    et son éventuelle clôture finale, coexistent — quel que soit l'ordre
    dans lequel le tri par score les présente (l'ultime versement, qui
    solde exactement le reliquat, obtient souvent un meilleur score que les
    versements intermédiaires et serait traité en premier).
    Retourne `(propositions acceptées, propositions rejetées par conflit)`."""
    ordered = sorted(proposals, key=lambda p: -p.mean_score)
    claimed_invoices_full: set[str] = set()  # factures déjà totalement soldées
    claimed_payments: set[str] = set()
    accepted: list[GroupProposal] = []
    rejected: list[GroupProposal] = []
    for proposal in ordered:
        conflict = any(p in claimed_payments for p in proposal.payment_ids)
        if not conflict and proposal.is_full:
            conflict = any(i in claimed_invoices_full for i in proposal.invoice_ids)
        if conflict:
            rejected.append(proposal)
            continue
        accepted.append(proposal)
        claimed_payments.update(proposal.payment_ids)
        if proposal.is_full:
            claimed_invoices_full.update(proposal.invoice_ids)
    return accepted, rejected
