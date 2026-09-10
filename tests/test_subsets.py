"""Tests de l'étage C — résolution d'ensembles (Phase 8, §6)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subsets import (  # noqa: E402
    Candidate,
    find_subsets,
    group_payments_for_nn,
    propose_groups,
    resolve_conflicts,
)


# ---------------------------------------------------------------------------
# find_subsets
# ---------------------------------------------------------------------------


def test_find_subsets_single_exact_match() -> None:
    candidates = [Candidate("INV1", 10000, 0.9), Candidate("INV2", 5000, 0.5)]
    results, budget_exceeded = find_subsets(10000, candidates)
    assert not budget_exceeded
    assert results[0].invoice_ids == ("INV1",)
    assert results[0].amount_gap == 0


def test_find_subsets_finds_pair_summing_to_amount() -> None:
    candidates = [
        Candidate("INV1", 6000, 0.8),
        Candidate("INV2", 4000, 0.7),
        Candidate("INV3", 9999999, 0.6),  # bien trop gros, jamais retenu
    ]
    results, _ = find_subsets(10000, candidates)
    top = results[0]
    assert set(top.invoice_ids) == {"INV1", "INV2"}
    assert top.amount_gap == 0


def test_find_subsets_prefers_smaller_cardinality_at_equal_score() -> None:
    # Un seul candidat couvre exactement le montant ; un couple aussi (même
    # score moyen) : la parcimonie doit départager en faveur du singleton.
    candidates = [
        Candidate("INV_SINGLE", 10000, 0.7),
        Candidate("INV_A", 6000, 0.7),
        Candidate("INV_B", 4000, 0.7),
    ]
    results, _ = find_subsets(10000, candidates)
    assert results[0].invoice_ids == ("INV_SINGLE",)


def test_find_subsets_respects_max_k() -> None:
    candidates = [Candidate(f"INV{i}", 1000, 0.9) for i in range(10)]
    results, _ = find_subsets(10000, candidates, max_k=3, tol_abs=0, tol_rel=0.0)
    # Aucune solution ne peut être trouvée en <= 3 factures de 1000 pour 10000
    assert all(len(r.invoice_ids) <= 3 for r in results)
    assert results == []


def test_find_subsets_respects_tolerance() -> None:
    candidates = [Candidate("INV1", 10600, 0.9)]  # 6% d'écart
    results, _ = find_subsets(10000, candidates, tol_abs=500, tol_rel=0.03)
    assert results == []
    results2, _ = find_subsets(10000, candidates, tol_abs=500, tol_rel=0.10)
    assert results2 and results2[0].invoice_ids == ("INV1",)


def test_find_subsets_filters_low_score_candidates() -> None:
    candidates = [Candidate("INV1", 10000, 0.01)]  # sous min_score par défaut
    results, _ = find_subsets(10000, candidates, min_score=0.05)
    assert results == []


def test_find_subsets_budget_exceeded_flag() -> None:
    candidates = [Candidate(f"INV{i}", 100, 0.9) for i in range(25)]
    results, budget_exceeded = find_subsets(2000, candidates, max_k=5, node_budget=10)
    assert budget_exceeded is True


def test_find_subsets_no_candidates_returns_empty() -> None:
    results, budget_exceeded = find_subsets(10000, [])
    assert results == []
    assert budget_exceeded is False


# ---------------------------------------------------------------------------
# group_payments_for_nn
# ---------------------------------------------------------------------------


def _payment(payment_id: str, debtor_id: str, value_date: str) -> dict:
    return dict(payment_id=payment_id, debtor_id=debtor_id, value_date=pd.Timestamp(value_date))


def test_group_payments_for_nn_groups_within_window() -> None:
    payments = [
        _payment("P1", "D1", "2024-05-01 08:00"),
        _payment("P2", "D1", "2024-05-01 20:00"),
        _payment("P3", "D1", "2024-05-04 10:00"),  # 74h après P1, > fenêtre 72h
    ]
    groups = group_payments_for_nn(payments, window_hours=72)
    group_sets = {frozenset(g) for g in groups}
    assert frozenset({"P1", "P2"}) in group_sets
    assert frozenset({"P3"}) in group_sets


def test_group_payments_for_nn_separates_debtors() -> None:
    payments = [
        _payment("P1", "D1", "2024-05-01 08:00"),
        _payment("P2", "D2", "2024-05-01 09:00"),
    ]
    groups = group_payments_for_nn(payments, window_hours=72)
    assert {"P1"} in [set(g) for g in groups]
    assert {"P2"} in [set(g) for g in groups]


# ---------------------------------------------------------------------------
# propose_groups
# ---------------------------------------------------------------------------


def test_propose_groups_individual_resolution_used_when_available() -> None:
    payments = [_payment("P1", "D1", "2024-05-01")]
    payments[0]["amount"] = 10000
    candidates = {"P1": [Candidate("INV1", 10000, 0.9)]}
    proposals = propose_groups(payments, candidates)
    assert len(proposals) == 1
    assert proposals[0].origin == "individual"
    assert proposals[0].invoice_ids == ("INV1",)


def test_propose_groups_nn_fallback_when_no_individual_match() -> None:
    p1 = _payment("P1", "D1", "2024-05-01 08:00")
    p1["amount"] = 6000
    p2 = _payment("P2", "D1", "2024-05-01 20:00")
    p2["amount"] = 4000
    # Aucune facture ne matche 6000 ou 4000 seule ; la somme 10000 matche
    # exactement l'union des deux factures.
    candidates = {
        "P1": [Candidate("INV1", 7000, 0.6), Candidate("INV2", 3000, 0.5)],
        "P2": [Candidate("INV1", 7000, 0.6), Candidate("INV2", 3000, 0.5)],
    }
    # fallback_min_score désactivé (>1) pour isoler le repli n↔n : sans ça,
    # le repli "candidat unique" résoudrait déjà P1/P2 individuellement.
    proposals = propose_groups([p1, p2], candidates, fallback_min_score=1.1, tol_abs=0, tol_rel=0.0)
    nn_proposals = [p for p in proposals if p.origin == "nn_aggregate"]
    assert len(nn_proposals) == 1
    assert set(nn_proposals[0].payment_ids) == {"P1", "P2"}
    assert set(nn_proposals[0].invoice_ids) == {"INV1", "INV2"}


def test_propose_groups_single_fallback_for_partial_n1_payment() -> None:
    # Paiement partiel (3000) sur une facture ouverte à 10000 : aucune somme
    # ne matche, mais le meilleur candidat doit quand même être proposé,
    # marqué is_full=False (ne clôt pas la facture).
    p1 = _payment("P1", "D1", "2024-05-01")
    p1["amount"] = 3000
    candidates = {"P1": [Candidate("INV1", 10000, 0.8)]}
    proposals = propose_groups([p1], candidates)
    assert len(proposals) == 1
    assert proposals[0].origin == "single_fallback"
    assert proposals[0].is_full is False
    assert proposals[0].invoice_ids == ("INV1",)


def test_propose_groups_single_fallback_for_retention_gap() -> None:
    # Écart de 5% (retenue de garantie) : hors tolérance find_subsets, mais
    # le score de l'étage B a déjà appris ce type d'écart (is_retention_gap).
    p1 = _payment("P1", "D1", "2024-05-01")
    p1["amount"] = 95000  # 5% sous 100000
    candidates = {"P1": [Candidate("INV1", 100000, 0.85)]}
    proposals = propose_groups([p1], candidates, tol_abs=500, tol_rel=0.03)
    assert proposals[0].origin == "single_fallback"
    assert proposals[0].invoice_ids == ("INV1",)


def test_propose_groups_no_fallback_below_min_score() -> None:
    p1 = _payment("P1", "D1", "2024-05-01")
    p1["amount"] = 3000
    candidates = {"P1": [Candidate("INV1", 10000, 0.01)]}
    proposals = propose_groups([p1], candidates, fallback_min_score=0.05)
    assert proposals == []


# ---------------------------------------------------------------------------
# resolve_conflicts
# ---------------------------------------------------------------------------


def test_resolve_conflicts_keeps_higher_score_on_invoice_clash() -> None:
    from src.subsets import GroupProposal

    p_high = GroupProposal(("P1",), ("INV1",), mean_score=0.9, amount_gap=0, origin="individual")
    p_low = GroupProposal(("P2",), ("INV1",), mean_score=0.5, amount_gap=0, origin="individual")
    accepted, rejected = resolve_conflicts([p_low, p_high])
    assert accepted == [p_high]
    assert rejected == [p_low]


def test_resolve_conflicts_keeps_higher_score_on_payment_clash() -> None:
    from src.subsets import GroupProposal

    p_high = GroupProposal(("P1",), ("INV1",), mean_score=0.9, amount_gap=0, origin="individual")
    p_low = GroupProposal(("P1",), ("INV2",), mean_score=0.4, amount_gap=0, origin="individual")
    accepted, rejected = resolve_conflicts([p_low, p_high])
    assert accepted == [p_high]
    assert rejected == [p_low]


def test_resolve_conflicts_accepts_disjoint_proposals() -> None:
    from src.subsets import GroupProposal

    p1 = GroupProposal(("P1",), ("INV1",), mean_score=0.9, amount_gap=0, origin="individual")
    p2 = GroupProposal(("P2",), ("INV2",), mean_score=0.8, amount_gap=0, origin="individual")
    accepted, rejected = resolve_conflicts([p1, p2])
    assert set(accepted) == {p1, p2}
    assert rejected == []


def test_resolve_conflicts_allows_second_partial_claim_on_same_invoice() -> None:
    # n↔1 : deux paiements partiels successifs sur la même facture, aucun
    # des deux n'est is_full -> les deux doivent être acceptés (§6.2).
    from src.subsets import GroupProposal

    p1 = GroupProposal(("P1",), ("INV1",), mean_score=0.9, amount_gap=3000, origin="single_fallback", is_full=False)
    p2 = GroupProposal(("P2",), ("INV1",), mean_score=0.8, amount_gap=0, origin="individual", is_full=True)
    accepted, rejected = resolve_conflicts([p1, p2])
    assert set(accepted) == {p1, p2}
    assert rejected == []


def test_resolve_conflicts_two_full_claims_on_same_invoice_conflict() -> None:
    # Deux clôtures totales concurrentes sur la même facture : un seul
    # gagnant, le score le plus haut.
    from src.subsets import GroupProposal

    p_full_high = GroupProposal(("P1",), ("INV1",), mean_score=0.9, amount_gap=0, origin="individual", is_full=True)
    p_full_low = GroupProposal(("P2",), ("INV1",), mean_score=0.5, amount_gap=0, origin="individual", is_full=True)
    accepted, rejected = resolve_conflicts([p_full_low, p_full_high])
    assert accepted == [p_full_high]
    assert rejected == [p_full_low]


def test_resolve_conflicts_partial_claim_not_blocked_by_prior_full_claim() -> None:
    # Une clôture totale à score élevé (souvent le dernier versement d'un
    # n↔1, qui matche exactement le reliquat) ne doit pas empêcher l'accep-
    # tation d'un versement partiel antérieur sur la même facture, même
    # traité après elle dans le tri par score décroissant (§6.2).
    from src.subsets import GroupProposal

    p_full = GroupProposal(("P2",), ("INV1",), mean_score=0.9, amount_gap=0, origin="individual", is_full=True)
    p_partial = GroupProposal(("P1",), ("INV1",), mean_score=0.5, amount_gap=100, origin="single_fallback", is_full=False)
    accepted, rejected = resolve_conflicts([p_partial, p_full])
    assert set(accepted) == {p_full, p_partial}
    assert rejected == []
