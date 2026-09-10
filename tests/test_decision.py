"""Tests de l'étage D — décision (Phase 9, §7.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision import (  # noqa: E402
    automation_precision_curve,
    classify,
    fit_segment_thresholds,
    threshold_for_target_precision,
    top1_with_margin,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# top1_with_margin
# ---------------------------------------------------------------------------


def test_top1_with_margin_picks_best_and_computes_gap() -> None:
    df = _df(
        [
            dict(payment_id="P1", invoice_id="A", label=False),
            dict(payment_id="P1", invoice_id="B", label=True),
            dict(payment_id="P1", invoice_id="C", label=False),
        ]
    )
    scores = [0.5, 0.9, 0.3]
    top1 = top1_with_margin(df, scores)
    assert len(top1) == 1
    assert top1.iloc[0]["invoice_id"] == "B"
    assert top1.iloc[0]["score"] == pytest.approx(0.9)
    assert top1.iloc[0]["margin"] == pytest.approx(0.4)  # 0.9 - 0.5


def test_top1_with_margin_single_candidate_margin_equals_score() -> None:
    df = _df([dict(payment_id="P2", invoice_id="X", label=True)])
    top1 = top1_with_margin(df, [0.7])
    assert top1.iloc[0]["margin"] == pytest.approx(0.7)


def test_top1_with_margin_ambiguous_pair_has_small_margin() -> None:
    # Deux factures du même débiteur scorent quasi pareil : marge faible,
    # même si le score absolu est élevé (§7.1).
    df = _df(
        [
            dict(payment_id="P3", invoice_id="A", label=True),
            dict(payment_id="P3", invoice_id="B", label=False),
        ]
    )
    top1 = top1_with_margin(df, [0.97, 0.96])
    assert top1.iloc[0]["score"] == pytest.approx(0.97)
    assert top1.iloc[0]["margin"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# threshold_for_target_precision
# ---------------------------------------------------------------------------


def test_threshold_for_target_precision_finds_smallest_valid_tau() -> None:
    top1 = _df(
        [
            dict(score=0.9, label=True),
            dict(score=0.8, label=True),
            dict(score=0.7, label=False),
            dict(score=0.6, label=True),
        ]
    )
    # score>=0.8 -> 2/2 correct (100%) ; score>=0.6 -> 3/4 (75%)
    tau = threshold_for_target_precision(top1, target_precision=0.99)
    assert tau == pytest.approx(0.8)


def test_threshold_for_target_precision_returns_none_when_unreachable() -> None:
    top1 = _df([dict(score=0.9, label=False), dict(score=0.5, label=True)])
    tau = threshold_for_target_precision(top1, target_precision=0.995)
    assert tau is None


# ---------------------------------------------------------------------------
# fit_segment_thresholds
# ---------------------------------------------------------------------------


def test_segment_threshold_falls_back_to_global_when_sparse() -> None:
    rows = [dict(score=0.9, label=True, market="BTP") for _ in range(5)]
    rows += [dict(score=0.9 - i * 0.001, label=True, market="SERVICES") for i in range(200)]
    top1_val = _df(rows)
    thresholds = fit_segment_thresholds(top1_val, ["market"], target_precision=0.99, min_segment_samples=100)
    assert ("BTP",) not in thresholds.per_segment_tau_high  # trop peu d'échantillons
    assert ("SERVICES",) in thresholds.per_segment_tau_high


def test_segment_threshold_used_when_sufficient_samples() -> None:
    # BTP : tolère moins d'automatisation -> seuil de segment plus haut que
    # ce que donnerait un seuil global sur un mélange plus favorable.
    btp_rows = [dict(score=0.5 + i * 0.001, label=(i % 3 != 0), market="BTP") for i in range(200)]
    top1_val = _df(btp_rows)
    thresholds = fit_segment_thresholds(top1_val, ["market"], target_precision=0.9, min_segment_samples=50)
    assert ("BTP",) in thresholds.per_segment_tau_high


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def test_classify_requires_both_score_and_margin_for_auto_validation() -> None:
    from src.decision import SegmentThresholds

    thresholds = SegmentThresholds(global_tau_high=0.95, per_segment_tau_high={}, segment_columns=[])
    top1 = _df(
        [
            dict(payment_id="P1", score=0.97, margin=0.01, label=True),  # score ok, marge trop faible
            dict(payment_id="P2", score=0.97, margin=0.5, label=True),  # les deux conditions ok
            dict(payment_id="P3", score=0.5, margin=0.5, label=False),  # score trop bas
        ]
    )
    result = classify(top1, thresholds, tau_low=0.02, margin_delta=0.05)
    decisions = dict(zip(result["payment_id"], result["decision"]))
    assert decisions["P1"] == "REVIEW"  # ambiguïté malgré un score élevé
    assert decisions["P2"] == "AUTO_VALIDATION"
    assert decisions["P3"] == "REVIEW"


def test_classify_rejects_below_tau_low() -> None:
    from src.decision import SegmentThresholds

    thresholds = SegmentThresholds(global_tau_high=0.95, per_segment_tau_high={}, segment_columns=[])
    top1 = _df([dict(payment_id="P1", score=0.01, margin=0.01, label=False)])
    result = classify(top1, thresholds, tau_low=0.02, margin_delta=0.05)
    assert result.iloc[0]["decision"] == "REJET"


def test_classify_applies_segment_specific_threshold() -> None:
    from src.decision import SegmentThresholds

    thresholds = SegmentThresholds(
        global_tau_high=0.8,
        per_segment_tau_high={("BTP",): 0.95},
        segment_columns=["market"],
    )
    top1 = _df(
        [
            dict(payment_id="P1", score=0.85, margin=0.5, label=True, market="BTP"),  # sous le seuil BTP
            dict(payment_id="P2", score=0.85, margin=0.5, label=True, market="SERVICES"),  # au-dessus du seuil global
        ]
    )
    result = classify(top1, thresholds, tau_low=0.02, margin_delta=0.05)
    decisions = dict(zip(result["payment_id"], result["decision"]))
    assert decisions["P1"] == "REVIEW"
    assert decisions["P2"] == "AUTO_VALIDATION"


# ---------------------------------------------------------------------------
# automation_precision_curve
# ---------------------------------------------------------------------------


def test_automation_precision_curve_shape_and_bounds() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 500
    scores = rng.uniform(0, 1, n)
    labels = scores > rng.uniform(0, 1, n) * 0.3  # corrélé au score, pas parfait
    val = _df([dict(payment_id=f"P{i}", score=s, margin=0.5, label=bool(l)) for i, (s, l) in enumerate(zip(scores, labels))])
    test = val.copy()

    curve = automation_precision_curve(val, test, target_precisions=[0.7, 0.9, 0.99])
    assert len(curve) == 3
    assert (curve["automation_rate"].between(0, 1, inclusive="both")).all()
    assert (curve["realized_precision"].dropna().between(0, 1, inclusive="both")).all()
