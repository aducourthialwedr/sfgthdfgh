"""Tests de l'orchestration de production / backtest (Phase 10, §8.4, §8.5).

Utilise un modèle factice (règle simple sur `amount_exact_match`) plutôt que
LightGBM entraîné : ce module teste l'orchestration (cadence quotidienne,
visibilité globale par jour, rétention du reliquat), pas la qualité du
scoring — déjà couverte par tests/test_model.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.batch import run_backtest  # noqa: E402
from src.decision import SegmentThresholds  # noqa: E402
from src.generator import GeneratorParams, generate  # noqa: E402


class _FakeModel:
    """Score = 0.95 si amount_exact_match, sinon 0.1 + un peu de bruit
    déterministe sur days_to_due pour départager les candidats."""

    def __init__(self):
        self.booster_ = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base = np.where(X["amount_exact_match"].to_numpy(), 0.95, 0.1)
        jitter = -np.abs(X["days_to_due"].to_numpy()) * 0.001
        p = np.clip(base + jitter, 0.0, 1.0)
        return np.stack([1 - p, p], axis=1)


class _FakeCalibrator:
    def predict(self, scores: np.ndarray) -> np.ndarray:
        return scores


@pytest.fixture(scope="module")
def small_tables() -> dict[str, pd.DataFrame]:
    params = GeneratorParams(n_debtors=20, n_assignors=5, n_invoices=800, n_months=14, seed=13)
    return generate(params)


def test_backtest_produces_decisions_with_valid_labels(small_tables: dict[str, pd.DataFrame]) -> None:
    payments = small_tables["payment"]
    end = pd.Timestamp(payments["value_date"].max())
    start = end - pd.Timedelta(days=10)

    thresholds = SegmentThresholds(global_tau_high=0.5, per_segment_tau_high={}, segment_columns=[])
    decisions, pending = run_backtest(
        small_tables, start, end,
        _FakeModel(), _FakeModel(), _FakeCalibrator(),
        thresholds, amount_breakpoints=(100000, 500000),
        margin_delta=0.0, tau_low=0.05, retention_days=60,
    )
    assert decisions, "aucune décision produite sur la fenêtre de test"
    for d in decisions:
        assert d.decision in {"AUTO_VALIDATION", "REVIEW", "REJET"}
        assert 0.0 <= d.score <= 1.0


def test_backtest_auto_validated_payments_leave_pending_queue(small_tables: dict[str, pd.DataFrame]) -> None:
    payments = small_tables["payment"]
    end = pd.Timestamp(payments["value_date"].max())
    start = end - pd.Timedelta(days=15)

    thresholds = SegmentThresholds(global_tau_high=0.5, per_segment_tau_high={}, segment_columns=[])
    decisions, pending = run_backtest(
        small_tables, start, end,
        _FakeModel(), _FakeModel(), _FakeCalibrator(),
        thresholds, amount_breakpoints=(100000, 500000),
        margin_delta=0.0, tau_low=0.05, retention_days=60,
    )
    auto_validated_payment_ids = {
        pid for d in decisions if d.decision == "AUTO_VALIDATION" for pid in d.proposal.payment_ids
    }
    assert auto_validated_payment_ids, "aucune auto-validation sur cet échantillon"
    assert not (auto_validated_payment_ids & set(pending.keys()))


def test_backtest_retention_expires_unresolved_payments(small_tables: dict[str, pd.DataFrame]) -> None:
    payments = small_tables["payment"]
    end = pd.Timestamp(payments["value_date"].max())
    start = end - pd.Timedelta(days=10)

    # tau_high inatteignable : rien n'est jamais auto-validé, tout reste en
    # attente jusqu'à expiration immédiate (retention_days=0).
    thresholds = SegmentThresholds(global_tau_high=2.0, per_segment_tau_high={}, segment_columns=[])
    decisions, pending = run_backtest(
        small_tables, start, end,
        _FakeModel(), _FakeModel(), _FakeCalibrator(),
        thresholds, amount_breakpoints=(100000, 500000),
        margin_delta=0.0, tau_low=0.0, retention_days=0,
    )
    assert all(d.decision != "AUTO_VALIDATION" for d in decisions)
    # rien ne devrait s'accumuler indéfiniment : à rétention nulle, la file
    # ne contient que les paiements du tout dernier jour traité.
    assert len(pending) <= payments[
        (pd.to_datetime(payments["value_date"]) >= end.normalize())
        & (pd.to_datetime(payments["value_date"]) < end.normalize() + pd.Timedelta(days=1))
    ].shape[0] + 1


def test_backtest_same_day_conflict_arbitrated_globally(small_tables: dict[str, pd.DataFrame]) -> None:
    """Deux paiements traités le même jour ne doivent jamais se voir tous
    les deux attribuer la même facture en AUTO_VALIDATION (§8.4, étape 4)."""
    payments = small_tables["payment"]
    end = pd.Timestamp(payments["value_date"].max())
    start = end - pd.Timedelta(days=20)

    thresholds = SegmentThresholds(global_tau_high=0.5, per_segment_tau_high={}, segment_columns=[])
    decisions, _pending = run_backtest(
        small_tables, start, end,
        _FakeModel(), _FakeModel(), _FakeCalibrator(),
        thresholds, amount_breakpoints=(100000, 500000),
        margin_delta=0.0, tau_low=0.05, retention_days=60,
    )
    auto = [d for d in decisions if d.decision == "AUTO_VALIDATION"]
    seen_invoices: set[str] = set()
    for d in auto:
        for inv_id in d.proposal.invoice_ids:
            assert inv_id not in seen_invoices or not d.proposal.is_full, (
                f"facture {inv_id} auto-validée deux fois"
            )
            if d.proposal.is_full:
                seen_invoices.add(inv_id)
