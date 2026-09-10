"""Tests de l'étage B — modèle et métriques (Phase 5, §5.1, §5.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import GeneratorParams, generate  # noqa: E402
from src.model import (  # noqa: E402
    TEST_END,
    TRAIN_END,
    VAL_END,
    add_competition_features,
    automation_rate_at_threshold,
    automation_threshold_for_precision,
    calibrated_scores,
    fit_calibrator,
    out_of_fold_scores,
    precision_at_1_and_mrr,
    prepare_features,
    raw_scores,
    temporal_split,
    train_model,
    train_two_pass,
    worst_errors,
)
from src.replay import build_training_rows  # noqa: E402


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    params = GeneratorParams(n_debtors=60, n_assignors=15, n_invoices=4000, n_months=14, seed=99)
    tables = generate(params)
    raw = build_training_rows(tables)
    return prepare_features(raw)


@pytest.fixture(scope="module")
def split(dataset: pd.DataFrame):
    return temporal_split(dataset, purge_days=3)


def test_split_blocks_are_disjoint_and_ordered(split) -> None:
    assert split.train["t"].max() < split.val["t"].min()
    assert split.val["t"].max() < split.test["t"].min()
    assert not split.train.empty
    assert not split.val.empty
    assert not split.test.empty


def test_split_respects_purge_gap(dataset: pd.DataFrame) -> None:
    s = temporal_split(dataset, purge_days=7)
    gap_train_val = (s.val["t"].min() - s.train["t"].max()).days
    gap_val_test = (s.test["t"].min() - s.val["t"].max()).days
    assert gap_train_val >= 7
    assert gap_val_test >= 7


def test_split_covers_no_rows_beyond_test_end(dataset: pd.DataFrame) -> None:
    s = temporal_split(dataset)
    assert s.test["t"].max() < TEST_END
    assert s.train["t"].min() >= pd.Timestamp("2024-02-01")


@pytest.fixture(scope="module")
def trained(split):
    model = train_model(split.train, split.val, num_boost_round=100, early_stopping_rounds=15)
    calibrator = fit_calibrator(model, split.val)
    return model, calibrator


def test_model_trains_and_predicts(trained, split) -> None:
    model, _ = trained
    from src.model import raw_scores

    scores = raw_scores(model, split.test)
    assert len(scores) == len(split.test)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_calibrator_is_monotonic(trained, split) -> None:
    model, calibrator = trained
    from src.model import raw_scores

    raw = raw_scores(model, split.test)
    order = np.argsort(raw)
    calibrated = calibrator.predict(raw[order])
    # monotone non décroissante par construction de la régression isotonique
    assert np.all(np.diff(calibrated) >= -1e-9)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_model_beats_trivial_score_on_auc(trained, split) -> None:
    from sklearn.metrics import roc_auc_score

    from src.model import raw_scores

    scores = raw_scores(trained[0], split.test)
    auc = roc_auc_score(split.test["label"], scores)
    assert auc > 0.8, f"AUC {auc:.3f} trop faible pour un modèle qui apprend quelque chose"


def test_precision_at_1_and_mrr_bounded(trained, split) -> None:
    model, calibrator = trained
    scores = calibrated_scores(model, calibrator, split.test)
    # Approximation d'un paiement "1↔1" : exactement une ligne positive parmi
    # ses candidats (un vrai 1-n aurait plusieurs positifs pour le même
    # payment_id, ce qui casserait l'invariant MRR<=1 mesuré ci-dessous).
    positive_counts = split.test[split.test["label"]].groupby("payment_id").size()
    payment_ids_1to1 = set(positive_counts[positive_counts == 1].index)
    precision_at_1, mrr = precision_at_1_and_mrr(split.test, scores, payment_ids_1to1)
    assert 0.0 <= precision_at_1 <= 1.0
    assert 0.0 <= mrr <= 1.0
    assert mrr >= precision_at_1 - 1e-9  # MRR >= precision@1 toujours (rang 1 contribue 1.0)


def test_automation_threshold_reaches_target_precision(trained, split) -> None:
    model, calibrator = trained
    val_scores = calibrated_scores(model, calibrator, split.val)
    tau = automation_threshold_for_precision(split.val, val_scores, target_precision=0.99)
    if tau is None:
        pytest.skip("aucun seuil n'atteint la précision cible sur cet échantillon réduit")
    test_scores = calibrated_scores(model, calibrator, split.test)
    result = automation_rate_at_threshold(
        split.test, test_scores, tau, n_total_payments=split.test["payment_id"].nunique()
    )
    assert 0.0 <= result["automation_rate"] <= 1.0
    assert result["n_selected"] >= 0


def test_worst_errors_returns_n_rows_sorted_by_severity(trained, split) -> None:
    model, calibrator = trained
    scores = calibrated_scores(model, calibrator, split.test)
    errs = worst_errors(split.test, scores, n=10)
    assert len(errs) <= 10
    assert (errs["error"].diff().dropna() <= 1e-9).all()  # décroissant


# ---------------------------------------------------------------------------
# Deuxième passe et features de compétition (Phase 7)
# ---------------------------------------------------------------------------


def test_competition_features_rank_and_margin_within_one_payment() -> None:
    df = pd.DataFrame(
        {
            "payment_id": ["P1", "P1", "P1"],
            "invoice_id": ["INV_A", "INV_B", "INV_C"],
            "t": pd.Timestamp("2024-05-01"),
            "label": [False, True, False],
        }
    )
    scores = np.array([0.5, 0.9, 0.1])  # meilleur candidat = INV_B
    out = add_competition_features(df, scores)

    ranks = dict(zip(out["invoice_id"], out["rank_in_payment"]))
    assert ranks == {"INV_B": 1, "INV_A": 2, "INV_C": 3}
    # marge = top(0.9) - second(0.5), répliquée sur toutes les lignes du paiement
    assert np.allclose(out["score_margin_to_second"].to_numpy(), 0.4)
    assert (out["n_candidates"] == 3).all()


def test_competition_features_single_candidate_margin_equals_score() -> None:
    df = pd.DataFrame(
        {"payment_id": ["P2"], "invoice_id": ["INV_X"], "t": [pd.Timestamp("2024-05-01")], "label": [True]}
    )
    out = add_competition_features(df, np.array([0.7]))
    assert out["score_margin_to_second"].iloc[0] == pytest.approx(0.7)
    assert out["rank_in_payment"].iloc[0] == 1
    assert out["n_candidates"].iloc[0] == 1


def test_competition_features_independent_across_payments() -> None:
    df = pd.DataFrame(
        {
            "payment_id": ["P1", "P1", "P2", "P2", "P2"],
            "invoice_id": ["A1", "A2", "B1", "B2", "B3"],
            "t": pd.Timestamp("2024-05-01"),
            "label": [True, False, False, True, False],
        }
    )
    scores = np.array([0.6, 0.4, 0.9, 0.2, 0.1])
    out = add_competition_features(df, scores)
    assert set(out.loc[out["payment_id"] == "P1", "n_candidates"]) == {2}
    assert set(out.loc[out["payment_id"] == "P2", "n_candidates"]) == {3}
    p1_margin = out.loc[out["payment_id"] == "P1", "score_margin_to_second"].iloc[0]
    p2_margin = out.loc[out["payment_id"] == "P2", "score_margin_to_second"].iloc[0]
    assert p1_margin == pytest.approx(0.2)  # 0.6 - 0.4
    assert p2_margin == pytest.approx(0.7)  # 0.9 - 0.2


def test_out_of_fold_scores_bounded_and_full_length(split) -> None:
    scores = out_of_fold_scores(split.train, n_splits=3, num_boost_round=50)
    assert len(scores) == len(split.train)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_out_of_fold_scores_less_optimistic_than_in_sample(split) -> None:
    """Les scores out-of-fold ne doivent pas être aussi artificiellement
    séparés qu'un score in-sample (le modèle n'a jamais vu la ligne qu'il
    score) — sanity check contre une fuite qui annulerait tout l'intérêt du
    out-of-fold."""
    in_sample_model = train_model(split.train, split.val, num_boost_round=50, early_stopping_rounds=1000)
    in_sample_scores = raw_scores(in_sample_model, split.train)
    oof_scores = out_of_fold_scores(split.train, n_splits=3, num_boost_round=50)

    from sklearn.metrics import roc_auc_score

    in_sample_auc = roc_auc_score(split.train["label"], in_sample_scores)
    oof_auc = roc_auc_score(split.train["label"], oof_scores)
    assert oof_auc <= in_sample_auc + 1e-9


@pytest.fixture(scope="module")
def two_pass(split):
    return train_two_pass(split)


def test_two_pass_adds_competition_columns(two_pass) -> None:
    from src.model import COMPETITION_COLUMNS

    for col in COMPETITION_COLUMNS:
        assert col in two_pass.test2.columns


def test_two_pass_model_predicts_valid_probabilities(two_pass) -> None:
    scores = raw_scores(two_pass.pass2_model, two_pass.test2)
    assert len(scores) == len(two_pass.test2)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_two_pass_score_separation_is_reasonable(two_pass) -> None:
    from sklearn.metrics import roc_auc_score

    scores = calibrated_scores(two_pass.pass2_model, two_pass.pass2_calibrator, two_pass.test2)
    auc = roc_auc_score(two_pass.test2["label"], scores)
    assert auc > 0.8, f"AUC deuxième passe {auc:.3f} trop faible"
