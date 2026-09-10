"""Étage D — décision (§7.1).

La calibration isotonique est déjà faite (`model.fit_calibrator`). Ce
module dérive les seuils (global et par segment) à partir d'une précision
cible sur la validation, applique la condition de marge, et classe chaque
paiement en AUTO_VALIDATION / REVIEW / REJET.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_TARGET_PRECISION = 0.995
DEFAULT_MARGIN_DELTA = 0.05
DEFAULT_TAU_LOW = 0.02
MIN_SEGMENT_SAMPLES = 100  # sous ce volume, le seuil de segment n'est pas fiable -> repli sur le seuil global


def top1_with_margin(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Une ligne par paiement : meilleur candidat, son score et la marge
    (écart au deuxième score). Calculés sur `scores` — le score final,
    calibré — pas sur `score_margin_to_second` (§5.3), qui est dérivé du
    score *brut de la passe 1* et sert de feature d'entrée au modèle, pas
    de quantité de décision."""
    tmp = df.copy()
    tmp["score"] = scores
    tmp["rank"] = tmp.groupby("payment_id")["score"].rank(method="first", ascending=False)
    top = tmp[tmp["rank"] == 1].drop(columns="rank").copy()
    second = (
        tmp[tmp["rank"] == 2][["payment_id", "score"]].rename(columns={"score": "second_score"})
    )
    top = top.merge(second, on="payment_id", how="left")
    top["second_score"] = top["second_score"].fillna(0.0)
    top["margin"] = top["score"] - top["second_score"]
    return top.drop(columns="second_score")


def threshold_for_target_precision(
    top1: pd.DataFrame, target_precision: float
) -> Optional[float]:
    """Plus petit seuil sur `score` tel que la précision des paiements avec
    score >= seuil atteigne `target_precision` (§7.1). `None` si aucun seuil
    n'y parvient."""
    if top1.empty:
        return None
    for tau in sorted(top1["score"].unique()):
        selected = top1[top1["score"] >= tau]
        if selected.empty:
            continue
        if selected["label"].mean() >= target_precision:
            return float(tau)
    return None


@dataclasses.dataclass
class SegmentThresholds:
    global_tau_high: float
    per_segment_tau_high: dict[tuple, float]
    segment_columns: list[str]


def fit_segment_thresholds(
    top1_val: pd.DataFrame,
    segment_columns: list[str],
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_segment_samples: int = MIN_SEGMENT_SAMPLES,
) -> SegmentThresholds:
    """τ_high global (§7.1) + un τ_high par segment (`market`,
    `bankroll_code`, tranche de montant...) quand le volume de validation du
    segment est suffisant pour l'estimer sans bruit ; sinon repli sur le
    seuil global — un segment épars ne doit pas produire un seuil erratique.
    """
    global_tau = threshold_for_target_precision(top1_val, target_precision)
    if global_tau is None:
        global_tau = 1.0 + 1e-9  # aucun seuil n'atteint la cible : rien n'est auto-validé

    per_segment: dict[tuple, float] = {}
    if segment_columns:
        for key, group in top1_val.groupby(segment_columns):
            if len(group) < min_segment_samples:
                continue
            tau = threshold_for_target_precision(group, target_precision)
            if tau is not None:
                per_segment[key if isinstance(key, tuple) else (key,)] = tau
    return SegmentThresholds(global_tau, per_segment, segment_columns)


def _tau_high_for_row(thresholds: SegmentThresholds, row: pd.Series) -> float:
    if not thresholds.segment_columns:
        return thresholds.global_tau_high
    key = tuple(row[c] for c in thresholds.segment_columns)
    return thresholds.per_segment_tau_high.get(key, thresholds.global_tau_high)


def classify(
    top1: pd.DataFrame,
    thresholds: SegmentThresholds,
    tau_low: float = DEFAULT_TAU_LOW,
    margin_delta: float = DEFAULT_MARGIN_DELTA,
) -> pd.DataFrame:
    """Classe chaque paiement en AUTO_VALIDATION / REVIEW / REJET (§7.1).
    Retourne `top1` avec une colonne `decision` et `tau_high_applied`."""
    out = top1.copy()
    out["tau_high_applied"] = out.apply(lambda r: _tau_high_for_row(thresholds, r), axis=1)
    is_auto = (out["score"] >= out["tau_high_applied"]) & (out["margin"] >= margin_delta)
    is_review = (~is_auto) & (out["score"] >= tau_low)
    out["decision"] = np.where(is_auto, "AUTO_VALIDATION", np.where(is_review, "REVIEW", "REJET"))
    return out


def automation_precision_curve(
    top1_val: pd.DataFrame,
    top1_test: pd.DataFrame,
    target_precisions: list[float],
    segment_columns: Optional[list[str]] = None,
    margin_delta: float = DEFAULT_MARGIN_DELTA,
    tau_low: float = DEFAULT_TAU_LOW,
) -> pd.DataFrame:
    """Courbe taux d'automatisation / précision (§5.4, §7.1) : pour chaque
    précision cible, calibre le(s) seuil(s) sur la validation puis mesure
    automatisation et précision réalisées sur le test."""
    rows = []
    n_test_payments = len(top1_test)
    for target in target_precisions:
        thresholds = fit_segment_thresholds(top1_val, segment_columns or [], target_precision=target)
        classified = classify(top1_test, thresholds, tau_low=tau_low, margin_delta=margin_delta)
        auto = classified[classified["decision"] == "AUTO_VALIDATION"]
        precision = auto["label"].mean() if len(auto) else float("nan")
        rows.append(
            dict(
                target_precision=target,
                automation_rate=len(auto) / n_test_payments if n_test_payments else float("nan"),
                realized_precision=precision,
                n_auto=len(auto),
            )
        )
    return pd.DataFrame(rows)
