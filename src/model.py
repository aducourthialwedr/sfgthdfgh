"""Étage B — scoring de paires (§5.1, §5.4).

LightGBM binaire sur les candidats de l'étage A (négatifs = candidats non
retenus, pas d'échantillonnage aléatoire), calibration isotonique sur la
validation, split temporel avec purge.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

NON_FEATURE_COLUMNS = ["payment_id", "invoice_id", "t", "label", "decision_current_amount"]
CATEGORICAL_COLUMNS = ["iban_route", "bankroll_code", "channel", "payment_type", "market", "product"]

# Bornes du split temporel (§5.4). Les 14 mois "pleins" du générateur
# (2024-02 à 2025-03) sont découpés 8 train / 2 val / 2 test ; les deux
# derniers mois (2025-02, 2025-03) et les franges creuses (2024-01,
# 2025-04+) sont volontairement laissés hors Phase 5, réservés au backtest
# de la Phase 10.
TRAIN_START = pd.Timestamp("2024-02-01")
TRAIN_END = pd.Timestamp("2024-10-01")
VAL_END = pd.Timestamp("2024-12-01")
TEST_END = pd.Timestamp("2025-02-01")
PURGE_DAYS = 5


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in NON_FEATURE_COLUMNS]


def prepare_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Caste les colonnes catégorielles en dtype `category` (LightGBM natif)."""
    dataset = dataset.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype("category")
    return dataset


@dataclasses.dataclass
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    dataset: pd.DataFrame,
    train_start: pd.Timestamp = TRAIN_START,
    train_end: pd.Timestamp = TRAIN_END,
    val_end: pd.Timestamp = VAL_END,
    test_end: pd.Timestamp = TEST_END,
    purge_days: int = PURGE_DAYS,
) -> TemporalSplit:
    """Split temporel train/val/test avec purge de `purge_days` de part et
    d'autre de chaque frontière (§5.4) — jamais de split aléatoire."""
    purge = pd.Timedelta(days=purge_days)
    t = dataset["t"]

    train = dataset[(t >= train_start) & (t < train_end - purge)]
    val = dataset[(t >= train_end + purge) & (t < val_end - purge)]
    test = dataset[(t >= val_end + purge) & (t < test_end)]
    return TemporalSplit(train=train, val=val, test=test)


def train_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    seed: int = 42,
) -> lgb.LGBMClassifier:
    """LightGBM binaire, `scale_pos_weight` ajusté sur le train, arrêt
    précoce sur la validation (§5.1)."""
    features = feature_columns(train)
    n_pos = int(train["label"].sum())
    n_neg = len(train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=num_boost_round,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        train[features],
        train["label"],
        eval_X=val[features],
        eval_y=val["label"],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def raw_scores(model: lgb.LGBMClassifier, df: pd.DataFrame) -> np.ndarray:
    features = feature_columns(df)
    return model.predict_proba(df[features])[:, 1]


def fit_calibrator(model: lgb.LGBMClassifier, val: pd.DataFrame) -> IsotonicRegression:
    """Régression isotonique du score brut vers une probabilité calibrée,
    ajustée sur la validation (§7.1)."""
    scores = raw_scores(model, val)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(scores, val["label"].astype(float))
    return calibrator


def calibrated_scores(
    model: lgb.LGBMClassifier, calibrator: IsotonicRegression, df: pd.DataFrame
) -> np.ndarray:
    return calibrator.predict(raw_scores(model, df))


# ---------------------------------------------------------------------------
# Métriques §5.4
# ---------------------------------------------------------------------------


def top1_per_payment(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Réduit le dataset à une ligne par paiement : le candidat de plus haut
    score calibré (utilisé pour precision@1 / MRR / taux d'automatisation)."""
    tmp = df[["payment_id", "invoice_id", "label"]].copy()
    tmp["score"] = scores
    tmp["rank"] = tmp.groupby("payment_id")["score"].rank(method="first", ascending=False)
    return tmp[tmp["rank"] == 1].drop(columns="rank")


def precision_at_1_and_mrr(
    df: pd.DataFrame, scores: np.ndarray, payment_ids_1to1: set[str]
) -> tuple[float, float]:
    """precision@1 et MRR restreints aux paiements réellement 1↔1."""
    tmp = df[["payment_id", "invoice_id", "label"]].copy()
    tmp["score"] = scores
    tmp = tmp[tmp["payment_id"].isin(payment_ids_1to1)]
    if tmp.empty:
        return float("nan"), float("nan")

    tmp["rank"] = tmp.groupby("payment_id")["score"].rank(method="first", ascending=False).astype(int)
    top1 = tmp[tmp["rank"] == 1]
    precision_at_1 = top1["label"].mean()

    true_rows = tmp[tmp["label"]]
    reciprocal_ranks = 1.0 / true_rows["rank"]
    n_payments = tmp["payment_id"].nunique()
    mrr = reciprocal_ranks.sum() / n_payments
    return float(precision_at_1), float(mrr)


def automation_threshold_for_precision(
    df: pd.DataFrame, scores: np.ndarray, target_precision: float = 0.995
) -> Optional[float]:
    """Plus petit seuil sur le score top-1 tel que la précision des
    paiements auto-validés (top-1 >= seuil) atteigne `target_precision`."""
    top1 = top1_per_payment(df, scores).sort_values("score")
    if top1.empty:
        return None
    thresholds = top1["score"].unique()
    best: Optional[float] = None
    for tau in sorted(thresholds):
        selected = top1[top1["score"] >= tau]
        if selected.empty:
            continue
        precision = selected["label"].mean()
        if precision >= target_precision:
            best = float(tau)
            break
    return best


def automation_rate_at_threshold(
    df: pd.DataFrame, scores: np.ndarray, threshold: float, n_total_payments: int
) -> dict:
    """Taux d'automatisation et précision réalisée à un seuil donné, sur le
    top-1 par paiement (§5.4, §7.1)."""
    top1 = top1_per_payment(df, scores)
    selected = top1[top1["score"] >= threshold]
    n_selected = len(selected)
    n_correct = int(selected["label"].sum())
    precision = n_correct / n_selected if n_selected else float("nan")
    automation_rate = n_selected / n_total_payments if n_total_payments else float("nan")
    return dict(
        threshold=threshold,
        n_selected=n_selected,
        n_correct=n_correct,
        precision=precision,
        automation_rate=automation_rate,
    )


def worst_errors(df: pd.DataFrame, scores: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Les `n` erreurs les plus sévères : score et label les plus
    contradictoires (faux positifs à score élevé, faux négatifs à score bas
    sur des vrais positifs)."""
    tmp = df.copy()
    tmp["score"] = scores
    tmp["error"] = np.where(tmp["label"], 1 - tmp["score"], tmp["score"])
    return tmp.sort_values("error", ascending=False).head(n)


# ---------------------------------------------------------------------------
# Deuxième passe et features de compétition (§5.3, §7)
# ---------------------------------------------------------------------------

COMPETITION_COLUMNS = ["rank_in_payment", "score_margin_to_second", "n_candidates"]


def add_competition_features(df: pd.DataFrame, pass1_scores: np.ndarray) -> pd.DataFrame:
    """Dérive `rank_in_payment`, `score_margin_to_second`, `n_candidates` à
    partir des scores bruts d'un premier modèle (§5.3, famille compétition).

    Ces trois features sont calculées par paiement, à partir de l'ensemble
    des candidats de CE paiement uniquement (déjà entièrement connu au
    moment de la décision) — aucune fuite temporelle possible. `n_candidates`
    et `score_margin_to_second` sont répliquées sur toutes les lignes d'un
    même paiement (ambiguïté du paiement dans son ensemble) ; `rank_in_payment`
    varie par ligne.
    """
    out = df.copy()
    out["_pass1_score"] = pass1_scores

    def _per_payment(group: pd.DataFrame) -> pd.DataFrame:
        scores = group["_pass1_score"].to_numpy()
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        top = scores[order[0]]
        second = scores[order[1]] if len(scores) > 1 else 0.0
        return pd.DataFrame(
            {
                "rank_in_payment": ranks,
                "score_margin_to_second": top - second,
                "n_candidates": len(scores),
            },
            index=group.index,
        )

    derived = pd.concat(
        [_per_payment(group) for _, group in out.groupby("payment_id", sort=False)]
    )
    out = out.drop(columns="_pass1_score").join(derived)
    return out


def out_of_fold_scores(
    train: pd.DataFrame, n_splits: int = 5, seed: int = 42, num_boost_round: int = 200
) -> np.ndarray:
    """Scores bruts du modèle de base sur le train, en out-of-fold (§7).

    Score directement sur le train avec le modèle qui l'a appris donnerait
    des scores artificiellement séparés (le modèle a mémorisé ces lignes) et
    biaiserait les features de compétition dérivées. `GroupKFold` sur
    `payment_id` garantit qu'aucune ligne d'un paiement ne participe à
    l'entraînement du modèle qui score ce même paiement — et qu'un paiement
    n'est jamais coupé entre deux folds.
    """
    from sklearn.model_selection import GroupKFold

    features = feature_columns(train)
    scores = np.zeros(len(train), dtype=float)
    groups = train["payment_id"].to_numpy()

    for fold_train_idx, fold_holdout_idx in GroupKFold(n_splits=n_splits).split(train, groups=groups):
        fold_train = train.iloc[fold_train_idx]
        n_pos = int(fold_train["label"].sum())
        n_neg = len(fold_train) - n_pos
        fold_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=num_boost_round,
            scale_pos_weight=(n_neg / n_pos if n_pos else 1.0),
            random_state=seed,
            verbosity=-1,
        )
        fold_model.fit(fold_train[features], fold_train["label"])
        holdout = train.iloc[fold_holdout_idx]
        scores[fold_holdout_idx] = fold_model.predict_proba(holdout[features])[:, 1]
    return scores


@dataclasses.dataclass
class TwoPassResult:
    pass1_model: lgb.LGBMClassifier
    pass2_model: lgb.LGBMClassifier
    pass2_calibrator: IsotonicRegression
    train2: pd.DataFrame
    val2: pd.DataFrame
    test2: pd.DataFrame


def train_two_pass(
    split: TemporalSplit, pass1_model: Optional[lgb.LGBMClassifier] = None
) -> TwoPassResult:
    """Architecture deux passes complète (§5.3 dernière famille) : entraîne
    le modèle de base (ou réutilise `pass1_model` s'il est fourni — évite un
    réentraînement identique quand on compare passe unique / deux passes),
    dérive les features de compétition (out-of-fold sur le train, scores
    directs sur val/test — le modèle de base ne les a jamais vus), puis
    entraîne et calibre le second modèle."""
    if pass1_model is None:
        pass1_model = train_model(split.train, split.val)

    # Les sous-modèles out-of-fold doivent avoir une complexité comparable au
    # modèle de base réel (même best_iteration_, atteint par arrêt précoce),
    # sinon les scores de compétition sur le train (out-of-fold) et sur
    # val/test (modèle de base direct) n'ont pas la même distribution
    # statistique — un décalage train/service qui dégrade la deuxième passe
    # au lieu de l'améliorer.
    oof_rounds = pass1_model.best_iteration_ or pass1_model.n_estimators
    train_pass1_scores = out_of_fold_scores(split.train, num_boost_round=oof_rounds)
    val_pass1_scores = raw_scores(pass1_model, split.val)
    test_pass1_scores = raw_scores(pass1_model, split.test)

    train2 = add_competition_features(split.train, train_pass1_scores)
    val2 = add_competition_features(split.val, val_pass1_scores)
    test2 = add_competition_features(split.test, test_pass1_scores)

    pass2_model = train_model(train2, val2)
    pass2_calibrator = fit_calibrator(pass2_model, val2)

    return TwoPassResult(
        pass1_model=pass1_model,
        pass2_model=pass2_model,
        pass2_calibrator=pass2_calibrator,
        train2=train2,
        val2=val2,
        test2=test2,
    )
