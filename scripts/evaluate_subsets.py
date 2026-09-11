"""Évalue l'étage C (Phase 8, §6) : performance de reconstruction de groupes
par type de cas (1↔1, 1↔n, n↔1, n↔n), sur la période de test.

Réutilise les modèles entraînés en Phase 7 (models/*.joblib) — ne les
réentraîne pas.

Usage :
    python scripts/train_model.py   # si models/ est vide
    python scripts/evaluate_subsets.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking import build_static_lookups, resolve_iban  # noqa: E402
from src.model import add_competition_features, calibrated_scores, prepare_features, raw_scores, temporal_split  # noqa: E402
from src.subsets import Candidate, propose_groups, resolve_conflicts  # noqa: E402

MODELS_DIR = Path("models")


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement", "ground_truth", "_technical_ibans"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "dataset").glob("month=*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)

    dataset = prepare_features(load_dataset(data_dir))
    split = temporal_split(dataset)

    pass1_model = joblib.load(MODELS_DIR / "stage_b_pass1_model.joblib")
    pass2_model = joblib.load(MODELS_DIR / "stage_b_pass2_model.joblib")
    calibrator = joblib.load(MODELS_DIR / "stage_b_pass2_calibrator.joblib")

    test1_scores = raw_scores(pass1_model, split.test)
    test2 = add_competition_features(split.test, test1_scores)
    test2 = test2.copy()
    test2["score"] = calibrated_scores(pass2_model, calibrator, test2)

    # --- reconstruit ce qu'il faut pour propose_groups (montant, débiteur) ---
    payment_raw = tables["payment"].set_index("payment_id")
    invoice_debtor = tables["invoice"].set_index("invoice_id")["debtor_id"]
    lookups = build_static_lookups(tables)

    candidates_by_payment: dict[str, list[Candidate]] = defaultdict(list)
    for row in test2.itertuples():
        payment_amount = int(payment_raw.loc[row.payment_id, "amount"])
        current_amount = payment_amount - int(row.amount_diff_abs)
        candidates_by_payment[row.payment_id].append(
            Candidate(invoice_id=row.invoice_id, amount=current_amount, score=float(row.score))
        )

    payments = []
    for payment_id, cands in candidates_by_payment.items():
        raw = payment_raw.loc[payment_id]
        route, direct_debtor_id = resolve_iban(dict(iban_debtor=raw["iban_debtor"]), lookups)
        if direct_debtor_id is not None:
            debtor_id = direct_debtor_id
        else:
            best = max(cands, key=lambda c: c.score)
            debtor_id = invoice_debtor.get(best.invoice_id)
        payments.append(
            dict(
                payment_id=payment_id,
                debtor_id=debtor_id,
                value_date=pd.Timestamp(raw["value_date"]),
                amount=int(raw["amount"]),
            )
        )

    print(f"Étage C exécuté sur {len(payments)} paiements (période de test)")
    proposals = propose_groups(payments, candidates_by_payment)
    accepted, rejected = resolve_conflicts(proposals)
    print(f"Propositions : {len(proposals)} ({sum(p.origin == 'individual' for p in proposals)} individuelles, "
          f"{sum(p.origin == 'nn_aggregate' for p in proposals)} agrégats n↔n)")
    print(f"Acceptées après résolution de conflits : {len(accepted)} ; rejetées : {len(rejected)}")

    # --- comparaison aux groupes réels (ground_truth) ------------------------
    # Un groupe n↔1 est reconstruit par PLUSIEURS propositions distinctes
    # (chaque paiement partiel est résolu individuellement, jamais regroupé
    # en une seule proposition) : la bonne comparaison n'est donc pas "une
    # proposition == le groupe entier", mais "l'UNION des propositions qui
    # touchent ce groupe reconstitue exactement ses membres".
    ground_truth = tables["ground_truth"]
    test_payment_ids = set(test2["payment_id"])

    n_evaluable = defaultdict(int)
    n_correct = defaultdict(int)
    for group_id, group in ground_truth.groupby("group_id"):
        true_payment_ids = set(group["payment_id"])
        if not true_payment_ids.issubset(test_payment_ids):
            continue  # groupe partiellement hors période de test, hors périmètre de cette mesure
        true_invoice_ids = set(group["invoice_id"])
        case_type = group["case_type"].iloc[0]
        n_evaluable[case_type] += 1

        touching = [
            p for p in accepted
            if set(p.payment_ids) & true_payment_ids or set(p.invoice_ids) & true_invoice_ids
        ]
        union_payments = set().union(*(p.payment_ids for p in touching)) if touching else set()
        union_invoices = set().union(*(p.invoice_ids for p in touching)) if touching else set()
        if union_payments == true_payment_ids and union_invoices == true_invoice_ids:
            n_correct[case_type] += 1

    print("\nReconstruction de groupes par type de cas (test) :")
    for case_type in sorted(n_evaluable):
        total = n_evaluable[case_type]
        correct = n_correct[case_type]
        print(f"  {case_type:<15} {correct:>5}/{total:<5} ({correct / total:.2%})")

    total_evaluable = sum(n_evaluable.values())
    total_correct = sum(n_correct.values())
    print(f"\nTotal : {total_correct}/{total_evaluable} ({total_correct / total_evaluable:.2%})")


if __name__ == "__main__":
    main()
