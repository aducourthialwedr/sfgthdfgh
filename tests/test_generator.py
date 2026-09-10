"""Tests du générateur de données synthétiques (Phase 0)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import GeneratorParams, generate  # noqa: E402


@pytest.fixture(scope="module")
def small_tables() -> dict[str, pd.DataFrame]:
    params = GeneratorParams(n_debtors=25, n_assignors=6, n_invoices=1200, n_months=14, seed=123)
    return generate(params)


def test_determinism() -> None:
    params = GeneratorParams(n_debtors=20, n_assignors=5, n_invoices=500, n_months=14, seed=7)
    t1 = generate(params)
    t2 = generate(params)
    for name in ("payment", "invoice", "imputation", "assignor", "debtor", "agreement"):
        pd.testing.assert_frame_equal(t1[name], t2[name])


def test_different_seed_gives_different_data() -> None:
    p1 = GeneratorParams(n_debtors=20, n_assignors=5, n_invoices=500, n_months=14, seed=1)
    p2 = GeneratorParams(n_debtors=20, n_assignors=5, n_invoices=500, n_months=14, seed=2)
    t1 = generate(p1)
    t2 = generate(p2)
    assert not t1["payment"]["label"].equals(t2["payment"]["label"])


def test_amounts_are_integers(small_tables: dict[str, pd.DataFrame]) -> None:
    assert small_tables["payment"]["amount"].dtype == np.int64
    assert small_tables["invoice"]["initial_amount"].dtype == np.int64
    assert small_tables["invoice"]["current_amount"].dtype == np.int64
    if not small_tables["imputation"].empty:
        assert small_tables["imputation"]["residual_amount"].dtype == np.int64


def test_referential_integrity(small_tables: dict[str, pd.DataFrame]) -> None:
    debtor_ids = set(small_tables["debtor"]["party_id"])
    assignor_ids = set(small_tables["assignor"]["party_id"])
    agreement_ids = set(small_tables["agreement"]["agreement_id"])
    invoice_ids = set(small_tables["invoice"]["invoice_id"])
    payment_ids = set(small_tables["payment"]["payment_id"])

    assert set(small_tables["invoice"]["debtor_id"]).issubset(debtor_ids)
    assert set(small_tables["invoice"]["agreement_id"]).issubset(agreement_ids)
    assert set(small_tables["agreement"]["debtor_id"]).issubset(debtor_ids)
    assert set(small_tables["agreement"]["client_id"]).issubset(assignor_ids)

    imputation = small_tables["imputation"]
    assert set(imputation["payment_id"]).issubset(payment_ids)
    assert set(imputation["invoice_id"]).issubset(invoice_ids)

    ground_truth = small_tables["ground_truth"]
    assert set(ground_truth["payment_id"]).issubset(payment_ids)
    assert set(ground_truth["invoice_id"]).issubset(invoice_ids)


def test_ground_truth_is_subset_of_imputation(small_tables: dict[str, pd.DataFrame]) -> None:
    imputation_pairs = set(
        zip(small_tables["imputation"]["payment_id"], small_tables["imputation"]["invoice_id"])
    )
    gt_pairs = set(
        zip(small_tables["ground_truth"]["payment_id"], small_tables["ground_truth"]["invoice_id"])
    )
    assert gt_pairs == imputation_pairs


def test_no_invoice_paid_twice_without_being_flagged(small_tables: dict[str, pd.DataFrame]) -> None:
    # Une facture peut apparaître plusieurs fois dans imputation (paiements
    # partiels / n-n), mais jamais dans deux groupes de cas différents.
    gt = small_tables["ground_truth"]
    groups_per_invoice = gt.groupby("invoice_id")["group_id"].nunique()
    assert (groups_per_invoice == 1).all()


def test_case_type_proportions_roughly_match_weights(small_tables: dict[str, pd.DataFrame]) -> None:
    # Les poids du générateur sont appliqués par plan de règlement (groupe),
    # pas par facture individuelle : un plan "1-n" consomme plusieurs factures
    # d'un coup, donc la proportion se mesure au niveau du group_id.
    gt = small_tables["ground_truth"]
    groups_with_case = gt.drop_duplicates("group_id")["case_type"]
    realized = groups_with_case.value_counts(normalize=True)
    from src.generator import DEFAULT_CASE_WEIGHTS

    for case_type in ["1-1_clean", "1-1_noisy", "1-n", "n-1"]:
        target = DEFAULT_CASE_WEIGHTS[case_type]
        assert abs(realized.get(case_type, 0.0) - target) < 0.10


def test_orphan_payments_exist(small_tables: dict[str, pd.DataFrame]) -> None:
    matched = set(small_tables["ground_truth"]["payment_id"])
    all_payments = set(small_tables["payment"]["payment_id"])
    orphans = all_payments - matched
    assert len(orphans) > 0


def test_label_noise_variety(small_tables: dict[str, pd.DataFrame]) -> None:
    labels = small_tables["payment"]["label"]
    assert (labels == "").any(), "aucun libellé vide généré"
    non_empty = labels[labels != ""]
    assert (non_empty.str.contains(r"[A-Za-z]")).any()
    numeric_only = non_empty[non_empty.str.fullmatch(r"\d+")]
    assert len(numeric_only) > 0, "aucun libellé purement numérique généré"


def test_iban_checksum_valid(small_tables: dict[str, pd.DataFrame]) -> None:
    def mod97_ok(iban: str) -> bool:
        rearranged = iban[4:] + iban[:4]
        numeric = "".join(str(int(ch, 36)) for ch in rearranged)
        return int(numeric) % 97 == 1

    sample = small_tables["debtor"]["iban"].head(10)
    assert all(mod97_ok(i) for i in sample)


def test_current_amount_as_of_end_is_consistent(small_tables: dict[str, pd.DataFrame]) -> None:
    """Par groupe de règlement, la somme des factures soldées doit être couverte
    (à l'écart près pour escompte/frais/retenue) par la somme des paiements liés."""
    invoice = small_tables["invoice"].set_index("invoice_id")
    payment = small_tables["payment"].set_index("payment_id")
    gt = small_tables["ground_truth"]

    for group_id, rows in gt.groupby("group_id"):
        invoice_ids = rows["invoice_id"].unique()
        payment_ids = rows["payment_id"].unique()
        fully_settled = invoice.loc[invoice_ids, "current_amount"].eq(0).all()
        if not fully_settled:
            continue  # facture partiellement soldée en fin de période, hors périmètre
        total_initial = invoice.loc[invoice_ids, "initial_amount"].sum()
        total_paid = payment.loc[payment_ids, "amount"].sum()
        gap = total_initial - total_paid
        assert gap >= 0, f"groupe {group_id} : paiements ({total_paid}) > factures ({total_initial})"
        assert gap <= total_initial * 0.10 + 4000 * len(invoice_ids), f"groupe {group_id} : écart {gap} trop grand"


def test_partial_status_before_last_split(small_tables: dict[str, pd.DataFrame]) -> None:
    gt = small_tables["ground_truth"]
    n1 = gt[gt["case_type"] == "n-1"]
    if n1.empty:
        pytest.skip("aucun cas n-1 généré dans cet échantillon")
    for group_id, rows in n1.groupby("group_id"):
        statuses = rows["status"].tolist()
        assert statuses[-1] == "FULL" or "FULL" in statuses
        assert statuses.count("FULL") == 1
