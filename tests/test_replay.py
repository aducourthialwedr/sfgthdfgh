"""Tests du rejeu et de la construction du dataset (Phase 4, §8.3).

Le point critique de cette phase : aucune feature ne doit jamais refléter
un événement postérieur à `t`. Deux angles de test complémentaires :
1. structurel — retirer du journal tout ce qui suit `t` ne change rien.
2. sémantique — une facture qui sera soldée plus tard doit quand même
   apparaître "ouverte" dans les features d'un paiement antérieur.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking import build_static_lookups, generate_candidates  # noqa: E402
from src.events import build_journal  # noqa: E402
from src.features import featurize  # noqa: E402
from src.generator import GeneratorParams, generate  # noqa: E402
from src.replay import build_training_rows, write_partitioned_dataset  # noqa: E402
from src.state import LedgerState  # noqa: E402


@pytest.fixture(scope="module")
def small_tables() -> dict[str, pd.DataFrame]:
    params = GeneratorParams(n_debtors=25, n_assignors=6, n_invoices=1200, n_months=14, seed=2024)
    return generate(params)


@pytest.fixture(scope="module")
def small_journal(small_tables: dict[str, pd.DataFrame]) -> list:
    return build_journal(small_tables)


def test_dataset_has_expected_columns_and_types(small_tables: dict[str, pd.DataFrame]) -> None:
    dataset = build_training_rows(small_tables)
    assert not dataset.empty
    for col in ["payment_id", "invoice_id", "t", "label"]:
        assert col in dataset.columns
    assert dataset["label"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(dataset["t"])
    # au moins une paire positive et une négative
    assert dataset["label"].any()
    assert (~dataset["label"]).any()


def test_no_row_without_a_true_pair_is_mislabeled(small_tables: dict[str, pd.DataFrame]) -> None:
    dataset = build_training_rows(small_tables)
    true_pairs = set(zip(small_tables["imputation"]["payment_id"], small_tables["imputation"]["invoice_id"]))
    for row in dataset.itertuples(index=False):
        expected = (row.payment_id, row.invoice_id) in true_pairs
        assert row.label == expected


def test_write_partitioned_dataset_splits_by_month(small_tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    dataset = build_training_rows(small_tables)
    out_dir = tmp_path / "dataset"
    write_partitioned_dataset(dataset, out_dir)

    files = sorted(out_dir.glob("month=*.parquet"))
    assert len(files) >= 2

    reloaded = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert len(reloaded) == len(dataset)
    for f in files:
        part = pd.read_parquet(f)
        months = pd.to_datetime(part["t"]).dt.to_period("M").astype(str).unique()
        assert len(months) == 1
        assert months[0] in f.name


# ---------------------------------------------------------------------------
# Anti-fuite
# ---------------------------------------------------------------------------


def _lookups_and_state(tables):
    lookups = build_static_lookups(tables)
    return lookups


def test_removing_future_events_does_not_change_features(
    small_tables: dict[str, pd.DataFrame], small_journal: list
) -> None:
    """Retire du journal tous les événements postérieurs à `t` : les
    candidats et features d'un paiement à `t` doivent être identiques."""
    lookups = _lookups_and_state(small_tables)

    # Repère un paiement avec au moins 2 candidats, ni trop tôt ni trop tard.
    state = LedgerState()
    target_index = None
    target_event = None
    target_candidates = None
    target_features = None
    for i, event in enumerate(small_journal):
        if event.type == "PAYMENT_RECEIVED" and 300 < i < len(small_journal) - 300:
            as_of = event.timestamp
            candidates = generate_candidates(
                event.data, state, as_of,
                lookups["debtor_by_iban"], lookups["assignor_by_iban"], lookups["debtor_name_tokens"],
            )
            if len(candidates) >= 2:
                target_index = i
                target_event = event
                target_candidates = candidates
                target_features = [
                    featurize(
                        event.data, state.get_invoice(inv_id, as_of=as_of), state, as_of,
                        lookups["debtor_by_iban"], lookups["assignor_by_iban"], lookups["debtor_name_tokens"],
                    )
                    for inv_id in candidates
                ]
                break
        state.apply(event)
    assert target_index is not None, "aucun paiement avec >=2 candidats trouvé dans l'échantillon"

    # Rejeu à partir d'un journal explicitement tronqué : tout ce qui a un
    # timestamp strictement postérieur à `t` est supprimé de la liste.
    cutoff = target_event.timestamp
    truncated_journal = [e for e in small_journal if e.timestamp <= cutoff]
    idx_in_truncated = truncated_journal.index(target_event)

    state2 = LedgerState()
    for event in truncated_journal[:idx_in_truncated]:
        state2.apply(event)

    as_of = target_event.timestamp
    candidates2 = generate_candidates(
        target_event.data, state2, as_of,
        lookups["debtor_by_iban"], lookups["assignor_by_iban"], lookups["debtor_name_tokens"],
    )
    features2 = [
        featurize(
            target_event.data, state2.get_invoice(inv_id, as_of=as_of), state2, as_of,
            lookups["debtor_by_iban"], lookups["assignor_by_iban"], lookups["debtor_name_tokens"],
        )
        for inv_id in candidates2
    ]

    assert candidates2 == target_candidates
    assert features2 == target_features


def test_amount_features_use_state_not_final_invoice_balance(
    small_tables: dict[str, pd.DataFrame], small_journal: list
) -> None:
    """Trouve une facture encore ouverte à `t` mais soldée plus tard dans
    l'historique (payée par un paiement ultérieur) : les features calculées
    à `t` doivent refléter le solde encore ouvert, pas le solde final (0)."""
    invoice_table = small_tables["invoice"].set_index("invoice_id")
    imputation_table = small_tables["imputation"].sort_values("updated_at")
    lookups = build_static_lookups(small_tables)

    # Facture soldée en plusieurs imputations (n-1) : solde encore ouvert
    # juste après la première imputation partielle.
    counts = imputation_table.groupby("invoice_id").size()
    multi_imputation_invoices = set(counts[counts >= 2].index)
    assert multi_imputation_invoices, "aucune facture à imputations multiples dans cet échantillon"

    first_rows = imputation_table[imputation_table["invoice_id"].isin(multi_imputation_invoices)]
    first_partial = first_rows[first_rows["status"] == "PARTIAL"].iloc[0]
    target_invoice_id = first_partial["invoice_id"]
    cutoff = pd.Timestamp(first_partial["updated_at"]) + pd.Timedelta(seconds=1)

    assert invoice_table.loc[target_invoice_id, "current_amount"] == 0  # soldée in fine

    state = LedgerState()
    for event in small_journal:
        if event.timestamp >= cutoff:
            break
        state.apply(event)

    live_amount = state.current_amount(target_invoice_id, as_of=cutoff)
    assert live_amount > 0, "la facture devrait encore être ouverte à ce cutoff"

    # Featurize un paiement fictif contre cette facture encore ouverte : le
    # amount_diff_abs doit être calculé contre `live_amount`, jamais contre 0.
    fake_payment = dict(
        payment_id="PMT_TEST_LEAK",
        value_date=cutoff,
        amount=live_amount,
        currency="EUR",
        iban_debtor="IBAN_TEST",
        label="TEST",
        channel="SEPA",
        payment_type="VIREMENT",
        bankroll_code="STANDARD",
    )
    invoice = state.get_invoice(target_invoice_id, as_of=cutoff)
    feats = featurize(
        fake_payment, invoice, state, cutoff,
        lookups["debtor_by_iban"], lookups["assignor_by_iban"], lookups["debtor_name_tokens"],
    )
    assert feats["amount_diff_abs"] == 0
    assert feats["amount_exact_match"] is True
