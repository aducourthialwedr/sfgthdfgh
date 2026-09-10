"""Tests du journal d'événements et de LedgerState (Phase 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import build_journal  # noqa: E402
from src.generator import GeneratorParams, generate  # noqa: E402
from src.state import LedgerState  # noqa: E402


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    params = GeneratorParams(n_debtors=30, n_assignors=8, n_invoices=1500, n_months=14, seed=321)
    return generate(params)


@pytest.fixture(scope="module")
def journal(tables: dict[str, pd.DataFrame]) -> list:
    return build_journal(tables)


def _replay(journal_events: list, until: pd.Timestamp | None = None) -> LedgerState:
    state = LedgerState()
    for event in journal_events:
        if until is not None and event.timestamp >= until:
            break
        state.apply(event)
    return state


def independent_current_amount(
    invoice_id: str, cutoff: pd.Timestamp, invoice_table: pd.DataFrame, imputation_table: pd.DataFrame
) -> int:
    """Recalcul indépendant (sans passer par LedgerState) pour cross-validation."""
    rows = imputation_table[
        (imputation_table["invoice_id"] == invoice_id) & (imputation_table["updated_at"] < cutoff)
    ]
    if rows.empty:
        return int(invoice_table.loc[invoice_id, "initial_amount"])
    rows = rows.sort_values(["updated_at", "payment_id"])
    last = rows.iloc[-1]
    return 0 if last["status"] == "FULL" else int(last["residual_amount"])


def test_journal_is_sorted_and_deterministic(tables: dict[str, pd.DataFrame]) -> None:
    j1 = build_journal(tables)
    j2 = build_journal(tables)
    assert [(e.timestamp, e.type, e.sort_key) for e in j1] == [
        (e.timestamp, e.type, e.sort_key) for e in j2
    ]
    timestamps = [e.timestamp for e in j1]
    assert timestamps == sorted(timestamps)


def test_journal_covers_all_source_rows(tables: dict[str, pd.DataFrame], journal: list) -> None:
    n_expected = (
        len(tables["invoice"])
        + len(tables["payment"])
        + len(tables["imputation"])
        + len(tables["debtor"])  # PARTY_OPENED (+ PARTY_CLOSED pour certains)
        + len(tables["assignor"])
        + len(tables["agreement"])  # AGREEMENT_CREATED (+ DISABLED pour certains)
        + tables["debtor"]["closed_at"].notna().sum()
        + tables["assignor"]["closed_at"].notna().sum()
        + tables["agreement"]["disabled_at"].notna().sum()
    )
    assert len(journal) == n_expected


def test_apply_rejects_out_of_order_event(journal: list) -> None:
    state = LedgerState()
    state.apply(journal[5])
    out_of_order = journal[0]
    if out_of_order.timestamp >= journal[5].timestamp:
        pytest.skip("premiers événements déjà croissants dans cet échantillon")
    with pytest.raises(ValueError):
        state.apply(out_of_order)


def test_current_amount_matches_independent_recomputation_at_cutoff(
    tables: dict[str, pd.DataFrame], journal: list
) -> None:
    """Le cœur du test anti-fuite : l'état à `cutoff` ne doit refléter que les
    imputations dont `updated_at < cutoff`, jamais celles d'après."""
    invoice_table = tables["invoice"].set_index("invoice_id")
    imputation_table = tables["imputation"]

    all_ts = sorted(e.timestamp for e in journal)
    cutoff = all_ts[len(all_ts) // 2]

    state = _replay(journal, until=cutoff)

    sample_ids = invoice_table.index[:150]
    checked = 0
    for inv_id in sample_ids:
        expected = independent_current_amount(inv_id, cutoff, invoice_table, imputation_table)
        actual = state.current_amount(inv_id, as_of=cutoff) if inv_id in state._invoices else int(
            invoice_table.loc[inv_id, "initial_amount"]
        )
        assert actual == expected, f"{inv_id}: attendu {expected}, obtenu {actual}"
        checked += 1
    assert checked > 0


def test_current_amount_ignores_future_imputations(
    tables: dict[str, pd.DataFrame], journal: list
) -> None:
    """Suppression explicite des événements postérieurs à un cutoff : l'état
    reconstruit ne doit pas changer par rapport à un rejeu tronqué au même point."""
    all_ts = sorted(e.timestamp for e in journal)
    cutoff = all_ts[len(all_ts) // 3]

    truncated_journal = [e for e in journal if e.timestamp < cutoff]
    state_from_full_replay_stopped_early = _replay(journal, until=cutoff)
    state_from_pretruncated_journal = _replay(truncated_journal, until=None)

    invoice_table = tables["invoice"].set_index("invoice_id")
    for inv_id in invoice_table.index[:100]:
        in_full = inv_id in state_from_full_replay_stopped_early._invoices
        in_trunc = inv_id in state_from_pretruncated_journal._invoices
        assert in_full == in_trunc
        if in_full:
            a = state_from_full_replay_stopped_early.current_amount(inv_id, as_of=cutoff)
            b = state_from_pretruncated_journal.current_amount(inv_id, as_of=cutoff)
            assert a == b


def test_current_amount_matches_generator_final_state(
    tables: dict[str, pd.DataFrame], journal: list
) -> None:
    state = _replay(journal)
    end = journal[-1].timestamp + pd.Timedelta(days=1)
    invoice_table = tables["invoice"].set_index("invoice_id")
    mismatches = []
    for inv_id, row in invoice_table.iterrows():
        ledger_amount = state.current_amount(inv_id, as_of=end)
        if ledger_amount != row["current_amount"]:
            mismatches.append((inv_id, ledger_amount, row["current_amount"]))
    assert not mismatches, f"{len(mismatches)} désaccords, ex. {mismatches[:5]}"


def test_open_invoices_reflects_closure(tables: dict[str, pd.DataFrame], journal: list) -> None:
    state = _replay(journal)
    end = journal[-1].timestamp + pd.Timedelta(days=1)
    ground_truth = tables["ground_truth"]
    fully_closed_via_full_status = set(
        ground_truth[ground_truth["status"] == "FULL"]["invoice_id"]
    )
    invoice_table = tables["invoice"].set_index("invoice_id")

    for debtor_id, group in tables["debtor"].groupby("party_id"):
        open_ids = {inv["invoice_id"] for inv in state.open_invoices(debtor_id, as_of=end)}
        for inv_id in open_ids:
            assert invoice_table.loc[inv_id, "current_amount"] > 0
        break  # un débiteur suffit pour ce test de cohérence structurelle


def test_party_is_active_before_open_and_after_close(
    tables: dict[str, pd.DataFrame], journal: list
) -> None:
    closed_debtors = tables["debtor"][tables["debtor"]["closed_at"].notna()]
    if closed_debtors.empty:
        pytest.skip("aucun débiteur fermé dans cet échantillon")
    row = closed_debtors.iloc[0]

    state = LedgerState()
    before_open = row["opened_at"] - pd.Timedelta(days=1)
    for event in journal:
        if event.timestamp >= row["opened_at"]:
            break
        state.apply(event)
    assert state.party_is_active(row["party_id"], as_of=before_open) is False

    state2 = _replay(journal)
    end = journal[-1].timestamp + pd.Timedelta(days=1)
    assert state2.party_is_active(row["party_id"], as_of=end) is False


def test_behavioral_stats_smoke(tables: dict[str, pd.DataFrame], journal: list) -> None:
    state = _replay(journal)
    end = journal[-1].timestamp + pd.Timedelta(days=1)
    debtor_id = tables["debtor"]["party_id"].iloc[0]
    stats = state.behavioral_stats(debtor_id, as_of=end)
    assert stats["debtor_payment_count"] >= 0
    if stats["debtor_payment_count"] > 0:
        assert 0.0 <= stats["debtor_partial_payment_rate"] <= 1.0
        assert 0.0 <= stats["debtor_grouping_rate"] <= 1.0
        assert 0.0 <= stats["debtor_ref_citation_rate"] <= 1.0


def test_as_of_before_clock_raises(journal: list) -> None:
    state = _replay(journal, until=journal[10].timestamp)
    with pytest.raises(ValueError):
        state.current_amount(journal[0].data.get("invoice_id", "INV000001"), as_of=journal[0].timestamp - pd.Timedelta(days=100))
