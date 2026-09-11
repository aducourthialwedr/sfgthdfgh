"""Benchmark à volume réel (débiteurs, cédants, factures, paiements) : mesure
le temps réel de génération, construction du journal, et blocking+
featurisation — sans se contenter d'extrapoler depuis un volume plus petit.

Le rejeu (blocking + featurize) est traité par lots et NE CONSERVE PAS
toutes les lignes de features en mémoire : à volume réel, accumuler
`payments × candidats_par_paiement` dictionnaires de features avant de
construire un seul DataFrame risquerait une explosion mémoire (pas
seulement un problème de vitesse) — chaque lot est mesuré puis jeté.

Usage :
    python scripts/benchmark_scale.py --n-debtors 2000000 --n-assignors 65000 --n-invoices 1700000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking import build_static_lookups, generate_candidates  # noqa: E402
from src.events import build_journal  # noqa: E402
from src.features import featurize  # noqa: E402
from src.generator import GeneratorParams, HIGH_GROUPING_CASE_WEIGHTS, generate  # noqa: E402
from src.state import LedgerState  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-debtors", type=int, default=2_000_000)
    parser.add_argument("--n-assignors", type=int, default=65_000)
    parser.add_argument("--n-invoices", type=int, default=1_700_000)
    parser.add_argument("--n-months", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=20_000, help="paiements par lot mesuré")
    parser.add_argument("--out-dir", type=str, default=None, help="si fourni, écrit aussi les tables sources en Parquet")
    args = parser.parse_args()

    log(f"Paramètres : n_debtors={args.n_debtors:,} n_assignors={args.n_assignors:,} "
        f"n_invoices={args.n_invoices:,} (pondération HIGH_GROUPING pour approcher le ratio réel factures/paiements)")

    params = GeneratorParams(
        n_debtors=args.n_debtors,
        n_assignors=args.n_assignors,
        n_invoices=args.n_invoices,
        n_months=args.n_months,
        seed=args.seed,
        case_weights=dict(HIGH_GROUPING_CASE_WEIGHTS),
    )

    t0 = time.time()
    log("Génération : parties, agreements, factures, paiements, imputations...")
    tables = generate(params, verbose=True)
    t_gen = time.time() - t0
    log(f"Génération terminée en {t_gen:.1f}s ({t_gen / 60:.1f} min)")
    for name in ["debtor", "assignor", "agreement", "invoice", "payment", "imputation"]:
        log(f"  {name:<10} {len(tables[name]):>12,} lignes")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for name, df in tables.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        log(f"Tables écrites dans {out_dir} en {time.time() - t0:.1f}s")

    t0 = time.time()
    log("Construction du journal d'événements...")
    journal = build_journal(tables)
    t_journal = time.time() - t0
    log(f"Journal : {len(journal):,} événements en {t_journal:.1f}s")

    t0 = time.time()
    log("Construction des référentiels statiques (index K4 inclus)...")
    lookups = build_static_lookups(tables)
    t_lookups = time.time() - t0
    log(f"Référentiels construits en {t_lookups:.1f}s "
        f"({len(lookups['name_index']):,} tokens indexés pour K4)")

    log("Rejeu (blocking + featurisation), traité par lots pour ne pas tout garder en mémoire...")
    state = LedgerState()
    n_payments = 0
    n_candidate_rows = 0
    t_replay_start = time.time()
    t_chunk_start = time.time()
    chunk_payments = 0
    chunk_rows = 0

    for event in journal:
        if event.type == "PAYMENT_RECEIVED":
            payment = event.data
            as_of = event.timestamp
            candidate_ids = generate_candidates(payment, state, as_of, lookups)
            for invoice_id in candidate_ids:
                invoice = state.get_invoice(invoice_id, as_of=as_of)
                featurize(payment, invoice, state, as_of, lookups)  # calculé, jeté (mesure de débit)
                chunk_rows += 1
            n_payments += 1
            chunk_payments += 1

            if chunk_payments >= args.chunk_size:
                elapsed = time.time() - t_chunk_start
                total_elapsed = time.time() - t_replay_start
                log(
                    f"  {n_payments:>10,} paiements traités | lot: {chunk_payments:,} paiements, "
                    f"{chunk_rows:,} candidats en {elapsed:.1f}s ({1000 * elapsed / chunk_payments:.2f} ms/paiement) | "
                    f"cumulé: {total_elapsed / 60:.1f} min"
                )
                n_candidate_rows += chunk_rows
                chunk_payments = 0
                chunk_rows = 0
                t_chunk_start = time.time()
        state.apply(event)

    n_candidate_rows += chunk_rows
    t_replay = time.time() - t_replay_start

    log("=" * 70)
    log("RÉSULTATS")
    log("=" * 70)
    log(f"Génération             : {t_gen:8.1f}s  ({t_gen / 60:.1f} min)")
    log(f"Journal                : {t_journal:8.1f}s")
    log(f"Référentiels statiques : {t_lookups:8.1f}s")
    log(f"Blocking + featurize   : {t_replay:8.1f}s  ({t_replay / 60:.1f} min)")
    log(f"Total                  : {(t_gen + t_journal + t_lookups + t_replay) / 60:.1f} min")
    log(f"Paiements traités      : {n_payments:,}")
    log(f"Lignes candidates      : {n_candidate_rows:,}")
    if n_payments:
        log(f"Débit                  : {1000 * t_replay / n_payments:.3f} ms/paiement")


if __name__ == "__main__":
    main()
