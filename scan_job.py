"""Nightly screener run for the Render cron job.

Scans the full universe, then appends a row to the Supabase `screener_results`
table. Each run is its own row (identity id + created_at default now()), so the
table is a history; the Next.js screener page reads the most recent row:

    select * from screener_results order by created_at desc limit 1

Run with: python scan_job.py
"""
import argparse
import json
import os
import sys

from screener import ScreenRequest, clean_records, drawdown_sort_key, get_universe, run_screen

TABLE = "screener_results"


def parse_args():
    ap = argparse.ArgumentParser(description="Nightly screener scan -> Supabase")
    ap.add_argument(
        "--max-tickers", type=int, default=None,
        help="cap the universe for a quick test run (default: full universe)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="scan and print, but do not write to Supabase",
    )
    ap.add_argument(
        "--require-accumulation", action="store_true",
        help="turn on the accumulation/base gate (off by default, as in prod)",
    )
    ap.add_argument(
        "--min-score", type=float, default=50.0,
        help="accumulation score threshold, 0-100 (default 50)",
    )
    return ap.parse_args()


def score_summary(results):
    """How many candidates would survive at each accumulation threshold.

    Printed on a dry run so ONE full scan answers 'where should I set the bar?'
    instead of re-scanning once per threshold.
    """
    scores = [r.get("accumulation_score") for r in results]
    have = sorted(s for s in scores if s is not None)
    lines = [
        "",
        f"[scan_job] accumulation scores: {len(have)}/{len(results)} measurable "
        f"({len(results) - len(have)} lacked usable history)",
    ]
    if have:
        mid = have[len(have) // 2]
        lines.append(f"[scan_job]   median {mid:.0f}, min {have[0]:.0f}, max {have[-1]:.0f}")
        for thr in (30, 40, 50, 60, 70, 80, 90):
            n = sum(1 for s in have if s >= thr)
            bar = "#" * min(50, n)
            lines.append(f"[scan_job]   >= {thr:>3}: {n:>4}  {bar}")
    return "\n".join(lines)


def main(args):
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not args.dry_run:
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"
            )
        # Imported here so a missing config fails before we spend 20 minutes
        # scanning. Skipped entirely on a dry run so testing needs no creds.
        from supabase import create_client

    params = ScreenRequest(
        use_all_time_high=True,
        drawdown_min=-30.0,
        fcf_yield_min=3.0,
        require_quiet=False,
        max_tickers=args.max_tickers,   # echoed into params so the row is honest
        # The full thesis: cheap on forward earnings, beaten down, and showing a
        # base that formed after the decline. --require-accumulation only forces
        # it on; the nightly run has it on regardless.
        require_accumulation=True,
        min_accumulation_score=args.min_score,
    )

    tickers = get_universe()
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"[scan_job] universe: {len(tickers)} tickers", flush=True)

    records = run_screen(tickers, params)          # full universe unless --max-tickers
    records.sort(key=drawdown_sort_key)            # most beaten-down first
    results = clean_records(records)               # NaN/inf -> None so jsonb is valid

    if args.dry_run:
        print(
            f"[scan_job] DRY RUN: {len(results)} candidates, "
            f"{len(json.dumps(results))} bytes - nothing written",
            flush=True,
        )
        print(score_summary(results), flush=True)
        # Best-scoring names first, so the dry run shows what the gate WOULD keep.
        top = sorted(results, key=lambda r: r.get("accumulation_score") or -1, reverse=True)
        print("\n[scan_job] top 15 by accumulation score:")
        print(f"  {'ticker':<8}{'score':>6}{'dd%':>8}{'base_rng%':>11}"
              f"{'slope/mo':>10}{'selloff%':>10}{'rebound%':>10}")
        for r in top[:15]:
            def f(k, w=10, d=1):
                v = r.get(k)
                return f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
            print(f"  {r['ticker']:<8}{f('accumulation_score', 6, 0)}{f('drawdown_%', 8)}"
                  f"{f('base_range_%', 11)}{f('base_slope_%_per_mo', 10)}"
                  f"{f('selloff_return_%', 10)}{f('rebound_from_base_low_%', 10)}")
        return

    # An empty screen is almost always thresholds set too tight, not a real
    # "nothing qualifies today". Writing it would make it the newest row and
    # blank the page; not writing leaves yesterday's screen up with an honest
    # timestamp, which is the better degraded state. Exit non-zero so Render
    # surfaces it instead of reporting a green run that changed nothing.
    if not results:
        raise RuntimeError(
            "screen returned 0 candidates - refusing to write an empty row "
            "(loosen min_accumulation_score / fwd_pe_max / fcf_yield_min)"
        )

    client = create_client(url, key)
    # Same shape as the /screen route in main.py. id and created_at are left to
    # their defaults; the service key bypasses RLS on this table.
    resp = client.table(TABLE).insert({
        "count": len(results),
        "params": params.model_dump(),
        "results": results,
    }).execute()

    # An insert that returns no row means nothing landed (RLS, constraint).
    # Fail loudly rather than logging a success the table doesn't back up.
    if not resp.data:
        raise RuntimeError(f"screener insert returned no row: {resp}")

    row_id = resp.data[0]["id"]
    print(
        f"[scan_job] saved {len(results)} candidates to {TABLE} "
        f"(id={row_id}, {len(json.dumps(results))} bytes)",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception as e:
        # Non-zero exit so Render marks the run failed instead of silently "succeeding".
        print(f"[scan_job] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
