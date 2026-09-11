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
    return ap.parse_args()


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
        print(json.dumps(results[:3], indent=2))
        return

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
