"""Nightly screener run for the Render cron job.

Scans the full universe, then appends a row to the Supabase `screener_results`
table. Each run is its own row (identity id + created_at default now()), so the
table is a history; the Next.js screener page reads the most recent row:

    select * from screener_results order by created_at desc limit 1

Run with: python scan_job.py
"""
import json
import os
import sys

from screener import ScreenRequest, clean_records, drawdown_sort_key, get_universe, run_screen

TABLE = "screener_results"


def main():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"
        )

    # Imported here so a missing config fails before we spend 20 minutes scanning.
    from supabase import create_client

    params = ScreenRequest(
        use_all_time_high=True,
        drawdown_min=-30.0,
        fcf_yield_min=3.0,
        require_quiet=False,
    )

    tickers = get_universe()
    print(f"[scan_job] universe: {len(tickers)} tickers", flush=True)

    records = run_screen(tickers, params)          # no max_tickers cap: full universe
    records.sort(key=drawdown_sort_key)            # most beaten-down first
    results = clean_records(records)               # NaN/inf -> None so jsonb is valid

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
        main()
    except Exception as e:
        # Non-zero exit so Render marks the run failed instead of silently "succeeding".
        print(f"[scan_job] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
