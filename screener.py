"""Universe screening logic, kept free of FastAPI.

main.py imports this for the /screen route; scan_job.py imports it for the
nightly cron run. Nothing here touches the app object, so importing this
module never spins up a web framework.
"""
import io
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from pydantic import BaseModel


class ScreenRequest(BaseModel):
    # simple two-gate screen: big drawdown + decent cash flow
    drawdown_min: float = -30.0       # keep if price is >= this far below the high (%)
    fcf_yield_min: float = 3.0        # generous bar so quality-growth (VEEV-type) passes (%)
    use_all_time_high: bool = True    # False -> use 52-week high
    max_tickers: int | None = None    # cap the universe for quick testing
    # "quiet base" gate: little price movement over the last ~month (consolidation)
    require_quiet: bool = False       # off by default; turn on to hunt non-movers
    quiet_lookback_days: int = 21     # ~1 trading month
    max_range_pct: float = 12.0       # last-month high-to-low range must be <= this (%)
    max_drift_pct: float = 6.0        # net change over the window must be within +/- this (%)


# ------------------------- shared helper -------------------------
def _num(v):
    """Make a value JSON-safe: numpy -> float, NaN/inf -> None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def drawdown_sort_key(r):
    """Sort ascending = most beaten-down first, with None pushed to the end.

    A missing drawdown would raise TypeError against a float; this runs
    unattended overnight, so it must not crash on a single bad record.
    """
    dd = r.get("drawdown_%")
    return float("inf") if dd is None else dd


def clean_records(records):
    """JSON-safe copy of the screen output (NaN/inf -> None)."""
    return [
        {k: (v if isinstance(v, (str, bool)) or v is None else _num(v))
         for k, v in r.items()}
        for r in records
    ]


def get_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
    ).text
    # pandas 2.2+ needs a file-like object; a raw HTML string makes read_html
    # try to open it as a path and dump the whole page into the error.
    buf = io.StringIO(html)

    table = None
    # 1) preferred: the constituents table by id
    try:
        buf.seek(0)
        table = pd.read_html(buf, attrs={"id": "constituents"})[0]
    except Exception:
        table = None
    # 2) fallback: whichever parsed table actually has a "Symbol" column    
    if table is None or "Symbol" not in table.columns:
        buf.seek(0)
        for cand in pd.read_html(buf):
            if "Symbol" in cand.columns:
                table = cand
                break

    if table is None or "Symbol" not in table.columns:
        raise RuntimeError("S&P 500 table not found on Wikipedia page")

    return table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def run_screen(tickers, p):
    records = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info

            fcf   = info.get("freeCashflow")
            mcap  = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if fcf is None or mcap is None or price is None or mcap <= 0:
                continue

            # gate 1: decent cash flow (generous so premium names pass)
            fcf_yield = fcf / mcap * 100
            if fcf_yield < p.fcf_yield_min:
                continue

            # gate 2: big drawdown from the high
            if p.use_all_time_high:
                # dropna: yfinance appends today's not-yet-closed session as a
                # NaN row. iloc[-1] would then be NaN, and since NaN > x is
                # always False the drawdown gate below would stop filtering.
                hist = tk.history(period="max")["Close"].dropna()
                if hist.empty:
                    continue
                high = float(hist.max())
                ref_price = float(hist.iloc[-1])   # same adjusted series as the high -> consistent
            else:
                high = info.get("fiftyTwoWeekHigh")
                if high is None or high <= 0:
                    continue
                ref_price = float(price)

            drawdown = (ref_price / high - 1) * 100
            if drawdown > p.drawdown_min:
                continue

            # gate 3 (optional): "quiet base" — little movement over ~1 month.
            # range_pct = last-month high-to-low spread; drift_pct = net change.
            # A true consolidation is BOTH a tight range AND a flat drift.
            range_pct = None
            drift_pct = None
            try:
                if p.use_all_time_high:
                    recent = hist                                # already dropna'd above
                else:
                    recent = tk.history(period="3mo")["Close"].dropna()
                window = recent.tail(p.quiet_lookback_days)
                if len(window) >= 5:
                    w_hi = float(window.max())
                    w_lo = float(window.min())
                    w_mean = float(window.mean())
                    w_first = float(window.iloc[0])
                    w_last = float(window.iloc[-1])
                    if w_mean > 0:
                        range_pct = (w_hi - w_lo) / w_mean * 100
                    if w_first > 0:
                        drift_pct = (w_last / w_first - 1) * 100
            except Exception:
                pass

            if p.require_quiet:
                if range_pct is None or drift_pct is None:
                    continue
                if range_pct > p.max_range_pct:            # too wide a range -> still swinging
                    continue
                if abs(drift_pct) > p.max_drift_pct:       # trending, not flat
                    continue

            records.append({
                "ticker": t, "sector": info.get("sector"),
                "price": price, "high": high,
                "drawdown_%": drawdown, "fcf_yield_%": fcf_yield,
                "range_1m_%": range_pct,                   # last-month high-to-low spread
                "drift_1m_%": drift_pct,                   # last-month net change
                "pe": info.get("trailingPE"),        # eyeball only
                "fwd_pe": info.get("forwardPE"),     # eyeball only
            })
        except Exception:
            continue
        finally:
            time.sleep(0.1)      # runs on EVERY path (continue/skip/error) -> no Yahoo hammering
    return records
