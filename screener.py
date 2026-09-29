"""Universe screening logic, kept free of FastAPI.

main.py imports this for the /screen route; scan_job.py imports it for the
nightly cron run. Nothing here touches the app object, so importing this
module never spins up a web framework.

The screen is a funnel:

    S&P 500
      -> gate 1: decent free cash flow yield      (is the business sound?)
      -> gate 2: large drawdown from the high     (has it been sold off?)
      -> gate 3: optional "quiet base"            (legacy, simple sideways test)
      -> gate 4: optional accumulation base       (has the selling exhausted?)

Gate 4 is the Wyckoff-flavoured part: it compares a recent "base" window
against the "selloff" window immediately preceding it. Everything is measured
as base-vs-selloff rather than in absolute terms, because the thesis is
specifically *stabilisation after a decline*, not "down a lot at some point in
the past and flat now".
"""
import io
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from pydantic import BaseModel

TRADING_DAYS_PER_MONTH = 21   # slopes are normalised to %/month for readability


class ScreenRequest(BaseModel):
    # simple two-gate screen: big drawdown + decent cash flow
    drawdown_min: float = -30.0       # keep if price is >= this far below the high (%)
    fcf_yield_min: float = 3.0        # generous bar so quality-growth (VEEV-type) passes (%)
    # Valuation gate. None turns it off entirely (the old behaviour, where pe /
    # fwd_pe were carried for display only). When set, a name must have a
    # POSITIVE forward P/E at or below this: unknown or negative forward
    # earnings can't be called cheap, so those are dropped rather than assumed.
    fwd_pe_max: float | None = 20.0
    use_all_time_high: bool = True    # False -> use 52-week high
    max_tickers: int | None = None    # cap the universe for quick testing
    # "quiet base" gate: little price movement over the last ~month (consolidation)
    require_quiet: bool = False       # off by default; turn on to hunt non-movers
    quiet_lookback_days: int = 21     # ~1 trading month
    max_range_pct: float = 12.0       # last-month high-to-low range must be <= this (%)
    max_drift_pct: float = 6.0        # net change over the window must be within +/- this (%)

    # ---------------- accumulation / base-forming gate ----------------
    # Diagnostics below are ALWAYS computed for anything clearing gates 1-2, so
    # you can eyeball why a name scored what it did. They only filter when
    # require_accumulation is on.
    require_accumulation: bool = False
    min_accumulation_score: float = 50.0   # out of 100; deliberately loose
    # MANDATORY by default: the base only counts if it followed a real decline.
    # Scored alone this was too weak -- names consolidating after a *rally* were
    # clearing 70/100 purely because they sit far below a years-old high.
    require_preceding_decline: bool = True

    base_window_days: int = 20        # the consolidation window we measure (~1 month)
    base_min_days: int = 10           # MANDATORY floor: ~2 weeks of base, not a 3-day pause
    selloff_window_days: int = 40     # the decline window immediately before the base
    selloff_min_days: int = 15        # below this the base-vs-selloff comparison is noise

    # Scoring thresholds. Loose on purpose: better to hand back a longer list to
    # chart-check by eye than to silently filter out every real setup.
    max_base_range_pct: float = 20.0          # base high-to-low spread (%)
    max_base_decline_pct_per_mo: float = 4.0  # MANDATORY: steeper decline = still bleeding
    max_vol_compression: float = 1.10         # base vol / selloff vol; < 1 means calming
    max_support_spread_pct: float = 8.0       # how tightly the lowest lows cluster (%)
    max_low_undercut_pct: float = 4.0         # how far late-base lows may undercut early ones
    max_rebound_pct: float = 30.0             # above this it's a V-shaped bounce, not a base
    min_selloff_decline_pct: float = 8.0      # the preceding window must actually have fallen


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
    """JSON-safe copy of the screen output (NaN/inf -> None).

    int passes through untouched so counts like base_days stay 10, not 10.0.
    bool is listed first because bool is a subclass of int.
    """
    return [
        {k: (v if isinstance(v, (str, bool, int)) or v is None else _num(v))
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


# ------------------------- technical helpers -------------------------
# Each returns None rather than raising when there isn't enough data. The whole
# module runs unattended, and "unknown" must stay distinguishable from "zero".

def _slope_pct_per_month(close):
    """Least-squares trend of price over the window, as % of mean price per month.

    This is the "is the downtrend flattening?" measurement. Normalising by the
    mean makes it comparable across share prices, and expressing it per month
    makes a 20-day base comparable to a 40-day selloff despite the differing
    window lengths. Negative = still sliding, ~0 = flat, positive = drifting up.
    """
    y = np.asarray(close, dtype=float)
    if len(y) < 3:
        return None
    mean = float(y.mean())
    if mean <= 0:
        return None
    x = np.arange(len(y), dtype=float)
    slope_per_day = float(np.polyfit(x, y, 1)[0])
    return slope_per_day / mean * 100 * TRADING_DAYS_PER_MONTH


def _daily_vol_pct(close):
    """Stdev of daily returns, in %. Used as a ratio, so the scale is arbitrary."""
    r = pd.Series(close, dtype=float).pct_change().dropna()
    if len(r) < 3:
        return None
    sd = float(r.std())
    return None if not np.isfinite(sd) else sd * 100


def _range_pct(close):
    """High-to-low spread across the window as % of its mean price."""
    y = np.asarray(close, dtype=float)
    if len(y) < 2:
        return None
    mean = float(y.mean())
    if mean <= 0:
        return None
    return (float(y.max()) - float(y.min())) / mean * 100


def _drift_pct(close):
    """Net first-to-last change across the window (%)."""
    y = np.asarray(close, dtype=float)
    if len(y) < 2 or y[0] <= 0:
        return None
    return (float(y[-1]) / float(y[0]) - 1) * 100


def _support_spread_pct(lows):
    """How tightly the lowest lows cluster, as % of their mean.

    A base should show price repeatedly finding buyers around the same area.
    Taking the cheapest fifth of the window's daily lows (min 3 bars) and
    measuring their spread approximates "multiple touches of one support zone"
    without trying to detect literal pivot points, which would be fragile.
    Small = a shelf; large = still stair-stepping down.
    """
    y = np.sort(np.asarray(lows, dtype=float))
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return None
    k = max(3, len(y) // 5)
    zone = y[:k]
    mean = float(zone.mean())
    if mean <= 0:
        return None
    return (float(zone.max()) - float(zone.min())) / mean * 100


def _low_undercut_pct(lows):
    """Late-base low vs early-base low (%). Negative = still making new lows.

    Higher lows are a confidence boost, not a requirement, so this is scored
    against a tolerance rather than demanded outright.
    """
    y = np.asarray(lows, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 6:
        return None
    half = len(y) // 2
    early = float(y[:half].min())
    late = float(y[half:].min())
    if early <= 0:
        return None
    return (late / early - 1) * 100


def _down_day_volume(df):
    """Mean volume on down days. None when volume data is missing or unusable.

    Volume from Yahoo is noisy, so this feeds a single scoring component rather
    than a hard gate.
    """
    if "Volume" not in df or "Close" not in df:
        return None
    ret = df["Close"].pct_change()
    vol = df["Volume"].where(np.isfinite(df["Volume"]))
    down = vol[ret < 0].dropna()
    down = down[down > 0]
    if len(down) < 3:
        return None
    return float(down.mean())


def compute_base_metrics(df, p):
    """Split recent history into [selloff | base] and measure the transition.

    Returns a dict with every diagnostic key always present (None where it
    couldn't be computed), so the output shape is stable whether or not the
    accumulation gate is switched on. Never raises.
    """
    out = {
        "base_days": None,
        "base_range_%": None,
        "base_drift_%": None,
        "base_slope_%_per_mo": None,
        "selloff_slope_%_per_mo": None,
        "selloff_return_%": None,
        "vol_compression": None,
        "support_spread_%": None,
        "low_undercut_%": None,
        "rebound_from_base_low_%": None,
        "down_volume_change_%": None,
        "accumulation_score": None,
    }
    try:
        n = len(df)
        if n < p.base_min_days:
            return out

        # The base is the most recent slice; the selloff is what came directly
        # before it. Adjacency is the point -- it's what makes "the base follows
        # the decline" a measurable claim instead of two unrelated observations.
        base = df.tail(p.base_window_days)
        base_days = len(base)
        out["base_days"] = base_days
        if base_days < p.base_min_days:
            return out

        base_close = base["Close"]
        base_low = base["Low"] if "Low" in base else base_close

        out["base_range_%"] = _range_pct(base_close)
        out["base_drift_%"] = _drift_pct(base_close)
        out["base_slope_%_per_mo"] = _slope_pct_per_month(base_close)
        out["support_spread_%"] = _support_spread_pct(base_low)
        out["low_undercut_%"] = _low_undercut_pct(base_low)

        # How far price has already travelled off the base low. A big number
        # means the move has happened -- a V-shaped bounce, not a base we're
        # still early in.
        base_low_px = float(np.nanmin(np.asarray(base_low, dtype=float)))
        last_px = float(base_close.iloc[-1])
        if base_low_px > 0:
            out["rebound_from_base_low_%"] = (last_px / base_low_px - 1) * 100

        # ---- the preceding decline ----
        selloff = df.iloc[max(0, n - base_days - p.selloff_window_days): n - base_days]
        if len(selloff) >= p.selloff_min_days:
            sell_close = selloff["Close"]
            out["selloff_slope_%_per_mo"] = _slope_pct_per_month(sell_close)
            out["selloff_return_%"] = _drift_pct(sell_close)

            base_vol = _daily_vol_pct(base_close)
            sell_vol = _daily_vol_pct(sell_close)
            if base_vol is not None and sell_vol and sell_vol > 0:
                # < 1 means the base is calmer than the decline that preceded it,
                # which is the volatility signature of absorption.
                out["vol_compression"] = base_vol / sell_vol

            base_dv = _down_day_volume(base)
            sell_dv = _down_day_volume(selloff)
            if base_dv is not None and sell_dv:
                # Negative = people are hitting the bid less hard than they were.
                out["down_volume_change_%"] = (base_dv / sell_dv - 1) * 100

        out["accumulation_score"] = score_accumulation(out, p)
        return out
    except Exception:
        return out


def score_accumulation(m, p):
    """Weighted 0-100 score from the base metrics. None if nothing was measurable.

    Scored rather than all-or-nothing on purpose: real bases rarely satisfy
    every textbook condition at once, and requiring that would return an empty
    list most nights. Components that couldn't be measured simply score 0 --
    they never award points on missing data.
    """
    parts = [
        # (points, condition) -- condition may be None when unmeasurable
        (15, m["base_days"] is not None and m["base_days"] >= p.base_min_days),
        (15, m["base_range_%"] is not None and m["base_range_%"] <= p.max_base_range_pct),
        (15, m["vol_compression"] is not None and m["vol_compression"] <= p.max_vol_compression),
        # Flattening: the base must be less negative than the decline before it,
        # and not itself a steep slide. Sideways or mildly up both qualify.
        (20, (
            m["base_slope_%_per_mo"] is not None
            and m["selloff_slope_%_per_mo"] is not None
            and m["base_slope_%_per_mo"] > m["selloff_slope_%_per_mo"]
            and m["base_slope_%_per_mo"] >= -p.max_base_decline_pct_per_mo
        )),
        (15, (
            m["support_spread_%"] is not None
            and m["support_spread_%"] <= p.max_support_spread_pct
            and (m["low_undercut_%"] is None
                 or m["low_undercut_%"] >= -p.max_low_undercut_pct)
        )),
        (10, m["down_volume_change_%"] is not None and m["down_volume_change_%"] < 0),
        # There must actually have been a decline for this to be a base after one.
        (10, (
            m["selloff_return_%"] is not None
            and m["selloff_return_%"] <= -p.min_selloff_decline_pct
        )),
    ]
    score = float(sum(pts for pts, cond in parts if cond))

    # V-shape penalty: a sharp crash that immediately ripped back up is not
    # accumulation, however tidy the last few bars look.
    reb = m["rebound_from_base_low_%"]
    if reb is not None and reb > p.max_rebound_pct:
        score *= 0.5
    return score


def passes_accumulation(m, p):
    """Mandatory conditions for the accumulation gate.

    Only three things are non-negotiable (plus the score): enough history to
    compute meaningfully, a base of at least base_min_days, and no strong
    ongoing downtrend. Everything else is expressed through the score.
    """
    if m["base_days"] is None or m["base_days"] < p.base_min_days:
        return False
    # Unmeasurable -> skip, rather than passing a name on unreliable numbers.
    if m["accumulation_score"] is None or m["base_slope_%_per_mo"] is None:
        return False
    if m["base_slope_%_per_mo"] < -p.max_base_decline_pct_per_mo:
        return False
    # The base has to be stabilisation *after* a decline, not a quiet stretch
    # following a rally. Without this the drawdown gate alone lets through names
    # that are merely far below an old high.
    if p.require_preceding_decline:
        sell = m["selloff_return_%"]
        if sell is None or sell > -p.min_selloff_decline_pct:
            return False
    return m["accumulation_score"] >= p.min_accumulation_score


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

            # gate 1b: valuation. Cheap on forward earnings, not just beaten
            # down. Checked here because info is already in hand -- it rejects
            # before the history fetch, so it costs no extra Yahoo request.
            fwd_pe = info.get("forwardPE")
            if p.fwd_pe_max is not None:
                if fwd_pe is None or fwd_pe <= 0 or fwd_pe > p.fwd_pe_max:
                    continue

            # One OHLCV fetch per ticker, reused by every stage below. The ATH
            # path needs the full series; otherwise 2y comfortably covers the
            # base + selloff windows. dropna: yfinance appends today's
            # not-yet-closed session as a NaN row, and since NaN > x is always
            # False that would quietly disable the drawdown gate below.
            period = "max" if p.use_all_time_high else "2y"
            df = tk.history(period=period, auto_adjust=True)
            if df is None or df.empty or "Close" not in df:
                continue
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue
            hist = df["Close"]

            # gate 2: big drawdown from the high
            if p.use_all_time_high:
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
            # Kept as-is: these two fields are live columns on the screener page.
            range_pct = None
            drift_pct = None
            try:
                window = hist.tail(p.quiet_lookback_days)
                if len(window) >= 5:
                    w_mean = float(window.mean())
                    w_first = float(window.iloc[0])
                    if w_mean > 0:
                        range_pct = (float(window.max()) - float(window.min())) / w_mean * 100
                    if w_first > 0:
                        drift_pct = (float(window.iloc[-1]) / w_first - 1) * 100
            except Exception:
                pass

            if p.require_quiet:
                if range_pct is None or drift_pct is None:
                    continue
                if range_pct > p.max_range_pct:            # too wide a range -> still swinging
                    continue
                if abs(drift_pct) > p.max_drift_pct:       # trending, not flat
                    continue

            # gate 4 (optional): accumulation base. Diagnostics are computed for
            # every survivor either way, so a disabled gate still tells you why.
            base = compute_base_metrics(df, p)
            if p.require_accumulation and not passes_accumulation(base, p):
                continue

            rec = {
                "ticker": t, "sector": info.get("sector"),
                "price": price, "high": high,
                "drawdown_%": drawdown, "fcf_yield_%": fcf_yield,
                "range_1m_%": range_pct,                   # last-month high-to-low spread
                "drift_1m_%": drift_pct,                   # last-month net change
                "pe": info.get("trailingPE"),        # eyeball only
                "fwd_pe": fwd_pe,                    # gated when fwd_pe_max is set
            }
            rec.update(base)                         # base/accumulation diagnostics
            records.append(rec)
        except Exception:
            continue
        finally:
            time.sleep(0.1)      # runs on EVERY path (continue/skip/error) -> no Yahoo hammering
    return records
