from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import numpy as np
import pandas as pd
import yfinance as yf

# Screening lives in screener.py so the nightly cron (scan_job.py) can import it
# without pulling in FastAPI. _num is shared: the risk endpoint below uses it too.
from screener import ScreenRequest, _num, clean_records, drawdown_sort_key, get_universe, run_screen

app = FastAPI()

TRADING_DAYS = 252
BENCHMARK = "SPY"
RF_ANNUAL = 0.045
LOOKBACK = "2y"


# ------------------------- models -------------------------
class Position(BaseModel):
    ticker: str
    shares: float
    cost_basis: float | None = None
    stop_loss: float | None = None   # sent by Next; None when no stop is set


class RiskRequest(BaseModel):
    positions: list[Position]
    cash: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok"}


def fetch_fundamentals(tickers):
    """Per-ticker valuation stats from yfinance .info.
    Missing values stay None so Claude can say 'unavailable' instead of
    guessing a multiple from memory (the whole reason this exists)."""
    out = {}
    for t in tickers:
        rec = {
            "fwd_pe": None, "pe": None, "peg": None, "ps": None,
            "fcf_yield_%": None, "sector": None, "pct_from_52w_high_%": None,
        }
        try:
            info = yf.Ticker(t).info
            rec["fwd_pe"] = _num(info.get("forwardPE"))
            rec["pe"]     = _num(info.get("trailingPE"))
            rec["peg"]    = _num(info.get("trailingPegRatio"))
            rec["ps"]     = _num(info.get("priceToSalesTrailing12Months"))
            rec["sector"] = info.get("sector")

            fcf  = info.get("freeCashflow")
            mcap = info.get("marketCap")
            if fcf is not None and mcap:
                rec["fcf_yield_%"] = _num(fcf / mcap * 100)

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            hi = info.get("fiftyTwoWeekHigh")
            if price and hi:
                rec["pct_from_52w_high_%"] = _num((price / hi - 1) * 100)
        except Exception:
            pass
        finally:
            time.sleep(0.1)   # be gentle with Yahoo, same as the screener
        out[t] = rec
    return out


# ========================= RISK (unchanged) =========================
def fetch_prices(tickers, period):
    px = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    return px[tickers].dropna()


def analyze(holdings, cash, prices, benchmark=BENCHMARK, rf_annual=RF_ANNUAL):
    tickers = [h["ticker"] for h in holdings]
    shares  = np.array([h["shares"] for h in holdings], float)
    cost    = np.array([h["cost_basis"] for h in holdings], float)

    R_all = prices.pct_change().dropna()
    R     = R_all[tickers]
    bench = R_all[benchmark]

    price     = prices.iloc[-1][tickers].values
    mkt_val   = shares * price
    cost_val  = shares * cost
    pnl       = mkt_val - cost_val
    pnl_pct   = pnl / cost_val
    equity_value = mkt_val.sum()
    total_value  = equity_value + cash
    w            = mkt_val / total_value
    cash_w       = cash / total_value

    Sigma    = R.cov().values * TRADING_DAYS
    sigma    = R.std().values * np.sqrt(TRADING_DAYS)
    port_var = w @ Sigma @ w
    sigma_p  = np.sqrt(port_var)

    Sw   = Sigma @ w
    mctr = Sw / sigma_p
    cctr = w * mctr
    pctr = cctr / sigma_p

    var_b     = bench.var()
    beta_i    = np.array([R[t].cov(bench) / var_b for t in tickers])
    port_beta = float(w @ beta_i)

    port_ret = R.values @ w

    def var_cvar(series, level):
        q    = np.percentile(series, (1 - level) * 100)
        cvar = series[series <= q].mean()
        return -q, -cvar

    v95, c95 = var_cvar(port_ret, 0.95)
    v99, _   = var_cvar(port_ret, 0.99)
    par_v95  = -(port_ret.mean() - 1.645 * port_ret.std())

    equity_curve = (1 + port_ret).cumprod()
    peak         = np.maximum.accumulate(equity_curve)
    max_dd       = (equity_curve / peak - 1).min()

    ann_ret  = port_ret.mean() * TRADING_DAYS
    ann_vol  = port_ret.std()  * np.sqrt(TRADING_DAYS)
    sharpe   = (ann_ret - rf_annual) / ann_vol
    downside = port_ret[port_ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino  = (ann_ret - rf_annual) / downside

    w_eq  = mkt_val / equity_value
    herf  = float((w_eq ** 2).sum())
    eff_n = 1 / herf

    per_pos = pd.DataFrame({
        "shares": shares, "cost_basis": cost, "price": price,
        "mkt_value": mkt_val, "weight_%": w * 100,
        "unreal_pnl": pnl, "pnl_%": pnl_pct * 100,
        "beta": beta_i, "standalone_vol_%": sigma * 100,
        "MCTR": mctr, "CCTR": cctr, "PCTR_%": pctr * 100,
    }, index=tickers)

    portfolio = {
        "total_value": total_value, "equity_value": equity_value, "cash": cash,
        "cash_weight_%": cash_w * 100, "total_unreal_pnl": pnl.sum(),
        "portfolio_vol_%": sigma_p * 100, "portfolio_beta": port_beta,
        "VaR_95_1d_%": v95 * 100, "VaR_95_1d_$": v95 * total_value,
        "CVaR_95_1d_%": c95 * 100, "CVaR_95_1d_$": c95 * total_value,
        "VaR_99_1d_%": v99 * 100, "VaR_95_parametric_%": par_v95 * 100,
        "max_drawdown_%": max_dd * 100,
        "sharpe": sharpe, "sortino": sortino,
        "herfindahl": herf, "effective_N": eff_n,
    }
    return per_pos, portfolio


@app.post("/risk")
def compute_risk(req: RiskRequest):
    holdings = [
        {
            "ticker": p.ticker,
            "shares": p.shares,
            "cost_basis": p.cost_basis if p.cost_basis is not None else 0.0,
        }
        for p in req.positions
    ]
    if not holdings:
        raise HTTPException(status_code=400, detail="No positions provided")

    tickers = [h["ticker"] for h in holdings]
    try:
        prices = fetch_prices(tickers + [BENCHMARK], LOOKBACK)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Price fetch failed: {e}")

    try:
        per_pos, portfolio = analyze(holdings, req.cash, prices)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Live valuation stats so the downstream Claude review doesn't guess multiples.
    try:
        fundamentals = fetch_fundamentals(tickers)
    except Exception:
        fundamentals = {}

    positions_out = []
    for ticker, row in per_pos.iterrows():
        rec = {"ticker": ticker}
        for col, val in row.items():
            rec[col] = _num(val)
        rec.update(fundamentals.get(ticker, {}))
        positions_out.append(rec)

    portfolio_out = {k: _num(v) for k, v in portfolio.items()}
    return {"positions": positions_out, "portfolio": portfolio_out}


# ========================= SCREENER =========================
# Logic lives in screener.py; this route is for manual testing (max_tickers).
# The Next.js screener page reads the Supabase snapshot written by scan_job.py.
@app.post("/screen")
def screen(req: ScreenRequest):
    try:
        tickers = get_universe()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Universe fetch failed: {e}")

    if req.max_tickers:
        tickers = tickers[: req.max_tickers]

    records = run_screen(tickers, req)
    records.sort(key=drawdown_sort_key)   # most beaten-down first (ordering, not ranking)

    clean = clean_records(records)
    return {
        "count": len(clean),
        "params": req.model_dump(),
        "results": clean,
    }
