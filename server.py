import os
import time
import threading
import logging
from datetime import datetime
from typing import Optional
import numpy as np

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ea_core import (
    NexusAIScalperEA, EAConfig, MarketData,
    AccountState, ClosedTrade, INSTRUMENT_PROFILES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus")

BOT_TOKEN = os.environ.get("BOT_TOKEN", "Nexus2024secure")
ea_cfg = EAConfig()
ea = NexusAIScalperEA(ea_cfg)

state = {
    "running": False,
    "connected": False,
    "demo_mode": True,
    "active_broker": "FusionMarkets",
    "account": {
        "balance": 10000.0,
        "equity": 10000.0,
        "free_margin": 10000.0,
        "margin": 0.0,
        "daily_pnl": 0.0,
        "leverage": 500,
        "currency": "USD",
        "server": "Demo",
        "name": "Demo Account",
        "login": "Ready",
        "open_trades": 0,
    },
    "positions": [],
    "closed_today": [],
    "equity_curve": [],
    "stats": {},
    "signals": [],
    "logs": [],
    "config": {
        "pairs": ea_cfg.active_pairs,
        "risk_pct": ea_cfg.risk_per_trade_pct,
        "min_ai_score": ea_cfg.min_ai_score,
        "chain_enabled": ea_cfg.momentum_chain_enabled,
        "max_chain": ea_cfg.momentum_max_chain,
    }
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

auth = HTTPBearer()


def verify(creds: HTTPAuthorizationCredentials = Depends(auth)):
    if creds.credentials != BOT_TOKEN:
        raise HTTPException(401, "Invalid token")
    return creds.credentials


def log(msg):
    entry = {"t": datetime.utcnow().strftime("%H:%M:%S"), "msg": msg, "level": "INFO"}
    state["logs"].insert(0, entry)
    state["logs"] = state["logs"][:200]
    logger.info(msg)


def demo_market(symbol):
    profile = INSTRUMENT_PROFILES.get(symbol, {})
    cls = profile.get("class", "forex")
    base = {"forex": 1.08, "cfd": 1950.0, "crypto": 42000.0}.get(cls, 1.08)
    scale = {"forex": 0.0003, "cfd": 2.0, "crypto": 200.0}.get(cls, 0.0003)
    seed = int(time.time() / 300) + abs(hash(symbol)) % 9999
    np.random.seed(seed % 2**31)
    n = 300
    pr = base + np.cumsum(np.random.randn(n) * scale)
    c = pr + np.random.randn(n) * scale * 0.3
    h = np.maximum(pr, c) + np.abs(np.random.randn(n)) * scale * 0.2
    l = np.minimum(pr, c) - np.abs(np.random.randn(n)) * scale * 0.2
    v = np.random.randint(100, 3000, n).astype(float)
    sp = {"forex": 0.9, "cfd": 20.0, "crypto": 80.0}.get(cls, 1.0)
    return MarketData(
        symbol=symbol, timestamp=datetime.utcnow(),
        open=pr, high=h, low=l, close=c, volume=v,
        spread_native=sp, asset_class=cls
    )


def run_bot():
    log("🤖 NEXUS EA started")
    tick = 0
    while state["running"]:
        try:
            account = AccountState(
                balance=state["account"]["balance"],
                equity=state["account"]["equity"],
                open_trades=len(state["positions"]),
                daily_pnl=state["account"]["daily_pnl"],
                daily_trades=len(state["closed_today"]),
                drawdown_pct=0.0,
                peak_balance=state["account"]["balance"],
                open_positions=state["positions"],
            )

            for symbol in ea_cfg.active_pairs:
                if not state["running"]:
                    break
                market = demo_market(symbol)
                signal = ea.analyze(market, account, datetime.utcnow())
                if signal:
                    state["signals"].insert(0, {
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "entry": signal.entry_price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "lot": signal.lot_size,
                        "ai": round(signal.confidence * 100, 1),
                        "reason": signal.reason,
                        "chain": signal.is_chain_entry,
                        "chain_n": signal.chain_count,
                        "time": signal.timestamp.strftime("%H:%M:%S"),
                    })
                    state["signals"] = state["signals"][:50]
                    state["positions"].append({
                        "id": f"{symbol}-{int(time.time())}",
                        "symbol": symbol,
                        "direction": signal.direction,
                        "entry": signal.entry_price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "lot": signal.lot_size,
                        "confidence": signal.confidence,
                        "chain": signal.is_chain_entry,
                        "chain_n": signal.chain_count,
                        "profit": 0.0,
                        "current": signal.entry_price,
                    })
                    log(f"[{symbol}] {'⛓️' if signal.is_chain_entry else '🎯'} {signal.direction} | AI:{signal.confidence:.0%}")

            closed = []
            for pos in state["positions"]:
                m = demo_market(pos["symbol"])
                cur = m.close[-1]
                p = INSTRUMENT_PROFILES.get(pos["symbol"], {})
                pip = p.get("pip", 0.0001)
                pv = p.get("pip_val", 10.0)
                pnl = ((cur - pos["entry"]) if pos["direction"] == "BUY"
                       else (pos["entry"] - cur)) / pip * pv * pos["lot"]
                pos["profit"] = round(pnl, 2)
                pos["current"] = round(cur, 5)
                hit_tp = ((pos["direction"] == "BUY" and cur >= pos["tp"]) or
                          (pos["direction"] == "SELL" and cur <= pos["tp"]))
                hit_sl = ((pos["direction"] == "BUY" and cur <= pos["sl"]) or
                          (pos["direction"] == "SELL" and cur >= pos["sl"]))
                if hit_tp or hit_sl:
                    reason = "TP" if hit_tp else "SL"
                    pnl_pips = p.get("tp_pips", 12) if hit_tp else -p.get("sl_pips", 5)
                    pnl_usd = round(pnl_pips * pv * pos["lot"], 2)
                    state["closed_today"].insert(0, {
                        "symbol": pos["symbol"],
                        "direction": pos["direction"],
                        "result": reason,
                        "pips": pnl_pips,
                        "usd": pnl_usd,
                        "chain": pos.get("chain", False),
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                    })
                    state["closed_today"] = state["closed_today"][:200]
                    state["account"]["daily_pnl"] += pnl_usd
                    state["account"]["balance"] += pnl_usd
                    log(f"[{pos['symbol']}] {'✅' if hit_tp else '❌'} {reason} | {pnl_pips:+.0f} pips | ${pnl_usd:+.2f}")
                    closed.append(pos)

            state["positions"] = [p for p in state["positions"] if p not in closed]

            tick += 1
            if tick % 6 == 0:
                state["equity_curve"].append({
                    "t": datetime.utcnow().strftime("%H:%M"),
                    "v": round(state["account"]["balance"], 2),
                })
                state["equity_curve"] = state["equity_curve"][-100:]

            state["stats"] = ea.summary()
            state["account"]["equity"] = (
                state["account"]["balance"] +
                sum(p.get("profit", 0) for p in state["positions"])
            )
            state["account"]["open_trades"] = len(state["positions"])
            time.sleep(5)

        except Exception as e:
            log(f"Error: {e}")
            time.sleep(10)

    log("⏹️ EA stopped")


@app.get("/api/health")
def health():
    return {"status": "ok", "running": state["running"]}


@app.get("/api/state")
def get_state(t: str = Depends(verify)):
    return state


@app.post("/api/start")
def start_ea(t: str = Depends(verify)):
    if state["running"]:
        return {"status": "already_running"}
    state["running"] = True
    state["connected"] = True
    threading.Thread(target=run_bot, daemon=True).start()
    return {"status": "started"}


@app.post("/api/stop")
def stop_ea(t: str = Depends(verify)):
    state["running"] = False
    state["connected"] = False
    return {"status": "stopped"}


class ConfigUpdate(BaseModel):
    risk_pct: Optional[float] = None
    min_ai_score: Optional[float] = None
    chain_enabled: Optional[bool] = None
    max_chain: Optional[int] = None


@app.put("/api/config")
def update_config(upd: ConfigUpdate, t: str = Depends(verify)):
    if upd.risk_pct is not None:
        ea_cfg.risk_per_trade_pct = upd.risk_pct
        state["config"]["risk_pct"] = upd.risk_pct
    if upd.min_ai_score is not None:
        ea_cfg.min_ai_score = upd.min_ai_score
        state["config"]["min_ai_score"] = upd.min_ai_score
    if upd.chain_enabled is not None:
        ea_cfg.momentum_chain_enabled = upd.chain_enabled
        state["config"]["chain_enabled"] = upd.chain_enabled
    if upd.max_chain is not None:
        ea_cfg.momentum_max_chain = upd.max_chain
        state["config"]["max_chain"] = upd.max_chain
    return state["config"]


# Serve static dashboard
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"status": "ok", "message": "NEXUS EA running. Static files not found."}