"""
NEXUS AI SCALPER — Production Server
Supports: Exness + FusionMarkets via MetaAPI cloud
Deploy: Railway one-click
"""

import asyncio, os, json, time, threading, logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

# ── EA Core ──────────────────────────────────────────────────────────
from ea_core import (
    NexusAIScalperEA, EAConfig, MarketData, AccountState,
    ClosedTrade, INSTRUMENT_PROFILES
)

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("nexus")

# ── MetaAPI ──────────────────────────────────────────────────────────
try:
    from metaapi_cloud_sdk import MetaApi
    METAAPI_OK = True
except ImportError:
    METAAPI_OK = False
    logger.warning("MetaAPI SDK not installed — running DEMO mode")

# ══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT VARIABLES (set these in Railway dashboard)
# ══════════════════════════════════════════════════════════════════════
BOT_TOKEN          = os.getenv("BOT_TOKEN",           "nexus-change-me")
PORT               = int(os.getenv("PORT",            "8000"))
METAAPI_TOKEN      = os.getenv("METAAPI_TOKEN",       "")
ACCOUNT_1_ID       = os.getenv("ACCOUNT_1_ID",        "")  # Exness
ACCOUNT_2_ID       = os.getenv("ACCOUNT_2_ID",        "")  # FusionMarkets
ACTIVE_ACCOUNT     = os.getenv("ACTIVE_ACCOUNT",      "1") # "1" or "2"

# ══════════════════════════════════════════════════════════════════════
#  EA + STATE
# ══════════════════════════════════════════════════════════════════════
ea_cfg = EAConfig()
ea     = NexusAIScalperEA(ea_cfg)

state = {
    "running":       False,
    "connected":     False,
    "demo_mode":     not METAAPI_OK,
    "active_broker": "Exness" if ACTIVE_ACCOUNT == "1" else "FusionMarkets",
    "account":       {},
    "positions":     [],
    "closed_today":  [],
    "equity_curve":  [],
    "stats":         {},
    "signals":       [],
    "logs":          [],
    "config": {
        "pairs":          ea_cfg.active_pairs,
        "risk_pct":       ea_cfg.risk_per_trade_pct,
        "min_ai_score":   ea_cfg.min_ai_score,
        "chain_enabled":  ea_cfg.momentum_chain_enabled,
        "max_chain":      ea_cfg.momentum_max_chain,
    },
    "brokers": {
        "1": {"name": "Exness",        "connected": False, "account_id": ACCOUNT_1_ID},
        "2": {"name": "FusionMarkets", "connected": False, "account_id": ACCOUNT_2_ID},
    }
}

ws_clients: list[WebSocket] = []
auth  = HTTPBearer()
_loop: asyncio.AbstractEventLoop = None

# MetaAPI objects
_meta_api   = None
_mt5_account = None
_mt5_conn    = None

# ══════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════
def verify(creds: HTTPAuthorizationCredentials = Depends(auth)):
    if creds.credentials != BOT_TOKEN:
        raise HTTPException(401, "Invalid token")
    return creds.credentials

# ══════════════════════════════════════════════════════════════════════
#  BROADCAST
# ══════════════════════════════════════════════════════════════════════
async def broadcast(data: dict):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(data)
        except:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)

def log(msg: str, level: str = "INFO"):
    entry = {"t": datetime.utcnow().strftime("%H:%M:%S"), "msg": msg, "level": level}
    state["logs"].insert(0, entry)
    state["logs"] = state["logs"][:300]
    logger.info(msg)
    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast({"type": "log", "data": entry}), _loop)

# ══════════════════════════════════════════════════════════════════════
#  METAAPI BROKER CONNECTION
# ══════════════════════════════════════════════════════════════════════
async def connect_broker(account_id: str) -> bool:
    global _meta_api, _mt5_account, _mt5_conn
    if not METAAPI_OK:
        log("⚠️  Running in DEMO mode — MetaAPI not installed")
        return False
    if not METAAPI_TOKEN or not account_id:
        log("⚠️  METAAPI_TOKEN or Account ID not set — running DEMO")
        return False
    try:
        log(f"🔌 Connecting to broker account {account_id}...")
        _meta_api    = MetaApi(METAAPI_TOKEN)
        _mt5_account = await _meta_api.metatrader_account_api.get_account(account_id)

        if _mt5_account.state not in ["DEPLOYING", "DEPLOYED"]:
            log("📡 Deploying MT5 account on MetaAPI cloud...")
            await _mt5_account.deploy()

        await _mt5_account.wait_deployed()
        _mt5_conn = _mt5_account.get_rpc_connection()
        await _mt5_conn.connect()
        await _mt5_conn.wait_synchronized()
        log("✅ MT5 account connected via MetaAPI")
        return True
    except Exception as e:
        log(f"❌ MetaAPI connection failed: {e}", "ERROR")
        return False

async def get_live_account() -> dict:
    if not _mt5_conn:
        return {}
    try:
        info = await _mt5_conn.get_account_information()
        return {
            "balance":     info.get("balance", 0),
            "equity":      info.get("equity", 0),
            "margin":      info.get("margin", 0),
            "free_margin": info.get("freeMargin", 0),
            "leverage":    info.get("leverage", 100),
            "currency":    info.get("currency", "USD"),
            "name":        info.get("name", ""),
            "login":       info.get("login", ""),
            "server":      info.get("broker", state["active_broker"]),
            "daily_pnl":   state["account"].get("daily_pnl", 0.0),
        }
    except Exception as e:
        log(f"Account info error: {e}", "ERROR")
        return state["account"]

async def get_live_positions() -> list:
    if not _mt5_conn:
        return []
    try:
        positions = await _mt5_conn.get_positions()
        result = []
        for p in positions:
            result.append({
                "id":        p.get("id", ""),
                "symbol":    p.get("symbol", ""),
                "direction": "BUY" if p.get("type") == "POSITION_TYPE_BUY" else "SELL",
                "entry":     p.get("openPrice", 0),
                "sl":        p.get("stopLoss", 0),
                "tp":        p.get("takeProfit", 0),
                "lot":       p.get("volume", 0),
                "profit":    round(p.get("profit", 0), 2),
                "current":   p.get("currentPrice", 0),
                "chain":     False,
                "chain_n":   0,
            })
        return result
    except Exception as e:
        log(f"Positions error: {e}", "ERROR")
        return state["positions"]

async def place_live_order(signal) -> dict:
    if not _mt5_conn:
        log(f"[DEMO] {signal.direction} {signal.lot_size} {signal.symbol} @ {signal.entry_price}")
        return {"success": True, "demo": True, "id": f"DEMO-{int(time.time())}"}
    try:
        if signal.direction == "BUY":
            r = await _mt5_conn.create_market_buy_order(
                signal.symbol, signal.lot_size,
                signal.stop_loss, signal.take_profit,
                {"comment": f"NEXUS|AI:{signal.confidence:.0%}|Chain:{signal.chain_count}"}
            )
        else:
            r = await _mt5_conn.create_market_sell_order(
                signal.symbol, signal.lot_size,
                signal.stop_loss, signal.take_profit,
                {"comment": f"NEXUS|AI:{signal.confidence:.0%}|Chain:{signal.chain_count}"}
            )
        log(f"✅ Live order placed: {signal.direction} {signal.lot_size} {signal.symbol}")
        return {"success": True, "id": r.get("orderId", "")}
    except Exception as e:
        log(f"❌ Order failed: {e}", "ERROR")
        return {"success": False, "error": str(e)}

async def get_live_candles(symbol: str) -> Optional[MarketData]:
    if not _mt5_conn:
        return None
    try:
        candles = await _mt5_conn.get_historical_candles(symbol, "5m", 300)
        if not candles or len(candles) < 50:
            return None
        profile = INSTRUMENT_PROFILES.get(symbol, {})
        opens  = np.array([c.get("open",  0) for c in candles])
        highs  = np.array([c.get("high",  0) for c in candles])
        lows   = np.array([c.get("low",   0) for c in candles])
        closes = np.array([c.get("close", 0) for c in candles])
        vols   = np.array([c.get("tickVolume", 1) for c in candles])
        tick   = await _mt5_conn.get_symbol_price(symbol)
        spread = 0.0
        if tick:
            pip = profile.get("pip", 0.0001)
            spread = round((tick.get("ask", 0) - tick.get("bid", 0)) / pip, 1)
        return MarketData(
            symbol=symbol, timestamp=datetime.utcnow(),
            open=opens, high=highs, low=lows, close=closes, volume=vols,
            spread_native=spread,
            asset_class=profile.get("class", "forex")
        )
    except:
        return None

# ══════════════════════════════════════════════════════════════════════
#  DEMO MARKET DATA (when no MetaAPI)
# ══════════════════════════════════════════════════════════════════════
def demo_market(symbol: str) -> MarketData:
    profile = INSTRUMENT_PROFILES.get(symbol, {})
    cls     = profile.get("class", "forex")
    base    = {"forex": 1.08, "cfd": 1950.0, "crypto": 42000.0}.get(cls, 1.08)
    scale   = {"forex": 0.0003, "cfd": 2.0, "crypto": 200.0}.get(cls, 0.0003)
    seed    = int(time.time() / 300) + abs(hash(symbol)) % 9999
    np.random.seed(seed % 2**31)
    n  = 300
    pr = base + np.cumsum(np.random.randn(n) * scale)
    o  = pr
    c  = pr + np.random.randn(n) * scale * 0.3
    h  = np.maximum(o, c) + np.abs(np.random.randn(n)) * scale * 0.2
    l  = np.minimum(o, c) - np.abs(np.random.randn(n)) * scale * 0.2
    v  = np.random.randint(100, 3000, n).astype(float)
    sp = {"forex": 0.9, "cfd": 20.0, "crypto": 80.0}.get(cls, 1.0)
    return MarketData(symbol=symbol, timestamp=datetime.utcnow(),
                      open=o, high=h, low=l, close=c, volume=v,
                      spread_native=sp, asset_class=cls)

def demo_account_state() -> AccountState:
    acc = state["account"]
    bal = acc.get("balance", 10000.0)
    eq  = acc.get("equity", bal)
    return AccountState(
        balance=bal, equity=eq,
        open_trades=len(state["positions"]),
        daily_pnl=acc.get("daily_pnl", 0.0),
        daily_trades=len(state["closed_today"]),
        drawdown_pct=max(0, (bal - eq) / bal * 100),
        peak_balance=max(bal, eq),
        open_positions=state["positions"],
    )

# ══════════════════════════════════════════════════════════════════════
#  MAIN EA LOOP
# ══════════════════════════════════════════════════════════════════════
def run_ea_thread():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(_ea_loop())

async def _ea_loop():
    log(f"🤖 NEXUS EA started | Broker: {state['active_broker']}")

    # Init demo account if needed
    if not state["account"]:
        state["account"] = {
            "balance": 10000.0, "equity": 10000.0,
            "free_margin": 10000.0, "margin": 0.0,
            "daily_pnl": 0.0, "daily_trades": 0,
            "leverage": 500, "currency": "USD",
            "server": state["active_broker"],
            "name": "Loading...", "login": "...",
        }

    # Connect to live broker
    acct_id = ACCOUNT_1_ID if ACTIVE_ACCOUNT == "1" else ACCOUNT_2_ID
    live = await connect_broker(acct_id)
    state["connected"] = True

    if live:
        info = await get_live_account()
        if info:
            state["account"].update(info)
            log(f"💰 Account: {info.get('name')} | Balance: {info.get('currency')} {info.get('balance'):,.2f}")
    else:
        log("🟡 Running in DEMO mode — no live broker connected")

    tick = 0
    while state["running"]:
        try:
            # ── Get account state ────────────────────────────────────
            if live:
                info = await get_live_account()
                if info:
                    state["account"].update(info)
                account = AccountState(
                    balance=state["account"].get("balance", 10000),
                    equity=state["account"].get("equity", 10000),
                    open_trades=len(state["positions"]),
                    daily_pnl=state["account"].get("daily_pnl", 0),
                    daily_trades=len(state["closed_today"]),
                    drawdown_pct=0.0,
                    peak_balance=state["account"].get("balance", 10000),
                    open_positions=state["positions"],
                )
                # Refresh live positions
                state["positions"] = await get_live_positions()
            else:
                account = demo_account_state()

            # ── Analyze each pair ────────────────────────────────────
            for symbol in ea_cfg.active_pairs:
                if not state["running"]:
                    break

                # Get market data
                market = None
                if live:
                    market = await get_live_candles(symbol)
                if not market:
                    market = demo_market(symbol)

                # Run EA analysis
                signal = ea.analyze(market, account, datetime.utcnow())

                if signal:
                    sig_entry = {
                        "symbol":  signal.symbol,
                        "direction": signal.direction,
                        "entry":   signal.entry_price,
                        "sl":      signal.stop_loss,
                        "tp":      signal.take_profit,
                        "lot":     signal.lot_size,
                        "ai":      round(signal.confidence * 100, 1),
                        "reason":  signal.reason,
                        "chain":   signal.is_chain_entry,
                        "chain_n": signal.chain_count,
                        "time":    signal.timestamp.strftime("%H:%M:%S"),
                    }
                    state["signals"].insert(0, sig_entry)
                    state["signals"] = state["signals"][:50]

                    # Place order
                    result = await place_live_order(signal)

                    if result.get("success"):
                        label = "⛓️ CHAIN" if signal.is_chain_entry else "🎯"
                        log(f"[{symbol}] {label} {signal.direction} | "
                            f"Lot:{signal.lot_size} AI:{signal.confidence:.0%} | "
                            f"Entry:{signal.entry_price:.5f} SL:{signal.stop_loss:.5f} TP:{signal.take_profit:.5f}")

                        if not live:
                            # Demo: track position internally
                            state["positions"].append({
                                "id":        result.get("id", f"{symbol}-{int(time.time())}"),
                                "symbol":    symbol,
                                "direction": signal.direction,
                                "entry":     signal.entry_price,
                                "sl":        signal.stop_loss,
                                "tp":        signal.take_profit,
                                "lot":       signal.lot_size,
                                "confidence":signal.confidence,
                                "chain":     signal.is_chain_entry,
                                "chain_n":   signal.chain_count,
                                "profit":    0.0,
                                "current":   signal.entry_price,
                            })
                    else:
                        log(f"[{symbol}] ❌ Order failed: {result.get('error','')}", "ERROR")

            # ── Demo P&L simulation ──────────────────────────────────
            if not live:
                closed = []
                for pos in state["positions"]:
                    m   = demo_market(pos["symbol"])
                    cur = m.close[-1]
                    p   = INSTRUMENT_PROFILES.get(pos["symbol"], {})
                    pip = p.get("pip", 0.0001)
                    pv  = p.get("pip_val", 10.0)

                    pnl = ((cur - pos["entry"]) if pos["direction"] == "BUY"
                           else (pos["entry"] - cur)) / pip * pv * pos["lot"]
                    pos["profit"]  = round(pnl, 2)
                    pos["current"] = round(cur, 5)

                    hit_tp = ((pos["direction"] == "BUY"  and cur >= pos["tp"]) or
                              (pos["direction"] == "SELL" and cur <= pos["tp"]))
                    hit_sl = ((pos["direction"] == "BUY"  and cur <= pos["sl"]) or
                              (pos["direction"] == "SELL" and cur >= pos["sl"]))

                    if hit_tp or hit_sl:
                        reason   = "TP" if hit_tp else "SL"
                        pnl_pips = (p.get("tp_pips", 12) if hit_tp
                                    else -p.get("sl_pips", 5))
                        pnl_usd  = round(pnl_pips * pv * pos["lot"], 2)

                        state["closed_today"].insert(0, {
                            "symbol":    pos["symbol"],
                            "direction": pos["direction"],
                            "result":    reason,
                            "pips":      pnl_pips,
                            "usd":       pnl_usd,
                            "chain":     pos.get("chain", False),
                            "chain_n":   pos.get("chain_n", 0),
                            "lot":       pos["lot"],
                            "time":      datetime.utcnow().strftime("%H:%M:%S"),
                        })
                        state["closed_today"] = state["closed_today"][:200]

                        state["account"]["daily_pnl"] = (
                            state["account"].get("daily_pnl", 0) + pnl_usd)
                        state["account"]["balance"] = (
                            state["account"].get("balance", 10000) + pnl_usd)

                        ico = "✅" if hit_tp else "❌"
                        log(f"[{pos['symbol']}] {ico} {reason} | "
                            f"{pnl_pips:+.0f} pips | ${pnl_usd:+.2f}")
                        closed.append(pos)

                        # Momentum chain check
                        if hit_tp:
                            ct = ClosedTrade(
                                symbol=pos["symbol"], direction=pos["direction"],
                                entry_price=pos["entry"],
                                exit_price=pos["tp"] if hit_tp else pos["sl"],
                                lot_size=pos["lot"], pnl_pips=pnl_pips,
                                pnl_usd=pnl_usd, confidence=pos["confidence"],
                                close_reason=reason, opened_at=datetime.utcnow(),
                                closed_at=datetime.utcnow(),
                                is_chain_entry=pos.get("chain", False),
                                chain_count=pos.get("chain_n", 0),
                            )
                            chain_sig = ea.on_trade_closed(ct, m, account)
                            if chain_sig:
                                log(f"[{pos['symbol']}] ⛓️ Chain #{chain_sig.chain_count} | "
                                    f"AI:{chain_sig.confidence:.0%}")

                state["positions"] = [p for p in state["positions"] if p not in closed]

            # ── Equity curve ─────────────────────────────────────────
            tick += 1
            if tick % 6 == 0:
                state["equity_curve"].append({
                    "t": datetime.utcnow().strftime("%H:%M"),
                    "v": round(state["account"].get("balance", 10000), 2),
                })
                state["equity_curve"] = state["equity_curve"][-100:]

            # ── Stats ────────────────────────────────────────────────
            state["stats"] = ea.summary()
            state["account"]["equity"] = (
                state["account"].get("balance", 10000) +
                sum(p.get("profit", 0) for p in state["positions"])
            )
            state["account"]["open_trades"] = len(state["positions"])

            # ── Broadcast to dashboard ───────────────────────────────
            await broadcast({
                "type":      "tick",
                "positions": state["positions"],
                "account":   state["account"],
                "signals":   state["signals"][:5],
                "closed":    state["closed_today"][:10],
                "stats":     state["stats"],
            })

            await asyncio.sleep(5)

        except Exception as e:
            log(f"❌ Loop error: {e}", "ERROR")
            await asyncio.sleep(10)

    log("⏹️ NEXUS EA stopped")
    if _mt5_conn:
        try:
            await _mt5_conn.close()
        except:
            pass

# ══════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(title="NEXUS AI Scalper", version="2.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "running": state["running"],
            "broker": state["active_broker"]}

@app.get("/api/state")
def get_state(t: str = Depends(verify)):
    return state

@app.post("/api/start")
def start_ea(t: str = Depends(verify)):
    if state["running"]:
        return {"status": "already_running"}
    state["running"] = True
    threading.Thread(target=run_ea_thread, daemon=True).start()
    return {"status": "started", "broker": state["active_broker"]}

@app.post("/api/stop")
def stop_ea(t: str = Depends(verify)):
    state["running"]   = False
    state["connected"] = False
    return {"status": "stopped"}

class BrokerSwitch(BaseModel):
    account: str  # "1" = Exness, "2" = FusionMarkets

@app.post("/api/broker/switch")
def switch_broker(req: BrokerSwitch, t: str = Depends(verify)):
    if state["running"]:
        return JSONResponse(status_code=400,
            content={"error": "Stop the EA before switching broker"})
    global ACTIVE_ACCOUNT
    ACTIVE_ACCOUNT = req.account
    state["active_broker"] = ("Exness" if req.account == "1"
                               else "FusionMarkets")
    return {"status": "switched", "broker": state["active_broker"]}

class ConfigUpdate(BaseModel):
    risk_pct:      Optional[float] = None
    min_ai_score:  Optional[float] = None
    chain_enabled: Optional[bool]  = None
    max_chain:     Optional[int]   = None

@app.put("/api/config")
def update_config(upd: ConfigUpdate, t: str = Depends(verify)):
    if upd.risk_pct is not None:
        ea_cfg.risk_per_trade_pct = upd.risk_pct
        state["config"]["risk_pct"] = upd.risk_pct
    if upd.min_ai_score is not None:
        ea_cfg.min_ai_score = upd.min_ai_score
        ea.cfg.min_ai_score = upd.min_ai_score
        state["config"]["min_ai_score"] = upd.min_ai_score
    if upd.chain_enabled is not None:
        ea_cfg.momentum_chain_enabled = upd.chain_enabled
        ea.cfg.momentum_chain_enabled = upd.chain_enabled
        state["config"]["chain_enabled"] = upd.chain_enabled
    if upd.max_chain is not None:
        ea_cfg.momentum_max_chain = upd.max_chain
        ea.cfg.momentum_max_chain = upd.max_chain
        state["config"]["max_chain"] = upd.max_chain
    return state["config"]

@app.websocket("/ws/{token}")
async def websocket_handler(websocket: WebSocket, token: str):
    if token != BOT_TOKEN:
        await websocket.close(1008)
        return
    await websocket.accept()
    ws_clients.append(websocket)
    await websocket.send_json({"type": "init", "data": state})
    try:
        while True:
            await asyncio.sleep(20)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        if websocket in ws_clients:
            ws_clients.remove(websocket)

# Serve dashboard
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
