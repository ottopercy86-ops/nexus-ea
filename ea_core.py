"""
NEXUS AI SCALPER — EA Core Engine
Multi-asset: Forex + Gold + Crypto CFDs
AI momentum chain re-entry after TP
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

INSTRUMENT_PROFILES = {
    "EURUSD": {"pip": 0.0001, "pip_val": 10.0, "class": "forex",  "sl_pips": 5,   "tp_pips": 12},
    "GBPUSD": {"pip": 0.0001, "pip_val": 10.0, "class": "forex",  "sl_pips": 5,   "tp_pips": 13},
    "USDJPY": {"pip": 0.01,   "pip_val": 9.1,  "class": "forex",  "sl_pips": 5,   "tp_pips": 12},
    "AUDUSD": {"pip": 0.0001, "pip_val": 10.0, "class": "forex",  "sl_pips": 5,   "tp_pips": 10},
    "USDCAD": {"pip": 0.0001, "pip_val": 7.5,  "class": "forex",  "sl_pips": 5,   "tp_pips": 10},
    "USDCHF": {"pip": 0.0001, "pip_val": 11.0, "class": "forex",  "sl_pips": 5,   "tp_pips": 10},
    "NZDUSD": {"pip": 0.0001, "pip_val": 10.0, "class": "forex",  "sl_pips": 5,   "tp_pips": 10},
    "XAUUSD": {"pip": 0.1,    "pip_val": 1.0,  "class": "cfd",    "sl_pips": 50,  "tp_pips": 120},
    "BTCUSD": {"pip": 1.0,    "pip_val": 0.01, "class": "crypto", "sl_pips": 200, "tp_pips": 500},
    "ETHUSD": {"pip": 0.1,    "pip_val": 0.1,  "class": "crypto", "sl_pips": 30,  "tp_pips": 80},
    "LTCUSD": {"pip": 0.01,   "pip_val": 1.0,  "class": "crypto", "sl_pips": 10,  "tp_pips": 25},
    "XRPUSD": {"pip": 0.0001, "pip_val": 10.0, "class": "crypto", "sl_pips": 5,   "tp_pips": 12},
}

@dataclass
class EAConfig:
    active_pairs: list = field(default_factory=lambda: [
        "EURUSD","GBPUSD","USDJPY","AUDUSD","XAUUSD","BTCUSD","ETHUSD"
    ])
    timeframe_minutes: int = 5
    risk_per_trade_pct: float = 0.5
    min_lot: float = 0.01
    max_lot: float = 10.0
    base_max_trades: int = 5
    trades_per_1k: float = 1.0
    ma_fast: int = 20
    ma_slow: int = 50
    ma_trend: int = 200
    rsi_period: int = 14
    rsi_ob: float = 65.0
    rsi_os: float = 35.0
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    adx_min: float = 25.0
    session_buffer_mins: int = 60
    block_rollover_start: int = 21
    block_rollover_end: int = 22
    max_spread_forex: float = 1.5
    max_spread_gold: float = 30.0
    max_spread_crypto: float = 150.0
    atr_spike_mult: float = 2.0
    min_ai_score: float = 0.62
    momentum_chain_enabled: bool = True
    momentum_confirm_bars: int = 2
    momentum_max_chain: int = 3
    momentum_min_score_boost: float = 0.05
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 8.0
    max_correlated_pairs: int = 2

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    spread_native: float = 0.0
    asset_class: str = "forex"
    def pip_size(self): return INSTRUMENT_PROFILES.get(self.symbol,{}).get("pip",0.0001)
    def pip_value(self): return INSTRUMENT_PROFILES.get(self.symbol,{}).get("pip_val",10.0)
    def sl_pips(self): return INSTRUMENT_PROFILES.get(self.symbol,{}).get("sl_pips",5)
    def tp_pips(self): return INSTRUMENT_PROFILES.get(self.symbol,{}).get("tp_pips",12)

@dataclass
class AccountState:
    balance: float
    equity: float
    open_trades: int
    daily_pnl: float
    daily_trades: int
    drawdown_pct: float
    peak_balance: float
    open_positions: list = field(default_factory=list)

@dataclass
class TradeSignal:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    confidence: float
    reason: str
    timestamp: datetime
    is_chain_entry: bool = False
    chain_count: int = 0
    signal_components: dict = field(default_factory=dict)

@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl_pips: float
    pnl_usd: float
    confidence: float
    close_reason: str
    opened_at: datetime
    closed_at: datetime
    is_chain_entry: bool = False
    chain_count: int = 0

class IndicatorEngine:
    @staticmethod
    def ema(data, period):
        alpha = 2.0 / (period + 1)
        out = np.empty_like(data); out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
        return out

    @staticmethod
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        ag = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        al = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = np.where(al == 0, 100, ag / al)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger(close, period=20, std=2.0):
        s = pd.Series(close)
        mid = s.rolling(period).mean().values
        dev = s.rolling(period).std().values
        return mid + std*dev, mid, mid - std*dev

    @staticmethod
    def atr(high, low, close, period=14):
        pc = np.roll(close,1); pc[0]=close[0]
        tr = np.maximum(high-low, np.maximum(np.abs(high-pc), np.abs(low-pc)))
        return pd.Series(tr).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def adx(high, low, close, period=14):
        ph=np.roll(high,1); ph[0]=high[0]
        pl=np.roll(low,1);  pl[0]=low[0]
        pc=np.roll(close,1);pc[0]=close[0]
        up=high-ph; dn=pl-low
        pdm=np.where((up>dn)&(up>0),up,0.0)
        mdm=np.where((dn>up)&(dn>0),dn,0.0)
        tr=np.maximum(high-low,np.maximum(np.abs(high-pc),np.abs(low-pc)))
        atr_=pd.Series(tr).ewm(span=period,adjust=False).mean().values
        pdi=100*pd.Series(pdm).ewm(span=period,adjust=False).mean().values/(atr_+1e-9)
        mdi=100*pd.Series(mdm).ewm(span=period,adjust=False).mean().values/(atr_+1e-9)
        dx=100*np.abs(pdi-mdi)/(pdi+mdi+1e-9)
        return pd.Series(dx).ewm(span=period,adjust=False).mean().values, pdi, mdi

    @staticmethod
    def rsi_divergence(close, rsi, lb=5):
        if len(close)<lb*2: return "none"
        if close[-1]<close[-lb] and rsi[-1]>rsi[-lb] and rsi[-1]<45: return "bull"
        if close[-1]>close[-lb] and rsi[-1]<rsi[-lb] and rsi[-1]>55: return "bear"
        return "none"

    @staticmethod
    def range_breakout(high, low, close, lookback=24):
        if len(high)<lookback+3: return {"type":"none"}
        ph=np.max(high[-lookback:-1]); pl=np.min(low[-lookback:-1])
        c=close[-1]; p=close[-2]; rng=ph-pl+1e-9
        if p<=ph<c: return {"type":"bull","level":ph,"strength":min(1.0,(c-ph)/rng*10)}
        if p>=pl>c: return {"type":"bear","level":pl,"strength":min(1.0,(pl-c)/rng*10)}
        return {"type":"none","level":0,"strength":0}

    @staticmethod
    def momentum_strength(close, bars=3):
        if len(close)<bars+1: return 0.0
        moves=np.diff(close[-bars-1:])
        total=np.sum(np.abs(moves))+1e-9
        return float(np.clip(np.sum(moves)/total,-1,1))

class AISignalScorer:
    BASE_WEIGHTS = {
        "trend_200":0.22,"ma_cross":0.10,"adx_strength":0.15,
        "rsi_zone":0.12,"rsi_divergence":0.13,"bb_position":0.09,
        "breakout":0.08,"session_quality":0.06,"spread_clear":0.05,
    }
    def __init__(self):
        self._weights={}; self._history={}; self._window=60
    def weights(self,sym):
        if sym not in self._weights:
            self._weights[sym]={k:v for k,v in self.BASE_WEIGHTS.items()}
        return self._weights[sym]
    def score(self,sym,comps):
        w=self.weights(sym)
        return float(np.clip(sum(w[k]*float(comps.get(k,0)) for k in w),0,1))
    def record(self,sym,score,won,chain=False):
        if sym not in self._history: self._history[sym]=[]
        self._history[sym].append({"score":score,"won":won,"chain":chain})
        if len(self._history[sym])>self._window: self._history[sym].pop(0)
        self._adapt(sym)
    def _adapt(self,sym):
        h=self._history[sym]
        if len(h)<15: return
        wr=sum(1 for x in h if x["won"])/len(h)
        w=self.weights(sym)
        if wr>0.65:
            w["trend_200"]=min(0.35,w["trend_200"]+0.003)
            w["adx_strength"]=min(0.25,w["adx_strength"]+0.002)
        elif wr<0.45:
            w["rsi_zone"]=min(0.20,w["rsi_zone"]+0.002)
            w["spread_clear"]=min(0.10,w["spread_clear"]+0.001)
        t=sum(w.values())
        for k in w: w[k]/=t
    def regime(self,adx,atr,atr_avg):
        if adx>30 and atr<atr_avg*1.3: return "trending"
        if adx<20 and atr<atr_avg*1.1: return "ranging"
        if atr>atr_avg*1.8: return "volatile"
        return "mixed"

CORRELATION_GROUPS = {
    "usd_long":  ["EURUSD","GBPUSD","AUDUSD","NZDUSD"],
    "usd_short": ["USDJPY","USDCAD","USDCHF"],
    "crypto":    ["BTCUSD","ETHUSD","LTCUSD","XRPUSD"],
    "gold":      ["XAUUSD"],
}

class RiskManager:
    def __init__(self,cfg): self.cfg=cfg
    def lot_size(self,balance,sl_pips,pip_val):
        risk=balance*(self.cfg.risk_per_trade_pct/100)
        sl_usd=sl_pips*pip_val
        lot=risk/sl_usd if sl_usd>0 else self.cfg.min_lot
        return round(float(np.clip(lot,self.cfg.min_lot,self.cfg.max_lot)),2)
    def max_trades(self,balance):
        extra=max(0,(balance-1000)/1000)
        return min(int(self.cfg.base_max_trades+extra*self.cfg.trades_per_1k),25)
    def can_trade(self,account):
        lp=(account.daily_pnl/max(account.balance,1))*100
        if lp<-self.cfg.daily_loss_limit_pct: return False,f"Daily loss limit {lp:.1f}%"
        if account.drawdown_pct>self.cfg.max_drawdown_pct: return False,"Drawdown limit"
        if account.open_trades>=self.max_trades(account.balance): return False,"Max trades"
        return True,"OK"
    def spread_ok(self,market):
        c=market.asset_class
        if c=="forex":  return market.spread_native<=self.cfg.max_spread_forex
        if c=="cfd":    return market.spread_native<=self.cfg.max_spread_gold
        if c=="crypto": return market.spread_native<=self.cfg.max_spread_crypto
        return True
    def corr_exposure(self,symbol,positions):
        syms=[p.get("symbol","") for p in positions]
        for grp in CORRELATION_GROUPS.values():
            if symbol in grp: return sum(1 for s in syms if s in grp)
        return 0

class SessionFilter:
    def __init__(self,cfg): self.cfg=cfg
    def check(self,utc_now,asset_class):
        h=utc_now.hour; m=utc_now.minute; tot=h*60+m
        if asset_class=="crypto":
            if self.cfg.block_rollover_start*60<=tot<self.cfg.block_rollover_end*60:
                return False,"Rollover"
            return True,"Crypto 24/7"
        if self.cfg.block_rollover_start*60<=tot<self.cfg.block_rollover_end*60:
            return False,"Rollover"
        buf=self.cfg.session_buffer_mins
        if (7*60+buf)<=tot<(16*60-buf): return True,"London"
        if (13*60+buf)<=tot<(21*60-buf): return True,"New York"
        return False,"Outside session"
    def is_overlap(self,utc_now): return 13<=utc_now.hour<16

class MomentumChainEngine:
    def __init__(self,cfg,scorer,ind):
        self.cfg=cfg; self.scorer=scorer; self.ind=ind
    def notify_tp(self,trade,market,account):
        if not self.cfg.momentum_chain_enabled: return None
        if trade.chain_count>=self.cfg.momentum_max_chain: return None
        c=market.close; mom=self.ind.momentum_strength(c,self.cfg.momentum_confirm_bars)
        if trade.direction=="BUY"  and mom<0.3:  return None
        if trade.direction=="SELL" and mom>-0.3: return None
        ema_f=self.ind.ema(c,20); ema_s=self.ind.ema(c,50); ema_t=self.ind.ema(c,200)
        rsi_v=self.ind.rsi(c,14)
        adx_v,pdi,mdi=self.ind.adx(market.high,market.low,c,14)
        cur=c[-1]; sym=trade.symbol; direction=trade.direction
        if direction=="BUY":
            comps={"trend_200":1.0 if cur>ema_t[-1] else 0.0,
                   "ma_cross":1.0 if ema_f[-1]>ema_s[-1] else 0.0,
                   "adx_strength":1.0 if adx_v[-1]>25 and pdi[-1]>mdi[-1] else 0.5,
                   "rsi_zone":1.0 if 40<rsi_v[-1]<70 else 0.0,
                   "rsi_divergence":0.5,"bb_position":0.7,
                   "breakout":min(1.0,abs(mom)),"session_quality":1.0,"spread_clear":1.0}
        else:
            comps={"trend_200":1.0 if cur<ema_t[-1] else 0.0,
                   "ma_cross":1.0 if ema_f[-1]<ema_s[-1] else 0.0,
                   "adx_strength":1.0 if adx_v[-1]>25 and mdi[-1]>pdi[-1] else 0.5,
                   "rsi_zone":1.0 if 30<rsi_v[-1]<60 else 0.0,
                   "rsi_divergence":0.5,"bb_position":0.7,
                   "breakout":min(1.0,abs(mom)),"session_quality":1.0,"spread_clear":1.0}
        score=self.scorer.score(sym,comps)
        if score<self.cfg.min_ai_score+self.cfg.momentum_min_score_boost: return None
        p=INSTRUMENT_PROFILES.get(sym,{}); sl_p=p.get("sl_pips",5)
        tp_p=p.get("tp_pips",12); pip_sz=p.get("pip",0.0001); pip_val=p.get("pip_val",10.0)
        entry=cur
        if direction=="BUY": sl=entry-sl_p*pip_sz; tp=entry+tp_p*pip_sz
        else:                sl=entry+sl_p*pip_sz; tp=entry-tp_p*pip_sz
        lot=round(np.clip(account.balance*0.005/(sl_p*pip_val),0.01,10.0),2)
        cn=trade.chain_count+1
        return TradeSignal(symbol=sym,direction=direction,
            entry_price=round(entry,5),stop_loss=round(sl,5),take_profit=round(tp,5),
            lot_size=lot,confidence=score,
            reason=f"⛓️ Chain #{cn} | Mom:{mom:.2f} | AI:{score:.0%}",
            timestamp=datetime.utcnow(),is_chain_entry=True,chain_count=cn,
            signal_components=comps)

class NexusAIScalperEA:
    def __init__(self,config=None):
        self.cfg=config or EAConfig()
        self.ind=IndicatorEngine(); self.scorer=AISignalScorer()
        self.risk=RiskManager(self.cfg); self.session=SessionFilter(self.cfg)
        self.chain=MomentumChainEngine(self.cfg,self.scorer,self.ind)
        self._stats={}

    def analyze(self,market,account,utc_now=None):
        utc_now=utc_now or datetime.utcnow()
        sym=market.symbol; profile=INSTRUMENT_PROFILES.get(sym,{})
        cls=profile.get("class","forex")
        ok,reason=self.risk.can_trade(account)
        if not ok: return None
        sess_ok,sess_name=self.session.check(utc_now,cls)
        if not sess_ok: return None
        spread_clear=self.risk.spread_ok(market)
        c=market.close; h=market.high; l=market.low
        ema_f=self.ind.ema(c,self.cfg.ma_fast); ema_s=self.ind.ema(c,self.cfg.ma_slow)
        ema_t=self.ind.ema(c,self.cfg.ma_trend); rsi_v=self.ind.rsi(c,self.cfg.rsi_period)
        bb_u,bb_m,bb_l=self.ind.bollinger(c,self.cfg.bb_period,self.cfg.bb_std)
        atr_v=self.ind.atr(h,l,c); adx_v,pdi,mdi=self.ind.adx(h,l,c,self.cfg.adx_period)
        rsi_div=self.ind.rsi_divergence(c,rsi_v)
        bo=self.ind.range_breakout(h,l,c); mom=self.ind.momentum_strength(c)
        cur=c[-1]; avg_atr=np.mean(atr_v[-20:]) if len(atr_v)>=20 else atr_v[-1]
        if atr_v[-1]>avg_atr*self.cfg.atr_spike_mult: return None
        overlap=self.session.is_overlap(utc_now); sess_q=1.0 if overlap else 0.75
        bull=dict(trend_200=1.0 if cur>ema_t[-1] else 0.0,
                  ma_cross=1.0 if ema_f[-1]>ema_s[-1] else 0.0,
                  adx_strength=1.0 if adx_v[-1]>self.cfg.adx_min and pdi[-1]>mdi[-1] else 0.5 if adx_v[-1]>20 else 0.0,
                  rsi_zone=1.0 if self.cfg.rsi_os<rsi_v[-1]<55 else 0.0,
                  rsi_divergence=1.0 if rsi_div=="bull" else 0.0,
                  bb_position=1.0 if bb_l[-1]<cur<bb_m[-1] else 0.3,
                  breakout=1.0 if bo["type"]=="bull" else 0.6 if mom>0.3 else 0.0,
                  session_quality=sess_q,spread_clear=1.0 if spread_clear else 0.0)
        bear=dict(trend_200=1.0 if cur<ema_t[-1] else 0.0,
                  ma_cross=1.0 if ema_f[-1]<ema_s[-1] else 0.0,
                  adx_strength=1.0 if adx_v[-1]>self.cfg.adx_min and mdi[-1]>pdi[-1] else 0.5 if adx_v[-1]>20 else 0.0,
                  rsi_zone=1.0 if 45<rsi_v[-1]<self.cfg.rsi_ob else 0.0,
                  rsi_divergence=1.0 if rsi_div=="bear" else 0.0,
                  bb_position=1.0 if bb_m[-1]<cur<bb_u[-1] else 0.3,
                  breakout=1.0 if bo["type"]=="bear" else 0.6 if mom<-0.3 else 0.0,
                  session_quality=sess_q,spread_clear=1.0 if spread_clear else 0.0)
        bs=self.scorer.score(sym,bull); ss=self.scorer.score(sym,bear)
        if bs>=ss and bs>=self.cfg.min_ai_score: direction,score,comps="BUY",bs,bull
        elif ss>bs and ss>=self.cfg.min_ai_score: direction,score,comps="SELL",ss,bear
        else: return None
        corr=self.risk.corr_exposure(sym,account.open_positions)
        if corr>=self.cfg.max_correlated_pairs: return None
        pip_sz=profile.get("pip",0.0001); pip_val=profile.get("pip_val",10.0)
        sl_pips=profile.get("sl_pips",5); tp_pips=profile.get("tp_pips",12)
        entry=cur
        if direction=="BUY": sl=entry-sl_pips*pip_sz; tp=entry+tp_pips*pip_sz
        else:                sl=entry+sl_pips*pip_sz; tp=entry-tp_pips*pip_sz
        lot=self.risk.lot_size(account.balance,sl_pips,pip_val)
        parts=[f"{direction}",f"AI:{score:.0%}",f"{sess_name}"]
        if comps.get("trend_200",0)>0.8: parts.append("200EMA✓")
        if comps.get("adx_strength",0)>0.8: parts.append("ADX✓")
        if rsi_div!="none": parts.append(f"RSI-Div:{rsi_div}")
        if bo.get("type")!="none": parts.append(f"BO:{bo['type'].upper()}")
        return TradeSignal(symbol=sym,direction=direction,
            entry_price=round(entry,5),stop_loss=round(sl,5),take_profit=round(tp,5),
            lot_size=lot,confidence=score,reason=" | ".join(parts),
            timestamp=utc_now,signal_components=comps)

    def on_trade_closed(self,trade,market,account):
        won=trade.close_reason=="TP"
        self.scorer.record(trade.symbol,trade.confidence,won,trade.is_chain_entry)
        self._update_stats(trade)
        if won: return self.chain.notify_tp(trade,market,account)
        return None

    def _update_stats(self,trade):
        sym=trade.symbol
        if sym not in self._stats:
            self._stats[sym]={"wins":0,"losses":0,"pips":0.0,"chain_wins":0,"chain_losses":0}
        s=self._stats[sym]
        if trade.close_reason=="TP":
            s["wins"]+=1; s["pips"]+=abs(trade.pnl_pips)
            if trade.is_chain_entry: s["chain_wins"]+=1
        else:
            s["losses"]+=1; s["pips"]-=abs(trade.pnl_pips)
            if trade.is_chain_entry: s["chain_losses"]+=1

    def summary(self):
        out={}
        for sym,s in self._stats.items():
            total=s["wins"]+s["losses"]
            out[sym]={"win_rate":round(s["wins"]/total*100,1) if total else 0,
                      "total_trades":total,"net_pips":round(s["pips"],1),
                      "chain_wins":s["chain_wins"],"chain_losses":s["chain_losses"],
                      "ai_weights":self.scorer.weights(sym)}
        return out
