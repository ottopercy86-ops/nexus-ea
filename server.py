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
from fastapi.responses import HTMLResponse
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
        "balance": 10000.0, "equity": 10000.0,
        "free_margin": 10000.0, "margin": 0.0,
        "daily_pnl": 0.0, "leverage": 500,
        "currency": "USD", "server": "Demo",
        "name": "Demo Account", "login": "Ready",
        "open_trades": 0,
    },
    "positions": [], "closed_today": [],
    "equity_curve": [], "stats": {},
    "signals": [], "logs": [],
    "config": {
        "pairs": ea_cfg.active_pairs,
        "risk_pct": ea_cfg.risk_per_trade_pct,
        "min_ai_score": ea_cfg.min_ai_score,
        "chain_enabled": ea_cfg.momentum_chain_enabled,
        "max_chain": ea_cfg.momentum_max_chain,
    }
}

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
auth = HTTPBearer()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="mobile-web-app-capable" content="yes">
<title>NEXUS AI SCALPER</title>
<style>
:root{--bg:#020409;--s1:#060c14;--s2:#0a1520;--s3:#0f1e2e;--border:#152030;--border2:#1e3348;--cyan:#00e5ff;--green:#00ff94;--red:#ff2d55;--gold:#ffd500;--purple:#b44fff;--txt:#a8c4d8;--txt2:#5a7a8e;--white:#e8f4ff;}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--txt);font-family:'Courier New',monospace;font-size:13px;display:flex;flex-direction:column}
#login{position:fixed;inset:0;z-index:500;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;background:var(--bg)}
.lt{font-size:10px;letter-spacing:5px;color:var(--cyan);opacity:.7;margin-bottom:6px}
.ln{font-family:Arial,sans-serif;font-size:36px;font-weight:900;color:var(--white);letter-spacing:3px;margin-bottom:2px}
.ls{font-size:10px;letter-spacing:4px;color:var(--txt2);margin-bottom:44px}
.lc{width:100%;max-width:340px;background:var(--s2);border:1px solid var(--border2);border-radius:12px;padding:28px 22px}
.fl{font-size:9px;letter-spacing:3px;color:var(--txt2);text-transform:uppercase;margin-bottom:7px}
.fg{margin-bottom:18px}
input{width:100%;background:var(--s1);border:1px solid var(--border);border-radius:7px;color:var(--white);font-family:'Courier New',monospace;font-size:13px;padding:11px 13px;outline:none}
input:focus{border-color:var(--cyan)}
.bc{width:100%;padding:13px;background:var(--cyan);color:#000;font-size:15px;font-weight:900;letter-spacing:3px;border:none;border-radius:8px;cursor:pointer;text-transform:uppercase}
#le{color:var(--red);font-size:11px;text-align:center;margin-top:10px;display:none}
#app{display:none;flex-direction:column;height:100vh;overflow:hidden}
.hdr{flex-shrink:0;display:flex;align-items:center;justify-content:space-between;padding:10px 14px 8px;border-bottom:1px solid var(--border);background:rgba(2,4,9,.97)}
.bn{font-size:16px;font-weight:900;color:var(--white);letter-spacing:2px}
.bv{font-size:9px;color:var(--txt2);letter-spacing:2px}
.pulse{width:8px;height:8px;border-radius:50%;background:var(--txt2);transition:all .3s}
.pulse.on{background:var(--green);box-shadow:0 0 10px var(--green)}
.xs{padding:6px 12px;border:1px solid var(--border2);background:transparent;color:var(--txt);font-size:11px;border-radius:6px;cursor:pointer}
.tabs{flex-shrink:0;display:flex;overflow-x:auto;scrollbar-width:none;border-bottom:1px solid var(--border);background:var(--s1);padding:0 6px}
.tabs::-webkit-scrollbar{display:none}
.tab{padding:10px 16px;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--txt2);cursor:pointer;white-space:nowrap;border-bottom:2px solid transparent}
.tab.act{color:var(--cyan);border-bottom-color:var(--cyan)}
.scroll{flex:1;overflow-y:auto;padding:12px}
.panel{display:none}
.panel.act{display:block}
.card{background:var(--s2);border:1px solid var(--border);border-radius:11px;padding:14px;margin-bottom:11px}
.chd{font-size:9px;letter-spacing:2.5px;color:var(--txt2);text-transform:uppercase;margin-bottom:12px}
.cr{display:flex;gap:9px;margin-bottom:12px}
.cb{flex:1;padding:13px 8px;border:none;border-radius:9px;font-size:14px;font-weight:900;letter-spacing:2px;text-transform:uppercase;cursor:pointer}
.cg{background:var(--green);color:#001a0a}
.cr2{background:var(--red);color:#fff}
.cb:disabled{opacity:.35;cursor:not-allowed}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:11px}
.sc{background:var(--s3);border:1px solid var(--border);border-radius:9px;padding:11px}
.sl{font-size:8px;letter-spacing:2px;color:var(--txt2);text-transform:uppercase;margin-bottom:5px}
.sv{font-size:20px;font-weight:600;color:var(--white)}
.sv.c{color:var(--cyan)}.sv.g{color:var(--green)}.sv.r{color:var(--red)}
.pos{background:var(--s3);border:1px solid var(--border);border-radius:9px;padding:11px;margin-bottom:8px;display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center}
.pb{padding:4px 9px;border-radius:5px;font-size:10px;font-weight:700;letter-spacing:1px}
.pb.buy{background:rgba(0,255,148,.12);color:var(--green);border:1px solid rgba(0,255,148,.25)}
.pb.sell{background:rgba(255,45,85,.12);color:var(--red);border:1px solid rgba(255,45,85,.25)}
.ps{font-size:15px;font-weight:600;color:var(--white)}
.pd{font-size:10px;color:var(--txt2)}
.pp{font-size:15px;font-weight:700;text-align:right}
.pp.g{color:var(--green)}.pp.r{color:var(--red)}
.sig{background:var(--s3);border-left:3px solid var(--cyan);border-radius:9px;padding:11px;margin-bottom:8px}
.sig.sell{border-left-color:var(--red)}
.sig.chain{border-left-color:var(--purple)}
.st{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.ss{font-size:14px;font-weight:700;color:var(--white)}
.sa{font-size:10px;padding:3px 8px;border-radius:20px;background:rgba(0,229,255,.1);color:var(--cyan);border:1px solid rgba(0,229,255,.2)}
.sp{display:flex;gap:14px;font-size:11px;color:var(--txt2);margin-bottom:5px}
.sr{font-size:10px;color:var(--txt2);line-height:1.4}
.li{display:flex;gap:8px;padding:6px 0;border-bottom:1px solid rgba(21,32,48,.5)}
.lt2{color:var(--txt2);font-size:10px;flex-shrink:0;width:52px}
.lm{font-size:11px;color:var(--txt);line-height:1.4;word-break:break-word}
.hi{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid rgba(21,32,48,.8)}
.hs{font-size:12px;font-weight:600;color:var(--white);width:60px}
.hr{font-size:11px;font-weight:700;flex:1}
.hr.tp{color:var(--green)}.hr.sl{color:var(--red)}
.hp{font-size:11px;font-weight:600;text-align:right}
.hp.g{color:var(--green)}.hp.r{color:var(--red)}
.ht{font-size:9px;color:var(--txt2);width:45px;text-align:right}
.empty{text-align:center;padding:28px 16px;color:var(--txt2);font-size:11px}
canvas{display:block;width:100%}
#toast{position:fixed;bottom:80px;left:50%;transform:translateX(-50%) translateY(20px);background:var(--s2);border:1px solid var(--cyan);border-radius:9px;padding:10px 20px;font-size:12px;color:var(--cyan);z-index:9999;opacity:0;transition:all .25s;pointer-events:none;white-space:nowrap}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.bnav{flex-shrink:0;display:flex;border-top:1px solid var(--border);background:rgba(2,4,9,.97)}
.bni{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:9px 4px;cursor:pointer;gap:3px}
.bni-i{font-size:20px}
.bni-l{font-size:9px;letter-spacing:1px;color:var(--txt2);text-transform:uppercase}
.bni.act .bni-l{color:var(--cyan)}
</style>
</head>
<body>
<div id="login">
  <div class="lt">// INITIALIZING</div>
  <div class="ln">NEXUS</div>
  <div class="ls">AI SCALPER CONTROL</div>
  <div class="lc">
    <div class="fg"><div class="fl">Server URL</div>
      <input id="lu" type="url" placeholder="https://nexus-ea.onrender.com"></div>
    <div class="fg"><div class="fl">Access Token</div>
      <input id="lt" type="password" placeholder="your token"></div>
    <button class="bc" onclick="doLogin()">CONNECT TO BOT</button>
    <div id="le"></div>
  </div>
</div>
<div id="toast"></div>
<div id="app">
  <div class="hdr">
    <div>
      <div class="bn">NEXUS EA</div>
      <div class="bv" id="hs">OFFLINE</div>
    </div>
    <div style="display:flex;align-items:center;gap:8px">
      <div class="pulse" id="pulse"></div>
      <button class="xs" onclick="doLogout()">EXIT</button>
    </div>
  </div>
  <div class="tabs">
    <div class="tab act" onclick="tab('dash',this)">DASHBOARD</div>
    <div class="tab" onclick="tab('pos',this)">POSITIONS</div>
    <div class="tab" onclick="tab('sig',this)">SIGNALS</div>
    <div class="tab" onclick="tab('hist',this)">HISTORY</div>
    <div class="tab" onclick="tab('log',this)">LOGS</div>
  </div>
  <div class="scroll" id="scroll">
    <div class="panel act" id="p-dash">
      <div class="card">
        <div class="chd">BOT CONTROL</div>
        <div class="cr">
          <button class="cb cg" id="bs" onclick="startBot()">▶ START</button>
          <button class="cb cr2" id="bst" onclick="stopBot()" disabled>■ STOP</button>
        </div>
        <div id="bm" style="font-size:11px;color:var(--txt2)">Connect your MT5 account to begin</div>
      </div>
      <div class="sg">
        <div class="sc"><div class="sl">BALANCE</div><div class="sv c" id="db">—</div></div>
        <div class="sc"><div class="sl">EQUITY</div><div class="sv" id="de">—</div></div>
        <div class="sc"><div class="sl">OPEN TRADES</div><div class="sv c" id="dot">0</div></div>
        <div class="sc"><div class="sl">DAY P&L</div><div class="sv" id="dp">$0.00</div></div>
      </div>
      <div class="card">
        <div class="chd">EQUITY CURVE</div>
        <canvas id="ec" height="110"></canvas>
      </div>
      <div class="card">
        <div class="chd">ACCOUNT</div>
        <div style="display:flex;flex-direction:column;gap:7px;font-size:12px">
          <div style="display:flex;justify-content:space-between"><span style="color:var(--txt2)">Server</span><span id="as">—</span></div>
          <div style="display:flex;justify-content:space-between"><span style="color:var(--txt2)">Broker</span><span id="ab" style="color:var(--cyan)">—</span></div>
          <div style="display:flex;justify-content:space-between"><span style="color:var(--txt2)">Balance</span><span id="abal">—</span></div>
          <div style="display:flex;justify-content:space-between"><span style="color:var(--txt2)">Leverage</span><span id="al">—</span></div>
        </div>
      </div>
    </div>
    <div class="panel" id="p-pos">
      <div class="card">
        <div class="chd">OPEN POSITIONS</div>
        <div id="pl"><div class="empty">No open positions</div></div>
      </div>
    </div>
    <div class="panel" id="p-sig">
      <div class="card">
        <div class="chd">AI SIGNALS</div>
        <div id="sl2"><div class="empty">Waiting for signals...</div></div>
      </div>
    </div>
    <div class="panel" id="p-hist">
      <div class="card">
        <div class="chd">TODAY'S TRADES</div>
        <div id="hl"><div class="empty">No trades today</div></div>
      </div>
    </div>
    <div class="panel" id="p-log">
      <div class="card">
        <div class="chd">ACTIVITY LOG</div>
        <div id="ll"><div class="empty">No logs yet</div></div>
      </div>
    </div>
  </div>
  <div class="bnav">
    <div class="bni act" onclick="tab('dash',null,this)"><div class="bni-i">⬡</div><div class="bni-l">Home</div></div>
    <div class="bni" onclick="tab('pos',null,this)"><div class="bni-i">📊</div><div class="bni-l">Trades</div></div>
    <div class="bni" onclick="tab('sig',null,this)"><div class="bni-i">📡</div><div class="bni-l">Signals</div></div>
    <div class="bni" onclick="tab('hist',null,this)"><div class="bni-i">📋</div><div class="bni-l">History</div></div>
    <div class="bni" onclick="tab('log',null,this)"><div class="bni-i">🔧</div><div class="bni-l">Logs</div></div>
  </div>
</div>
<script>
let cfg={url:'',token:''};let eq=[];let pt=null;
function doLogin(){
  const url=document.getElementById('lu').value.trim().replace(/\/$/,'');
  const token=document.getElementById('lt').value.trim();
  if(!url||!token){showErr('Enter URL and token');return}
  showErr('');
  const btn=document.querySelector('.bc');btn.textContent='CONNECTING...';btn.disabled=true;
  fetch(url+'/api/state',{headers:{Authorization:'Bearer '+token}})
    .then(r=>{if(!r.ok)throw new Error('HTTP '+r.status);return r.json()})
    .then(d=>{
      cfg={url,token};localStorage.setItem('nx',JSON.stringify(cfg));
      document.getElementById('login').style.display='none';
      document.getElementById('app').style.display='flex';
      applyState(d);pt=setInterval(refresh,10000);toast('🟢 Connected!');
    })
    .catch(e=>{showErr('Failed: '+e.message)})
    .finally(()=>{btn.textContent='CONNECT TO BOT';btn.disabled=false});
}
function doLogout(){clearInterval(pt);cfg={url:'',token:''};localStorage.removeItem('nx');document.getElementById('app').style.display='none';document.getElementById('login').style.display='flex';}
function showErr(m){const e=document.getElementById('le');e.textContent=m;e.style.display=m?'block':'none'}
window.addEventListener('load',()=>{const s=localStorage.getItem('nx');if(s){const c=JSON.parse(s);document.getElementById('lu').value=c.url;document.getElementById('lt').value=c.token}});
function refresh(){fetch(cfg.url+'/api/state',{headers:{Authorization:'Bearer '+cfg.token}}).then(r=>r.json()).then(applyState).catch(()=>{})}
function applyState(d){
  const ac=d.account||{};
  setText('db',fmt(ac.balance));setText('de',fmt(ac.equity));
  setText('dot',ac.open_trades||0);
  const pnl=ac.daily_pnl||0;const pe=document.getElementById('dp');
  pe.textContent=(pnl>=0?'+':'')+fmt(pnl);pe.className='sv '+(pnl>=0?'g':'r');
  setText('as',ac.server||'—');setText('ab',d.active_broker||'—');
  setText('abal',fmt(ac.balance)+' '+(ac.currency||'USD'));
  setText('al',ac.leverage?'1:'+ac.leverage:'—');
  setText('hs',d.connected?'LIVE':'OFFLINE');
  document.getElementById('pulse').classList.toggle('on',!!d.connected);
  const run=d.running;
  document.getElementById('bs').disabled=run;document.getElementById('bst').disabled=!run;
  document.getElementById('bm').textContent=run?'EA is active — monitoring all pairs':'Press START to begin trading';
  renderPos(d.positions||[]);renderSigs(d.signals||[]);renderHist(d.closed_today||[]);renderLogs(d.logs||[]);
  if(d.equity_curve&&d.equity_curve.length>1){eq=d.equity_curve;drawChart()}
}
function renderPos(positions){
  const c=document.getElementById('pl');
  if(!positions.length){c.innerHTML='<div class="empty">No open positions</div>';return}
  c.innerHTML=positions.map(p=>`<div class="pos"><span class="pb ${p.direction.toLowerCase()}">${p.direction}</span><div><div class="ps">${p.symbol}</div><div class="pd">${p.lot} lot @ ${p.entry}</div></div><div class="pp ${(p.profit||0)>=0?'g':'r'}">${(p.profit||0)>=0?'+':''}$${Math.abs(p.profit||0).toFixed(2)}</div></div>`).join('')
}
function renderSigs(sigs){
  const c=document.getElementById('sl2');
  if(!sigs.length){c.innerHTML='<div class="empty">Waiting for signals...</div>';return}
  c.innerHTML=sigs.map(s=>`<div class="sig ${s.chain?'chain':s.direction.toLowerCase()}"><div class="st"><span class="ss">${s.symbol} ${s.chain?'⛓️':''}</span><span class="sa">${s.ai}% AI</span></div><div class="sp"><span>Entry:${s.entry}</span><span>SL:${s.sl}</span><span>TP:${s.tp}</span></div><div class="sr">${s.reason||''}</div></div>`).join('')
}
function renderHist(trades){
  const c=document.getElementById('hl');
  if(!trades.length){c.innerHTML='<div class="empty">No trades today</div>';return}
  c.innerHTML=trades.map(t=>`<div class="hi"><div class="hs">${t.symbol}</div><div class="hr ${t.result==='TP'?'tp':'sl'}">${t.result==='TP'?'✅ TP':'❌ SL'}</div><div class="hp ${(t.pips||0)>=0?'g':'r'}">${(t.pips||0)>=0?'+':''}${t.pips||0}p</div><div class="ht">${t.time||''}</div></div>`).join('')
}
function renderLogs(logs){
  const c=document.getElementById('ll');
  if(!logs.length){c.innerHTML='<div class="empty">No logs yet</div>';return}
  c.innerHTML=logs.slice(0,50).map(l=>`<div class="li"><span class="lt2">${l.t}</span><span class="lm">${l.msg}</span></div>`).join('')
}
function drawChart(){
  const canvas=document.getElementById('ec');if(!canvas||eq.length<2)return;
  const dpr=window.devicePixelRatio||1;const W=canvas.parentElement.clientWidth-28;const H=110;
  canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.width=W+'px';canvas.style.height=H+'px';
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const vals=eq.map(e=>e.v);const min=Math.min(...vals)*.9998;const max=Math.max(...vals)*1.0002;
  const rng=max-min||1;const xStep=W/(vals.length-1);const yFor=v=>H-((v-min)/rng)*(H-20)-10;
  const isUp=vals[vals.length-1]>=vals[0];const col=isUp?'#00ff94':'#ff2d55';
  const gr=ctx.createLinearGradient(0,0,0,H);
  gr.addColorStop(0,isUp?'rgba(0,255,148,.2)':'rgba(255,45,85,.2)');gr.addColorStop(1,'rgba(0,0,0,0)');
  ctx.beginPath();ctx.moveTo(0,yFor(vals[0]));vals.forEach((v,i)=>ctx.lineTo(i*xStep,yFor(v)));
  ctx.lineTo((vals.length-1)*xStep,H);ctx.lineTo(0,H);ctx.closePath();ctx.fillStyle=gr;ctx.fill();
  ctx.beginPath();ctx.moveTo(0,yFor(vals[0]));vals.forEach((v,i)=>ctx.lineTo(i*xStep,yFor(v)));
  ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke();
}
async function startBot(){
  try{await api('/api/start','POST');toast('🚀 EA started');setTimeout(refresh,1500)}catch(e){toast('❌ '+e.message)}
}
async function stopBot(){
  try{await api('/api/stop','POST');toast('⏹️ EA stopped');setTimeout(refresh,1500)}catch(e){toast('❌ '+e.message)}
}
async function api(path,method='GET',body=null){
  const r=await fetch(cfg.url+path,{method,headers:{Authorization:'Bearer '+cfg.token,'Content-Type':'application/json'},body:body?JSON.stringify(body):null});
  if(!r.ok)throw new Error('HTTP '+r.status);return r.json();
}
function tab(name,tabEl,navEl){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('act'));
  document.getElementById('p-'+name).classList.add('act');
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('act'));
  document.querySelectorAll('.bni').forEach(n=>n.classList.remove('act'));
  if(tabEl)tabEl.classList.add('act');if(navEl)navEl.classList.add('act');
  document.getElementById('scroll').scrollTop=0;
  if(name==='dash')setTimeout(drawChart,50);
}
function fmt(n){if(n==null)return'—';return parseFloat(n).toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2})}
function setText(id,v){const e=document.getElementById(id);if(e)e.textContent=v}
let tt;function toast(m){const e=document.getElementById('toast');e.textContent=m;e.classList.add('show');clearTimeout(tt);tt=setTimeout(()=>e.classList.remove('show'),3000)}
window.addEventListener('resize',drawChart);
</script>
</body>
</html>"""


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


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=DASHBOARD_HTML)


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