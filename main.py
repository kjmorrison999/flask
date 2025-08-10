# ==== COPY START eth_signal_bot.py ====
import requests, time, io, datetime as dt
import matplotlib.pyplot as plt
import numpy as np

# ---------- CONFIG (already filled) ----------
TELEGRAM_BOT_TOKEN = "8206522218:AAFOSHt78rZvbYB3LxG-2eWCT8iTuXrlpKM"
TELEGRAM_CHAT_ID   = "8012994854"
CHECK_INTERVAL_SEC = 15 * 60            # run every 15 minutes
CONFIDENCE_THRESHOLD = 75
PAIR = "ETHUSDT"                         # perps on Binance futures
SPOT_SYMBOL = "ETHBTC"                   # for ETH/BTC relative strength

BINANCE_F = "https://fapi.binance.com"
BINANCE_S = "https://api.binance.com"

# ---------- Telegram ----------
def tg_send_text(msg:str):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=20)
    except Exception: pass

def tg_send_photo(caption:str, png_bytes:bytes):
    try:
        files = {"photo": ("img.png", png_bytes)}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                      data=data, files=files, timeout=30)
    except Exception: pass

def now_utc():
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ---------- Data fetchers ----------
def get_price():
    r = requests.get(f"{BINANCE_F}/fapi/v1/ticker/price", params={"symbol": PAIR}, timeout=15).json()
    return float(r["price"])

def get_klines_futures(interval="5m", limit=36):
    r = requests.get(f"{BINANCE_F}/fapi/v1/klines",
                     params={"symbol": PAIR, "interval": interval, "limit": limit}, timeout=20).json()
    return [{"close": float(k[4]), "vol": float(k[5]), "taker_buy_base": float(k[13])} for k in r]

def get_klines_spot(interval="5m", limit=36):
    r = requests.get(f"{BINANCE_S}/api/v3/klines",
                     params={"symbol": "ETHUSDT", "interval": interval, "limit": limit}, timeout=20).json()
    # index 9 = taker buy base on spot
    return [{"close": float(k[4]), "vol": float(k[5]), "taker_buy_base": float(k[9])} for k in r]

def get_ethbtc_series():
    r = requests.get(f"{BINANCE_S}/api/v3/klines",
                     params={"symbol": SPOT_SYMBOL, "interval": "15m", "limit": 96}, timeout=20).json()
    return [float(k[4]) for k in r]

def get_open_interest_hist():
    r = requests.get(f"{BINANCE_F}/futures/data/openInterestHist",
                     params={"symbol": PAIR, "period": "5m", "limit": 2}, timeout=20).json()
    if isinstance(r, list) and len(r) >= 2:
        return float(r[-1]["sumOpenInterest"]), float(r[-2]["sumOpenInterest"])
    return None, None

def get_funding():
    r = requests.get(f"{BINANCE_F}/fapi/v1/fundingRate",
                     params={"symbol": PAIR, "limit": 1}, timeout=15).json()
    return float(r[0]["fundingRate"])

def get_long_short_ratio():
    r = requests.get(f"{BINANCE_F}/futures/data/topLongShortAccountRatio",
                     params={"symbol": PAIR, "period": "5m", "limit": 1}, timeout=20).json()
    return float(r[0]["longShortRatio"])

def get_depth_delta_1pct():
    depth = requests.get(f"{BINANCE_F}/fapi/v1/depth",
                         params={"symbol": PAIR, "limit": 1000}, timeout=15).json()
    best_bid = float(depth["bids"][0][0]); best_ask = float(depth["asks"][0][0])
    mid = (best_bid + best_ask)/2.0
    lo, hi = mid*0.99, mid*1.01
    buy = sum(float(p)*float(q) for p,q in depth["bids"] if lo <= float(p) <= hi)
    sell= sum(float(p)*float(q) for p,q in depth["asks"] if lo <= float(p) <= hi)
    return buy - sell

def get_liquidations(limit=200):
    r = requests.get(f"{BINANCE_F}/fapi/v1/allForceOrders",
                     params={"symbol": PAIR, "limit": limit}, timeout=20).json()
    liqs = []
    for x in r:
        try:
            price = float(x.get("avgPrice") or x.get("price") or 0)
            qty   = float(x.get("executedQty", "0"))
            if price>0 and qty>0: liqs.append((price, qty))
        except Exception: pass
    return liqs

def get_btc_dominance_and_total3():
    g = requests.get("https://api.coingecko.com/api/v3/global", timeout=20).json()
    total = g["data"]["total_market_cap"]["usd"]
    btc_p = g["data"]["market_cap_percentage"]["btc"]/100.0
    eth_p = g["data"]["market_cap_percentage"]["eth"]/100.0
    btc_dom = btc_p*100.0
    total3 = total * (1.0 - btc_p - eth_p)
    return btc_dom, total3

# ---------- Calculations ----------
def cvd_from_klines(kl):
    cvd = 0.0; series = []
    for k in kl:
        buy = k["taker_buy_base"]; sell = max(k["vol"] - buy, 0.0)
        cvd += (buy - sell); series.append(cvd)
    return series

def slope(vals):
    if len(vals) < 3: return 0.0
    x = np.arange(len(vals)); y = np.array(vals, dtype=float)
    xm, ym = x.mean(), y.mean()
    num = ((x-xm)*(y-ym)).sum(); den = ((x-xm)**2).sum() or 1.0
    return float(num/den)

def liq_heatmap_png(liqs):
    buf = io.BytesIO()
    if not liqs:
        plt.figure(figsize=(6,2), dpi=140); plt.text(0.5,0.5,"No recent liqs", ha="center", va="center")
        plt.axis("off"); plt.tight_layout(); plt.savefig(buf, format="png"); buf.seek(0); return buf
    prices = [p for p,_ in liqs]; qtys = [q for _,q in liqs]
    lo,hi = min(prices), max(prices); bins = max(20, min(80, len(prices)//3 or 20))
    hist, edges = np.histogram(prices, bins=bins, weights=qtys, range=(lo,hi))
    heat = hist.reshape(1,-1)
    plt.close("all"); fig,ax = plt.subplots(figsize=(7,1.8), dpi=140)
    ax.imshow(heat, aspect="auto", origin="lower", extent=[edges[0], edges[-1], 0, 1])
    ax.set_yticks([]); ax.set_xlabel("Price (USD) — liquidation density")
    plt.tight_layout(); plt.savefig(buf, format="png"); buf.seek(0); return buf

# ---------- Scoring & signal ----------
def analyze_and_score():
    price = get_price()

    spot = get_klines_spot("5m", 36)
    futs = get_klines_futures("5m", 36)
    spot_cvd = cvd_from_klines(spot)
    fut_cvd  = cvd_from_klines(futs)
    spot_tr  = slope(spot_cvd[-6:])     # last ~30m
    fut_tr   = slope(fut_cvd[-6:])

    oi_curr, oi_prev = get_open_interest_hist()
    oi_up = (oi_curr is not None and oi_prev is not None and oi_curr > oi_prev)

    funding = get_funding()
    lsr = get_long_short_ratio()
    ob_delta = get_depth_delta_1pct()

    btc_dom, total3 = get_btc_dominance_and_total3()
    ethbtc = get_ethbtc_series()
    ethbtc_up = (len(ethbtc) >= 5 and ethbtc[-1] > ethbtc[-5])

    liqs = get_liquidations(200)

    long_s = short_s = 0
    long_s += 15 if spot_tr > 0 else 0; short_s += 15 if spot_tr <= 0 else 0
    long_s += 15 if fut_tr  > 0 else 0; short_s += 15 if fut_tr  <= 0 else 0
    if price >= 4250 and oi_up: long_s += 15
    if price < 4250 and not oi_up: short_s += 10
    if -0.0002 <= funding <= 0.0004: long_s += 8; short_s += 8
    if lsr > 1.2: short_s += 6
    if lsr < 0.85: long_s += 6
    if ob_delta > 0: long_s += 10
    else: short_s += 10
    if btc_dom > 58.5: short_s += 8
    else: long_s += 5
    if ethbtc_up: long_s += 8
    else: short_s += 5

    bias = "Long" if long_s >= short_s else "Short"
    confidence = int(min(100, max(long_s, short_s)))

    recent_high = max(k["close"] for k in futs[-18:])
    recent_low  = min(k["close"] for k in futs[-18:])
    if bias == "Long":
        entry = (round(max(recent_low, price*0.985),2), round(price*0.995,2))
        sl  = round(entry[0]*0.99, 2)
        tp1 = round(price*1.015, 2); tp2 = round(min(recent_high*1.01, price*1.035), 2)
    else:
        entry = (round(price*1.005,2), round(min(recent_high, price*1.015),2))
        sl  = round(entry[1]*1.01, 2)
        tp1 = round(price*0.985, 2); tp2 = round(max(recent_low*0.99, price*0.965), 2)

    return {
        "timestamp": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "price": price, "bias": bias, "confidence": confidence,
        "spot_trend": spot_tr, "fut_trend": fut_tr, "oi_up": oi_up,
        "funding": funding, "lsr": lsr, "ob_delta": ob_delta,
        "btc_dom": btc_dom, "ethbtc_up": ethbtc_up,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "liqs": liqs
    }

def sleep_to_next_quarter():
    now = dt.datetime.utcnow()
    mins = ((now.minute // 15) + 1) * 15
    if mins >= 60:
        nxt = now.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    else:
        nxt = now.replace(minute=mins, second=0, microsecond=0)
    time.sleep(max(5, (nxt - now).total_seconds()))

def main():
    # Startup: send test + heatmap
    try:
        img = liq_heatmap_png(get_liquidations(120))
        tg_send_photo("✅ Bot started — waiting for the next 15-minute interval.", img.getvalue())
    except Exception:
        tg_send_text("✅ Bot started — waiting for the next 15-minute interval.")
    sleep_to_next_quarter()

    last_day = None
    daily = []

    while True:
        try:
            s = analyze_and_score()
            daily.append(s)

            if s["confidence"] >= CONFIDENCE_THRESHOLD:
                img = liq_heatmap_png(s["liqs"])
                why = (f"SpotCVD:{'up' if s['spot_trend']>0 else 'down'}, "
                       f"FutCVD:{'up' if s['fut_trend']>0 else 'down'}, "
                       f"OI:{'up' if s['oi_up'] else 'down'}, "
                       f"Funding:{s['funding']:.5f}, L/S:{s['lsr']:.2f}, "
                       f"±1%Δ:{'buy' if s['ob_delta']>0 else 'sell'}, "
                       f"BTC.D:{s['btc_dom']:.1f}")
                msg = (f"📈 ETH Signal ({s['timestamp']} UTC)\n"
                       f"Price: {s['price']:.2f}\nBias: {s['bias']}\n"
                       f"Confidence: {s['confidence']}%\n"
                       f"Entry: {s['entry'][0]} – {s['entry'][1]}\n"
                       f"SL: {s['sl']}\nTP1/TP2: {s['tp1']} / {s['tp2']}\n"
                       f"Why: {why}")
                tg_send_photo(msg, img.getvalue())

            # daily summary at 23:59 UTC
            now = dt.datetime.utcnow()
            if (last_day != now.date()) and (now.hour == 23 and now.minute >= 59):
                if daily:
                    lines = [f"{x['timestamp']}  {x['bias']}  {x['confidence']}%  Px {x['price']:.2f}"
                             for x in daily[-20:]]
                    tg_send_text("🧾 Daily ETH summary (last signals):\n" + "\n".join(lines))
                daily = []
                last_day = now.date()

        except Exception as e:
            tg_send_text(f"⚠️ Bot error: {repr(e)}")

        sleep_to_next_quarter()

if __name__ == "__main__":
    main()
# ==== COPY END eth_signal_bot.py ====
