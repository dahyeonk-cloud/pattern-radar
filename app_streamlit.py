
# US Pattern Radar — 15/30/60/240m (Patched)
# How to run locally:
#   pip install -r requirements.txt
#   streamlit run app_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="🚀미장 진입 패턴 레이더 (15/30/60/240m)", layout="wide")

# -------------------- Utilities
ET = pytz.timezone("America/New_York")
KST = pytz.timezone("Asia/Seoul")

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()

def zscore(x, win=50):
    r = x.rolling(win)
    return (x - r.mean()) / (r.std(ddof=0) + 1e-9)

def body(df):
    return (df["Close"] - df["Open"]).abs()

def candle_range(df):
    return (df["High"] - df["Low"]).abs()

def slim_mask(df, k_body=0.6, k_range=0.7, look=50):
    med_b = body(df).rolling(look).median()
    med_r = candle_range(df).rolling(look).median()
    b_ok = body(df) < k_body * (med_b + 1e-9)
    r_ok = candle_range(df) < k_range * (med_r + 1e-9)
    return (b_ok & r_ok)

TF_PARAMS = {
    15:  dict(look=50, vz=50, obv_slope=5, comp_len=18, comp_ratio=0.70, k4=0.035, trig_z=1.0, spike_z=2.0),
    30:  dict(look=50, vz=50, obv_slope=6, comp_len=18, comp_ratio=0.70, k4=0.030, trig_z=1.2, spike_z=2.2),
    60:  dict(look=60, vz=60, obv_slope=8, comp_len=20, comp_ratio=0.70, k4=0.028, trig_z=1.3, spike_z=2.4),
    240: dict(look=80, vz=80, obv_slope=10,comp_len=24, comp_ratio=0.70, k4=0.025, trig_z=1.5, spike_z=2.6),
}

TF_WEIGHTS = {15:0.20, 30:0.35, 60:0.35, 240:0.10}

def resample_ohlcv(df_5m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    rule = f"{tf_min}T"
    o = df_5m['Open'].resample(rule).first().rename('Open')
    h = df_5m['High'].resample(rule).max().rename('High')
    l = df_5m['Low'].resample(rule).min().rename('Low')
    c = df_5m['Close'].resample(rule).last().rename('Close')
    v = df_5m['Volume'].resample(rule).sum().rename('Volume')
    out = pd.concat([o, h, l, c, v], axis=1).dropna()
    return out

def score_pack(df: pd.DataFrame, tf: int):
    p = TF_PARAMS[tf]
    ema5, ema20 = ema(df.Close,5), ema(df.Close,20)
    v_z = zscore(df.Volume, p['vz'])
    obv_line = obv(df.Close, df.Volume)
    obv_up = obv_line > ema(obv_line, 10)

    # ① Shake-out
    med_b = body(df).rolling(p['look']).median()
    cond_down = (df.Close.shift(1)<df.Open.shift(1)) & (body(df).shift(1) > 1.5*med_b.shift(1)) & (v_z.shift(1) > 1.0)
    cond_up   = (df.Close>df.Open) & (body(df) > 1.2*med_b) & (v_z > 1.0)
    flat_seq  = slim_mask(df, look=p['look']).rolling(6).mean() >= 0.6
    obv_slope = obv_line.diff(p['obv_slope']) > 0
    S1 = (cond_down.astype(int)*30 + cond_up.astype(int)*40 + flat_seq.astype(int)*15 + obv_slope.astype(int)*15).clip(0,100)

    # ② Compression
    slim_ratio = slim_mask(df, look=p['look']).rolling(p['comp_len']).mean()
    in_band = ((df.Close.between(ema5, ema20)) | (df.Close.between(ema20, ema5))).rolling(p['comp_len']).mean()
    S2 = (slim_ratio*100*0.5 + in_band*100*0.3 + (-v_z.clip(upper=0))*20).fillna(0).clip(0,100)

    # ③ Controlled
    mid = ema(df.Close, 20)
    band = (df.Close - mid).abs().rolling(30).std()
    band_norm = band / (df.Close.rolling(30).mean()+1e-9)
    obv_flat = obv_line.diff(10).abs() < obv_line.rolling(50).std()*0.3
    S3 = ((1 - band_norm.clip(0,1))*60 + (-v_z.clip(upper=0))*20 + obv_flat.astype(int)*20).fillna(0).clip(0,100)

    # ④ Triggering
    S4 = (v_z.clip(lower=0)*30 + obv_up.astype(int)*40 + (df.Close > ema20).astype(int)*30).fillna(0).clip(0,100)

    # ⑤ Distribution
    upper = (df.High - df[['Open','Close']].max(axis=1)).clip(lower=0)
    long_upper = upper > (candle_range(df)*0.5)
    long_ratio = long_upper.rolling(20).mean()
    ema5_down = ema5 < ema5.shift(1)
    S5 = (long_ratio*100*0.5 + (~obv_up).astype(int)*30 + ema5_down.astype(int)*20).fillna(0).clip(0,100)

    # ⑥ Liquidity Test
    slim = slim_mask(df, look=max(30, p['look']-10))
    choppy = (df.Close.diff().abs() < df.Close.rolling(30).std()*0.3).rolling(12).mean()
    S6 = (slim.rolling(12).mean()*60 + (v_z.clip(upper=0).abs())*20 + choppy*20).fillna(0).clip(0,100)

    scores = pd.DataFrame({
        "① Shakeout":S1,"② Compression":S2,"③ Controlled":S3,
        "④ Trigger":S4,"⑤ Distribution":S5,"⑥ LiquidityTest":S6
    }, index=df.index)
    winner = scores.idxmax(axis=1)
    conf = scores.max(axis=1)
    return scores, winner, conf

def classify_multi_tf_from_5m(df_5m: pd.DataFrame):
    pack = {}
    for tf in [15,30,60,240]:
        dft = resample_ohlcv(df_5m, tf)
        scores, winner, conf = score_pack(dft, tf)
        pack[tf] = dict(df=dft, scores=scores, winner=winner, conf=conf)

    idx = pack[30]['scores'].index.intersection(pack[60]['scores'].index).intersection(pack[15]['scores'].index).intersection(pack[240]['scores'].index)
    if len(idx) == 0:
        raise ValueError("데이터가 부족하거나 기간이 짧습니다. 기간(days)을 늘리거나 다른 심볼로 시도하세요.")
    latest = idx[-1]
    label_space = pack[30]['scores'].columns
    agg = {lab:0.0 for lab in label_space}
    for tf,w in TF_WEIGHTS.items():
        for lab in label_space:
            agg[lab] += pack[tf]['scores'].loc[latest, lab] * w
    final_label = max(agg, key=agg.get)
    final_conf = agg[final_label]

    hard_exit = (pack[30]['winner'].loc[latest]=="⑤ Distribution" or pack[60]['winner'].loc[latest]=="⑤ Distribution")
    long_ready = (pack[30]['winner'].loc[latest] in ["① Shakeout","② Compression","③ Controlled"]) and \
                 (pack[60]['winner'].loc[latest] in ["① Shakeout","② Compression","③ Controlled"])
    trigger_ok = (final_label=="④ Trigger") and all([pack[tf]['scores'].loc[latest,"④ Trigger"] >= TF_PARAMS[tf]['trig_z']*20 for tf in [30,60]])

    if hard_exit: action = "청산/재진입 금지 (⑤ 분배 신호 우세)"
    elif trigger_ok: action = "추세 진입(단기) — OBV 동반 확인 후 규모 확대"
    elif long_ready: action = "상방 준비 — 거래량 임계 상회 시 분할 진입"
    elif final_label=="⑥ LiquidityTest": action = "관망 — 상위 프레임 확정까지 대기"
    else: action = "조건 미충족 — 관망"

    result = {
        "timestamp": str(latest),
        "final_pattern": final_label,
        "final_confidence": round(float(final_conf), 1),
        "tf_winners": {str(tf): pack[tf]['winner'].loc[latest] for tf in [15,30,60,240]},
        "tf_confidences": {str(tf): round(float(pack[tf]['conf'].loc[latest]),1) for tf in [15,30,60,240]},
        "action": action,
        "frames": [15,30,60,240],
    }
    return result, pack

def fetch_yf_5m(symbol: str, days: int = 30) -> pd.DataFrame:
    period = f\"{days}d\"
    df = yf.download(symbol, period=period, interval=\"5m\", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Ensure timezone-aware index
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is None:
        idx = idx.tz_localize('UTC')
    df.index = idx
    df = df.rename(columns={\"Open\":\"Open\",\"High\":\"High\",\"Low\":\"Low\",\"Close\":\"Close\",\"Volume\":\"Volume\"})
    # Regular session filter
    idx_et = df.index.tz_convert(ET)
    mask = (idx_et.time >= dt.time(9,30)) & (idx_et.time <= dt.time(16,0))
    df = df.loc[mask].copy()
    # Convert to KST for display
    df = df.tz_convert(KST)
    df = df[[\"Open\",\"High\",\"Low\",\"Close\",\"Volume\"]].dropna()
    if len(df) < 200:
        return pd.DataFrame()
    return df

def plot_panel(ax_price, ax_vol, dft: pd.DataFrame, title: str):
    ax_price.plot(dft.index, dft[\"Close\"], label=\"Close\")
    ax_price.plot(dft.index, ema(dft[\"Close\"],5), label=\"EMA5\", linewidth=1)
    ax_price.plot(dft.index, ema(dft[\"Close\"],20), label=\"EMA20\", linewidth=1)
    ax_price.set_title(title)
    ax_price.legend(loc=\"upper left\", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    ax_vol.bar(dft.index, dft[\"Volume\"])
    ax_vol.set_ylabel(\"Vol\")
    ax_vol.grid(True, alpha=0.2)

# -------------------- UI
st.title(\"📘 세력 행동 패턴 레이더 — US Market (15/30/60/240m)\")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    symbol = st.text_input(\"심볼(Symbol)\", \"TSLA\").upper().strip()
with col2:
    days = st.slider(\"가져올 기간(일)\", min_value=10, max_value=120, value=60, step=10)
with col3:
    tail_n = st.slider(\"차트 표시 봉 수(프레임별)\", min_value=100, max_value=400, value=200, step=50)

st.caption(\"※ 데이터는 yfinance(지연/제한) 기반 데모입니다. 실전은 Polygon/Alpaca로 교체 권장.\")

if st.button(\"분석 실행\"):
    with st.spinner(\"데이터 수집 및 패턴 판독 중...\"):
        df5 = fetch_yf_5m(symbol, days)
        if df5 is None or df5.empty:
            st.error(\"데이터가 부족합니다. 기간(days)을 늘리거나, TSLA/AAPL/NVDA 같은 티커로 다시 시도해 보세요.\")
        else:
            try:
                result, pack = classify_multi_tf_from_5m(df5)
            except Exception as e:
                st.exception(e)
                st.stop()

            st.success(f\"[{symbol}] {result['final_pattern']} | 신뢰도 {result['final_confidence']:.1f} | 권장: {result['action']}\")
            st.json(result)

            for tf in [15,30,60,240]:
                dft = pack[tf]['df'].tail(tail_n)
                st.subheader(f\"🕒 {tf}분 — {pack[tf]['winner'].iloc[-1]}  (conf {pack[tf]['conf'].iloc[-1]:.1f})\")
                fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, height_ratios=[3,1])
                plot_panel(ax_price, ax_vol, dft, title=f\"{symbol} — {tf}m\")
                st.pyplot(fig)
                plt.close(fig)

            st.caption(\"Tip: 30m·60m가 같은 방향(①②③ 계열 or ⑤)일 때 신호를 더 강하게 보세요. ④ 단독 급등은 OBV 동반 여부 반드시 확인.\")
