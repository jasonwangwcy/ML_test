#!/usr/bin/env python3
# predict.py

import sys
import pandas as pd
import joblib
from pathlib import Path

# ———— 設定 ————
MODEL_PATH   = Path("/Users/jason/台積電實習/project_test/models/rf_xfer_time.pkl")
METRICS_CSV  = Path("/Users/jason/台積電實習/project_test/data/metrics.csv")
ENCODING     = "big5"   # 你的 metrics.csv 實際編碼

# ———— 特徵準備函式 ————
def prepare_features(ts_str, from_phase, from_floor, to_phase, to_floor):
    # 1. 讀取所有指標資料
    df_met = pd.read_csv(
        METRICS_CSV,
        parse_dates=["ts"],
        encoding=ENCODING
    )
    # 對齊到分鐘
    df_met["ts"] = df_met["ts"].dt.floor("min")
    # 2. pivot → 每組 (phase, floor, metric) 一欄
    pivot = (
        df_met
        .set_index(["ts", "phase", "floor"])
        .unstack(["phase", "floor"])
    )
    pivot.columns = [f"{p}_{f}_{m}" for m, p, f in pivot.columns]
    pivot = pivot.reset_index().sort_values("ts")

    # 3. lag/rolling 特徵 (過去 5 分鐘的平均與最大)
    lag_cols = [c for c in pivot.columns if "OHT_UT" in c]
    for col in lag_cols:
        pivot[f"{col}_mean5"] = pivot[col].rolling(window=5, min_periods=1).mean()
        pivot[f"{col}_max5"]  = pivot[col].rolling(window=5, min_periods=1).max()

    # 4. 找出對應 ts 的那一行
    ts = pd.to_datetime(ts_str).floor("min")
    row = pivot[pivot["ts"] == ts]
    if row.empty:
        raise ValueError(f"No metrics found for timestamp {ts}")
    feat = row.iloc[0].to_dict()

    # 5. 加上你的分類與時間衍生特徵
    feat.update({
        "from_phase":   from_phase,
        "from_floor":   from_floor,
        "to_phase":     to_phase,
        "to_floor":     to_floor,
        "minute_of_day": ts.hour * 60 + ts.minute,
        "weekday":       ts.weekday()
    })

    # 6. 回傳 DataFrame
    return pd.DataFrame([feat])


# ———— 主程式 ————
def main():
    # 參數檢查
    if len(sys.argv) != 6:
        print("Usage: python predict.py <ts> <from_phase> <from_floor> <to_phase> <to_floor>")
        print("  e.g.: python predict.py 2025-07-23T14:05 P5 1F P6 3F")
        sys.exit(1)

    ts_str, fp, ff, tp, tf = sys.argv[1:]

    # 載入模型
    model = joblib.load(MODEL_PATH)

    # 準備特徵
    X_new = prepare_features(ts_str, fp, ff, tp, tf)

    # 進行預測
    pred_sec = model.predict(X_new)[0]
    print(f"預估傳送時間 (秒)：{pred_sec:.1f}")


if __name__ == "__main__":
    main()
