"""
完整流程：
1) 讀檔 2) 清理 3) 特徵工程 (包含滯後/滾動統計) 4) 時序切分
5) 建 Pipeline + RandomForest 回歸 6) 評估 7) 儲存模型
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import os, joblib

# ---------- 1. 讀檔 ----------
ENC = "big5"  # 或你實際的編碼

# 如果只想改 metrics.csv，一樣也可以把 transfer_log.csv 換成絕對路徑
df_log = pd.read_csv(
    "/Users/jason/台積電實習/project_test/data/transfer_log.csv",
    parse_dates=["ts"],
    encoding=ENC
)

df_met = pd.read_csv(
    "/Users/jason/台積電實習/project_test/data/metrics.csv",
    parse_dates=["ts"],
    encoding=ENC
)
# ---------- 2. 基本清理 ----------
# 2.1 將 metrics 缺值補 0（或用前向填補：.fillna(method="ffill")）
df_met = df_met.fillna(0)

# 2.2 秒→分鐘對齊（確保兩表一致的 timestamp 粒度）
df_log["ts"] = df_log["ts"].dt.floor("min")
df_met["ts"] = df_met["ts"].dt.floor("min")

# ---------- 3. 特徵工程 ----------
# 3.1 旋轉 metrics：ts 為 index，其餘展開列 → 欄位 = Phase_Floor_指標
pivot = (
    df_met
    .set_index(["ts", "phase", "floor"])
    .unstack(["phase", "floor"])                # MultiIndex column
)
pivot.columns = [f"{p}_{f}_{m}" for m, p, f in pivot.columns]
pivot = pivot.reset_index()

# 3.2 建立 lag / rolling 特徵（以同一分鐘為基準往過去取）
# 例如：過去 5 分鐘 OHT_UT 的平均
lag_cols = [c for c in pivot.columns if "OHT_UT" in c]
for col in lag_cols:
    pivot[f"{col}_mean5"] = pivot[col].rolling(window=5, min_periods=1).mean()
    pivot[f"{col}_max5"]  = pivot[col].rolling(window=5, min_periods=1).max()

# 3.3 合併 label
data = df_log.merge(pivot, on="ts", how="left")

# 3.4 時間衍生：分鐘‑of‑day, 星期幾 (可能捕捉尖峰)
data["minute_of_day"] = data["ts"].dt.hour * 60 + data["ts"].dt.minute
data["weekday"]       = data["ts"].dt.weekday       # 0=Monday

# ---------- 4. 準備 X, y ----------
target      = "Xfer_time"
categorical = ["from_phase", "from_floor", "to_phase", "to_floor", "weekday"]
# 所有非 cat 亦非 target/timestamp 欄 → numeric features
numerical   = [c for c in data.columns 
               if c not in categorical + ["ts", target]]

X = data[categorical + numerical]
y = data[target]

# ---------- 5. 建 Pipeline ----------
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical),
    ],
    remainder="drop",
)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    oob_score=True,
    random_state=42,
)

pipe = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("rf",  rf),
    ]
)

# ---------- 6. 時序交叉驗證 & 調參 ----------
tscv = TimeSeriesSplit(n_splits=4, test_size=None, gap=0)
param_grid = {
    "rf__n_estimators": [300, 500],
    "rf__max_depth": [None, 20, 40],
    "rf__max_features": ["sqrt", 0.3],
}
gcv = GridSearchCV(
    pipe,
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2,
)
gcv.fit(X, y)

print("最佳參數:", gcv.best_params_)

# ---------- 7. 最終模型評估 ----------
best_model = gcv.best_estimator_

# 用最後一次 split 當 hold‑out：手動切
train_end  = tscv.split(X).__iter__().__next__()[1][-1]  # 取得最後驗證集最後索引
X_train, X_test = X.iloc[:train_end+1], X.iloc[train_end+1:]
y_train, y_test = y.iloc[:train_end+1], y.iloc[train_end+1:]

best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))
print("R²  :", r2_score(y_test, pred))
print("OOB R²:", best_model.named_steps["rf"].oob_score_)

# ---------- 8. 特徵重要度 ----------
importances = best_model.named_steps["rf"].feature_importances_
feat_names  = best_model.named_steps["pre"].get_feature_names_out()
fi = (
    pd.Series(importances, index=feat_names)
      .sort_values(ascending=False)
      .head(20)
)
print("Top‑20 Feature Importance:\n", fi)

# ---------- 9. 儲存模型 ----------
os.makedirs("/Users/jason/台積電實習/project_test/models", exist_ok=True)
joblib.dump(best_model,
    "/Users/jason/台積電實習/project_test/models/rf_xfer_time.pkl"
)
print("模型已儲存到 /Users/jason/台積電實習/project_test/models/rf_xfer_time.pkl")