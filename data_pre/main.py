import os
import re
import glob
import pandas as pd

# 這四種 metric 名稱要跟你檔案開頭一致
METRICS = ['OHT_UT', 'JAM', 'Carousel_下貨失敗', 'Carousel_UT']

def parse_filename(fn: str):
    """
    從檔名 "OHT_UT_P5_1F.csv" 解析出
      metric = "OHT_UT"
      phase  = "P5"
      floor  = "1F"
    若格式不符，回傳 None
    """
    base = os.path.basename(fn)
    name, _ = os.path.splitext(base)
    m = re.match(rf"^({'|'.join(METRICS)})_([^_]+)_([^_]+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def load_metric_df(path: str):
    """
    讀 CSV、parse datetime，回傳 ts-indexed DataFrame
    """
    df = pd.read_csv(
        path,
        parse_dates=['detectTime'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M')
    )
    metric = parse_filename(path)[0]
    df = df.rename(columns={'detectTime':'ts', metric:metric})
    df['ts'] = df['ts'].dt.floor('T')
    return df.set_index('ts')


def merge_group(input_dir: str, phase: str, floor: str, paths: dict, out_dir: str):
    """
    phase, floor 各一組；paths: metric → 檔案路徑
    把四張表合併、補 0、加上 phase/floor，輸出 CSV
    """
    dfs = []
    for m in METRICS:
        if m not in paths:
            raise ValueError(f"[{phase},{floor}] 少了 {m} 的 CSV")
        dfs.append(load_metric_df(paths[m]))
    merged = pd.concat(dfs, axis=1)

    # 建立完整 index
    start = merged.index.min().floor('D')
    end   = merged.index.max().ceil('D') - pd.Timedelta(minutes=1)
    full_idx = pd.date_range(start=start, end=end, freq='T')

    merged = merged.reindex(full_idx, fill_value=0)
    merged.index.name = 'ts'
    merged = merged.reset_index()
    merged['phase'] = phase
    merged['floor'] = floor

    cols = ['ts','phase','floor'] + METRICS
    out_path = os.path.join(out_dir, f"merged_{phase}_{floor}.csv")
    merged[cols].to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Output: {out_path}")


def main():
    input_dir = "./data"      # 你的 CSV 放這裡
    output_dir = "./merged"   # 輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 掃所有 CSV
    files = glob.glob(os.path.join(input_dir, "*.csv"))

    # group_files[(phase,floor)] = { metric: path, ... }
    group_files = {}
    for fn in files:
        parsed = parse_filename(fn)
        if not parsed:
            print("跳過不符規則：", fn)
            continue
        metric, phase, floor = parsed
        group_files.setdefault((phase, floor), {})[metric] = fn

    # 對每一組做合併
    for (phase, floor), paths in group_files.items():
        try:
            merge_group(input_dir, phase, floor, paths, output_dir)
        except ValueError as e:
            print("錯誤：", e)

if __name__ == "__main__":
    main()
