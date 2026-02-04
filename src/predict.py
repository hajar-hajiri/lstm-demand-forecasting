import argparse
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import load_json
from src.data.make_dataset import load_raw, add_calendar_features, filter_short_series

def _calendar_row(dt: pd.Timestamp):
    dow = dt.dayofweek
    month = dt.month
    is_weekend = 1 if dow >= 5 else 0
    day = dt.day
    return np.array([dow, month, is_weekend, day], dtype=np.float32)

def main(cfg_path: str, store: int, item: int, start_date: str, horizon_days: int):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    meta = load_json(cfg["artifacts"]["meta_path"])
    model = tf.keras.models.load_model(cfg["artifacts"]["model_path"])

    df = load_raw(cfg["data"]["raw_csv"])
    df = add_calendar_features(df)
    df = filter_short_series(df, min_days=cfg["data"]["min_history_days"])

    g = df[(df["store"] == store) & (df["item"] == item)].sort_values("date")
    if g.empty:
        raise ValueError("store/item not found in data")

    lookback = meta["lookback"]
    step = meta["horizon"]
    use_calendar = meta["use_calendar"]

    store2id = meta["store2id"]
    item2id = meta["item2id"]
    store_id = store2id.get(str(store), store2id.get(store))
    item_id = item2id.get(str(item), item2id.get(item))

    start_date = pd.to_datetime(start_date)
    hist = g[g["date"] < start_date].tail(lookback)
    if len(hist) < lookback:
        raise ValueError("Not enough history before start_date")

    sales_hist = hist["sales"].values.astype(np.float32).reshape(lookback, 1)

    mean = float(meta["sales_scaler_mean"])
    scale = float(meta["sales_scaler_scale"])
    def scale_sales(x): return (x - mean) / (scale + 1e-12)

    sales_hist_scaled = scale_sales(sales_hist)

    preds = []
    current_date = start_date

    for _ in range(int(np.ceil(horizon_days / step))):
        X = {
            "sales_seq": sales_hist_scaled[None, :, :],
            "store_id": np.array([[store_id]], dtype=np.int32),
            "item_id": np.array([[item_id]], dtype=np.int32),
        }
        if use_calendar:
            cal_seq = []
            hist_dates = pd.date_range(end=current_date - pd.Timedelta(days=1), periods=lookback, freq="D")
            for dt in hist_dates:
                cal_seq.append(_calendar_row(dt))
            X["cal_seq"] = np.stack(cal_seq)[None, :, :]

        yhat = model.predict(X, verbose=0)[0]  # (step,)
        preds.extend(yhat.tolist())

        new_sales = np.array(yhat[:step], dtype=np.float32).reshape(-1, 1)
        sales_hist = np.vstack([sales_hist, new_sales])[-lookback:]
        sales_hist_scaled = scale_sales(sales_hist)

        current_date += pd.Timedelta(days=step)

    preds = preds[:horizon_days]
    out_dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")
    out = pd.DataFrame({"date": out_dates, "store": store, "item": item, "forecast_sales": preds})
    out_path = "artifacts/forecast.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved forecast to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--store", type=int, required=True)
    p.add_argument("--item", type=int, required=True)
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--horizon-days", type=int, default=90)
    args = p.parse_args()
    main(args.config, args.store, args.item, args.start_date, args.horizon_days)
