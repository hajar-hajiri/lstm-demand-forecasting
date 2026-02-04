import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils import ensure_dir, load_json
from src.data.make_dataset import load_raw, add_calendar_features, filter_short_series
from src.data.windowing import make_windows
from src.models.baselines import naive_last_value, moving_average
from src.models.lgbm_baseline import predict_multi_horizon

def mae(y, yhat): return np.mean(np.abs(y - yhat))
def rmse(y, yhat): return np.sqrt(np.mean((y - yhat) ** 2))
def smape(y, yhat):
    denom = (np.abs(y) + np.abs(yhat) + 1e-8)
    return 2.0 * np.mean(np.abs(y - yhat) / denom)

def _pack_inputs(batch, use_calendar):
    X = {"sales_seq": batch["sales_seq"], "store_id": batch["store_id"], "item_id": batch["item_id"]}
    if use_calendar:
        X["cal_seq"] = batch["cal_seq"]
    return X

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    meta = load_json(cfg["artifacts"]["meta_path"])
    model = tf.keras.models.load_model(cfg["artifacts"]["model_path"])

    df = load_raw(cfg["data"]["raw_csv"])
    df = add_calendar_features(df)
    df = filter_short_series(df, min_days=cfg["data"]["min_history_days"])

    pack = make_windows(
        df,
        lookback=meta["lookback"],
        horizon=meta["horizon"],
        train_end=cfg["split"]["train_end"],
        val_end=cfg["split"]["val_end"],
        test_end=cfg["split"]["test_end"],
        use_calendar=meta["use_calendar"],
    )

    # ✅ Save RAW sequences BEFORE scaling (for baselines + fair LightGBM)
    train_sales_seq_raw = pack["train"]["sales_seq"].copy()
    test_sales_seq_raw = pack["test"]["sales_seq"].copy()

    # Scale sales inputs with stored scaler (for LSTM)
    mean = float(meta["sales_scaler_mean"])
    scale = float(meta["sales_scaler_scale"])
    def scale_sales(x): return (x - mean) / (scale + 1e-12)

    for split in ["train", "val", "test"]:
        pack[split]["sales_seq"] = scale_sales(pack[split]["sales_seq"])

    test = pack["test"]
    X_test = _pack_inputs(test, meta["use_calendar"])
    y_true = test["y"].astype(np.float32)

    # LSTM (trained on scaled inputs)
    y_pred = model.predict(X_test, batch_size=2048, verbose=0)

    # ✅ Baselines on RAW (real sales scale)
    y_naive = naive_last_value(test_sales_seq_raw, meta["horizon"])
    y_ma7 = moving_average(test_sales_seq_raw, meta["horizon"], window=7)

    # ✅ LightGBM baseline on RAW too (more interpretable + consistent)
    y_lgbm = predict_multi_horizon(
        train_sales_seq=train_sales_seq_raw,
        train_y=pack["train"]["y"].astype(np.float32),
        test_sales_seq=test_sales_seq_raw,
        horizon=meta["horizon"],
    )

    print("=== Test metrics (lower is better) ===")
    for name, yhat in [
        ("Naive(last)", y_naive),
        ("MA(7)", y_ma7),
        ("LightGBM", y_lgbm),
        ("LSTM", y_pred),
    ]:
        print(f"{name:12s}  MAE={mae(y_true,yhat):.3f}  RMSE={rmse(y_true,yhat):.3f}  sMAPE={smape(y_true,yhat):.3f}")

    # Figure propre : comparaison sur un exemple
    ensure_dir("reports/figures")
    i = np.random.randint(0, len(y_true))
    plt.figure()
    plt.plot(range(meta["horizon"]), y_true[i], marker="o", label="true")
    plt.plot(range(meta["horizon"]), y_pred[i], marker="o", label="lstm")
    plt.plot(range(meta["horizon"]), y_naive[i], marker="o", label="naive")
    plt.plot(range(meta["horizon"]), y_ma7[i], marker="o", label="ma(7)")
    plt.plot(range(meta["horizon"]), y_lgbm[i], marker="o", label="lightgbm")
    plt.title("Forecast comparison (next 7 days)")
    plt.xlabel("horizon day")
    plt.ylabel("sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/forecast_comparison.png")
    plt.close()

    print("Saved: reports/figures/forecast_comparison.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
