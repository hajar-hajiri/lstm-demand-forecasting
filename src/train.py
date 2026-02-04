import argparse
import yaml
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from src.utils import set_seed, ensure_dir, save_json
from src.data.make_dataset import load_raw, add_calendar_features, filter_short_series
from src.data.windowing import make_windows
from src.models.lstm_model import build_model

def _pack_inputs(batch, use_calendar: bool):
    X = {
        "sales_seq": batch["sales_seq"],
        "store_id": batch["store_id"],
        "item_id": batch["item_id"],
    }
    if use_calendar:
        X["cal_seq"] = batch["cal_seq"]
    return X

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    ensure_dir(cfg["artifacts"]["dir"])

    df = load_raw(cfg["data"]["raw_csv"])
    df = add_calendar_features(df)
    df = filter_short_series(df, min_days=cfg["data"]["min_history_days"])

    lookback = cfg["features"]["lookback"]
    horizon = cfg["features"]["horizon"]
    use_calendar = bool(cfg["features"]["add_calendar"])

    pack = make_windows(
        df,
        lookback=lookback,
        horizon=horizon,
        train_end=cfg["split"]["train_end"],
        val_end=cfg["split"]["val_end"],
        test_end=cfg["split"]["test_end"],
        use_calendar=use_calendar,
    )
    meta = pack["meta"]

    # Scale sales inputs (train only)
    scaler = StandardScaler()
    scaler.fit(pack["train"]["sales_seq"].reshape(-1, 1))

    def scale_sales(arr):
        shp = arr.shape
        flat = arr.reshape(-1, 1)
        flat2 = scaler.transform(flat)
        return flat2.reshape(shp)

    for split in ["train", "val", "test"]:
        if pack[split]["sales_seq"].shape[0] > 0:
            pack[split]["sales_seq"] = scale_sales(pack[split]["sales_seq"])


    model = build_model(
        lookback=lookback,
        horizon=horizon,
        calendar_dim=meta["calendar_dim"] if use_calendar else 0,
        n_stores=meta["n_stores"],
        n_items=meta["n_items"],
        emb_dim=cfg["train"]["emb_dim"],
        lstm_units=cfg["train"]["lstm_units"],
        dropout=cfg["train"]["dropout"],
    )

    opt = tf.keras.optimizers.Adam(learning_rate=cfg["train"]["lr"])
    model.compile(optimizer=opt, loss="mae")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["train"]["early_stopping_patience"],
            restore_best_weights=True,
        )
    ]

    X_train = _pack_inputs(pack["train"], use_calendar)
    y_train = pack["train"]["y"]
    X_val = _pack_inputs(pack["val"], use_calendar)
    y_val = pack["val"]["y"]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    model.save(cfg["artifacts"]["model_path"])

    meta_out = {
        "lookback": lookback,
        "horizon": horizon,
        "use_calendar": use_calendar,
        "calendar_cols": meta["calendar_cols"],
        "n_stores": meta["n_stores"],
        "n_items": meta["n_items"],
        "store2id": meta["store2id"],
        "item2id": meta["item2id"],
        "sales_scaler_mean": float(scaler.mean_[0]),
        "sales_scaler_scale": float(scaler.scale_[0]),
    }
    save_json(cfg["artifacts"]["meta_path"], meta_out)

    print(f"Saved model: {cfg['artifacts']['model_path']}")
    print(f"Saved meta:  {cfg['artifacts']['meta_path']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
