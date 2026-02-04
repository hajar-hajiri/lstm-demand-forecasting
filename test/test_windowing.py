import pandas as pd
from src.data.windowing import make_windows

def test_make_windows_shapes():
    dates = pd.date_range("2016-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": list(dates),
        "store": [1]*40,
        "item": [1]*40,
        "sales": list(range(40)),
        "dow": [d.dayofweek for d in dates],
        "month": [d.month for d in dates],
        "is_weekend": [1 if d.dayofweek >= 5 else 0 for d in dates],
        "day": [d.day for d in dates],
    })

    pack = make_windows(
        df, lookback=7, horizon=3,
        train_end="2016-02-01", val_end="2016-02-05", test_end="2016-02-10",
        use_calendar=True
    )
    tr = pack["train"]
    assert tr["sales_seq"].shape[1:] == (7, 1)
    assert tr["y"].shape[1] == 3
    assert tr["store_id"].shape[1] == 1
    assert tr["item_id"].shape[1] == 1
