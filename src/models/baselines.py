import numpy as np

def naive_last_value(sales_seq: np.ndarray, horizon: int) -> np.ndarray:
    last = sales_seq[:, -1, 0]
    return np.repeat(last[:, None], horizon, axis=1)

def moving_average(sales_seq: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    w = min(window, sales_seq.shape[1])
    avg = sales_seq[:, -w:, 0].mean(axis=1)
    return np.repeat(avg[:, None], horizon, axis=1)
