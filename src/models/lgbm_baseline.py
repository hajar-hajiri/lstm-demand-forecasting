import numpy as np
from lightgbm import LGBMRegressor

def make_lag_features(sales_seq: np.ndarray) -> np.ndarray:
    x = sales_seq[:, :, 0]
    last = x[:, -1]
    mean7 = x[:, -7:].mean(axis=1)
    mean14 = x[:, -14:].mean(axis=1) if x.shape[1] >= 14 else x.mean(axis=1)
    std7 = x[:, -7:].std(axis=1)
    trend = x[:, -1] - x[:, -7] if x.shape[1] >= 7 else x[:, -1] - x[:, 0]
    return np.stack([last, mean7, mean14, std7, trend], axis=1)

def train_lgbm_one_step(X_train_feat: np.ndarray, y_train_1d: np.ndarray) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train_feat, y_train_1d)
    return model

def predict_multi_horizon(
    train_sales_seq: np.ndarray,
    train_y: np.ndarray,
    test_sales_seq: np.ndarray,
    horizon: int,
) -> np.ndarray:
    Xtr = make_lag_features(train_sales_seq)
    Xte = make_lag_features(test_sales_seq)

    preds = []
    for h in range(horizon):
        m = train_lgbm_one_step(Xtr, train_y[:, h])
        preds.append(m.predict(Xte))
    return np.stack(preds, axis=1)
