import tensorflow as tf
from tensorflow.keras import layers as L

def build_model(
    lookback: int,
    horizon: int,
    calendar_dim: int,
    n_stores: int,
    n_items: int,
    emb_dim: int = 8,
    lstm_units: int = 64,
    dropout: float = 0.2,
):
    in_sales = L.Input(shape=(lookback, 1), name="sales_seq")
    inputs = [in_sales]
    x_parts = [in_sales]

    if calendar_dim and calendar_dim > 0:
        in_cal = L.Input(shape=(lookback, calendar_dim), name="cal_seq")
        inputs.append(in_cal)
        x_parts.append(in_cal)

    in_store = L.Input(shape=(1,), dtype="int32", name="store_id")
    in_item = L.Input(shape=(1,), dtype="int32", name="item_id")
    inputs += [in_store, in_item]

    store_emb = L.Embedding(input_dim=n_stores, output_dim=emb_dim)(in_store)
    item_emb = L.Embedding(input_dim=n_items, output_dim=emb_dim)(in_item)

    store_emb = L.Reshape((emb_dim,))(store_emb)
    item_emb = L.Reshape((emb_dim,))(item_emb)

    store_rep = L.RepeatVector(lookback)(store_emb)
    item_rep = L.RepeatVector(lookback)(item_emb)

    x_parts += [store_rep, item_rep]

    x = L.Concatenate(axis=-1)(x_parts)
    x = L.LSTM(lstm_units, return_sequences=False)(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(128, activation="relu")(x)
    out = L.Dense(horizon, name="y")(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model
