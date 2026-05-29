"""Train TensorFlow TimeMixer on ETT datasets and compare against PyTorch baseline.

Uses identical data preprocessing, splits, and hyperparameters as the PT scripts.
"""
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timemixer_tf import TimeMixer, TimeMixerConfig


# ---------------------------------------------------------------------------
# ETT data loader — identical splits to PT Dataset_ETT_hour / Dataset_ETT_minute
# ---------------------------------------------------------------------------
def load_ett_data(root_path, data_path, dataset_type, seq_len, pred_len,
                  features="M", target="OT", scale=True):
    """Load ETT data with exact same splits as PT data_loader.py."""
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    scaler = StandardScaler()

    if dataset_type in ("ETTh1", "ETTh2"):
        # Hourly: 12 months * 30 days * 24 hours
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    else:  # ETTm1, ETTm2
        # 15-min: 12 * 30 * 24 * 4
        border1s = [0, 12 * 30 * 24 * 4 - seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

    if features in ("M", "MS"):
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    else:
        df_data = df_raw[[target]]

    if scale:
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
    else:
        data = df_data.values

    # Time features (matching PT timeenc=1 → timeF)
    df_stamp = df_raw[["date"]]
    df_stamp["date"] = pd.to_datetime(df_stamp.date)

    def _time_features(dates, freq="h"):
        """Match PT time_features for freq='h' (4 dims) or 't' (5 dims)."""
        dti = pd.DatetimeIndex(dates)
        feats = np.column_stack([
            dti.month,
            dti.day,
            dti.weekday,
            dti.hour,
        ])
        if freq == "t":
            feats = np.column_stack([feats, dti.minute // 15])
        return feats.astype(np.float32)

    freq = "h" if dataset_type in ("ETTh1", "ETTh2") else "t"

    def _make_dataset(flag):
        idx = {"train": 0, "val": 1, "test": 2}[flag]
        b1, b2 = border1s[idx], border2s[idx]
        d = data[b1:b2]
        stamp = df_stamp.iloc[b1:b2]
        ts = _time_features(stamp["date"].values, freq=freq)

        n = len(d) - seq_len - pred_len + 1
        # Vectorized sliding windows: shape [n, seq_len, features]
        xs = np.lib.stride_tricks.sliding_window_view(d, seq_len, axis=0)
        xs = np.swapaxes(xs, 1, 2).copy()  # [n, C, seq] -> [n, seq, C]
        ys = np.lib.stride_tricks.sliding_window_view(d[seq_len:], pred_len, axis=0)
        ys = np.swapaxes(ys, 1, 2).copy()

        xms = np.lib.stride_tricks.sliding_window_view(ts, seq_len, axis=0)
        xms = np.swapaxes(xms, 1, 2).copy()
        yms = np.lib.stride_tricks.sliding_window_view(ts[seq_len:], pred_len, axis=0)
        yms = np.swapaxes(yms, 1, 2).copy()

        return (xs[:n].astype(np.float32), ys[:n].astype(np.float32),
                xms[:n].astype(np.float32), yms[:n].astype(np.float32))

    return _make_dataset, scaler, data.shape[1]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_config(model, config, train_data, val_data, test_data, epochs):
    """Train the TF model, matching PT training semantics."""
    (x_train, y_train, xm_train, ym_train) = train_data
    (x_val, y_val, xm_val, ym_val) = val_data
    (x_test, y_test, xm_test, ym_test) = test_data

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, xm_train, y_train, ym_train)
    ).shuffle(1024).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_val, xm_val, y_val, ym_val)
    ).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, xm_test, y_test, ym_test)
    ).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    best_val = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        # Train
        train_losses = []
        for batch_x, batch_xm, batch_y, batch_ym in train_ds:
            B = tf.shape(batch_x)[0]
            with tf.GradientTape() as tape:
                pred = model(batch_x, batch_xm, training=True)
                f_dim = -1  # 'M' task
                pred = pred[:, -config.pred_len:, f_dim:]
                true = batch_y[:, -config.pred_len:, f_dim:]
                loss = loss_fn(true, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(loss.numpy())

        train_loss = np.mean(train_losses)

        # Validate
        val_losses = []
        for batch_x, batch_xm, batch_y, batch_ym in val_ds:
            pred = model(batch_x, batch_xm, training=False)
            f_dim = -1
            pred = pred[:, -config.pred_len:, f_dim:]
            true = batch_y[:, -config.pred_len:, f_dim:]
            val_losses.append(loss_fn(true, pred).numpy())
        val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_weights = [w.numpy() for w in model.trainable_variables]
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best weights and evaluate
    if best_weights:
        for w, bw in zip(model.trainable_variables, best_weights):
            w.assign(bw)

    # Test
    test_losses = []
    preds_all, trues_all = [], []
    for batch_x, batch_xm, batch_y, batch_ym in test_ds:
        pred = model(batch_x, batch_xm, training=False)
        f_dim = -1
        pred = pred[:, -config.pred_len:, f_dim:]
        true = batch_y[:, -config.pred_len:, f_dim:]
        test_losses.append(loss_fn(true, pred).numpy())
        preds_all.append(pred.numpy())
        trues_all.append(true.numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)

    mse_val = np.mean((preds_all - trues_all) ** 2)
    mae_val = np.mean(np.abs(preds_all - trues_all))

    return mse_val, mae_val


# ---------------------------------------------------------------------------
# Main: train on all ETT datasets and compare
# ---------------------------------------------------------------------------
def main():
    ETT_CONFIGS = [
        ("ETTh1", "ETTh1", 7, 128, 10, 10, 16, 32, 2),
        ("ETTh2", "ETTh2", 7, 16, 30, 15, 16, 32, 2),
        ("ETTm2", "ETTm2", 7, 128, 30, 15, 32, 32, 2),
    ]

    # PT baseline results from earlier runs
    PT_RESULTS = {
        ("ETTh1", 96): (0.3854, 0.4014),
        ("ETTh1", 192): (0.4429, 0.4304),
        ("ETTh1", 336): (0.5127, 0.4700),
        ("ETTh1", 720): (0.4925, 0.4734),
        ("ETTh2", 96): (0.2903, 0.3428),
        ("ETTh2", 192): (0.3827, 0.4015),
        ("ETTh2", 336): (0.4162, 0.4310),
        ("ETTh2", 720): (0.4192, 0.4406),
        ("ETTm1", 96): (0.3266, 0.3626),
        ("ETTm1", 192): (0.3661, 0.3840),
        ("ETTm1", 336): (0.3943, 0.4061),
        ("ETTm1", 720): (0.4525, 0.4409),
        ("ETTm2", 96): (0.1751, 0.2584),
        ("ETTm2", 192): (0.2421, 0.3023),
        ("ETTm2", 336): (0.2952, 0.3400),
        ("ETTm2", 720): (0.3932, 0.3967),
    }

    print("=" * 70)
    print("TensorFlow TimeMixer Training — ETT Benchmark")
    print("=" * 70)

    all_tf = {}
    root = "./dataset/ETT-small/"

    for dataset, data_class, enc_in, batch_size, epochs, patience, d_model, d_ff, e_layers in ETT_CONFIGS:
        print(f"\n{'='*50}")
        print(f"  {dataset}  (d_model={d_model}, batch={batch_size}, epochs={epochs})")
        print(f"{'='*50}")

        for pred_len in [96, 192, 336, 720]:
            print(f"\n--- pred_len={pred_len} ---")

            make_data, scaler, n_feat = load_ett_data(
                root, f"{dataset}.csv", data_class,
                seq_len=96, pred_len=pred_len, features="M")

            train_data = make_data("train")
            val_data = make_data("val")
            test_data = make_data("test")

            print(f"  Train: {train_data[0].shape[0]}, Val: {val_data[0].shape[0]}, Test: {test_data[0].shape[0]}")

            config = TimeMixerConfig(
                task_name="long_term_forecast",
                seq_len=96, pred_len=pred_len,
                enc_in=enc_in, c_out=enc_in,
                d_model=d_model, d_ff=d_ff, e_layers=e_layers,
                down_sampling_layers=3, down_sampling_window=2,
                channel_independence=1, use_norm=1,
                learning_rate=0.01,
                batch_size=batch_size,
                train_epochs=epochs, patience=patience,
                dropout=0.1, embed="timeF",
                freq="h" if dataset in ("ETTh1", "ETTh2") else "t",
            )

            model = TimeMixer(config)
            # Build by calling once
            dummy = tf.random.normal([1, 96, enc_in])
            _ = model(dummy, tf.random.normal([1, 96, 4]), training=False)

            mse, mae = train_one_config(model, config, train_data, val_data, test_data, epochs)
            all_tf[(dataset, pred_len)] = (mse, mae)

            pt_mse, pt_mae = PT_RESULTS.get((dataset, pred_len), (None, None))
            print(f"  TF  -> MSE={mse:.4f}, MAE={mae:.4f}")
            if pt_mse:
                delta = mse - pt_mse
                print(f"  PT  -> MSE={pt_mse:.4f}, MAE={pt_mae:.4f}  (Δ={delta:+.4f})")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: TensorFlow vs PyTorch")
    print("=" * 70)
    print(f"{'Dataset':<8} {'Pred':>5} {'TF MSE':>8} {'PT MSE':>8} {'Δ MSE':>8} {'Match':>8}")
    print("-" * 50)

    diffs = []
    for (ds, pl), (tf_mse, tf_mae) in sorted(all_tf.items()):
        pt_mse, pt_mae = PT_RESULTS.get((ds, pl), (None, None))
        if pt_mse:
            delta = tf_mse - pt_mse
            diffs.append(abs(delta))
            ok = "✓" if abs(delta) < 0.05 else "✗"
            print(f"{ds:<8} {pl:>5} {tf_mse:8.4f} {pt_mse:8.4f} {delta:+8.4f} {ok:>8}")

    if diffs:
        print(f"\nAvg absolute Δ: {np.mean(diffs):.4f} MSE")
        print(f"Max absolute Δ: {np.max(diffs):.4f} MSE")
        print(f"Models {'MATCH' if np.mean(diffs) < 0.03 else 'differ — investigate'}")

    return all_tf


if __name__ == "__main__":
    main()
