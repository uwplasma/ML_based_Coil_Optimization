import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Import your loader constants/shapes ---
from surface_coil_loader import (
    load_full_dataset,  # only used for shape sanity if you want
)
# If you prefer no dependency on load_full_dataset, copy the constants below to match your TFRecords:
MAX_COILS = 6
FEATURES_PER_COIL = 100
TOTAL_ROWS = MAX_COILS  # N + 1 (last row often used for scalar token)
# -------------------------------

def parse_tfrecord_fn(example_proto):
    """
    Minimal parser matching surface_coil_loader.py to access coil_data & coil_mask.
    """
    feature_description = {
        "ID": tf.io.FixedLenFeature([], tf.string),
        "coil_data": tf.io.FixedLenFeature([TOTAL_ROWS * FEATURES_PER_COIL], tf.float32),
        "coil_mask": tf.io.FixedLenFeature([MAX_COILS], tf.int64),
        "surface_data": tf.io.FixedLenFeature([(441 + 1) * 4], tf.float32),  # not used but present
        "surface_mask": tf.io.FixedLenFeature([441], tf.int64),
        # If your TFRecords include an ID, you can add: "ID": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    coil_data = tf.reshape(parsed["coil_data"], [TOTAL_ROWS, FEATURES_PER_COIL])
    coil_mask = parsed["coil_mask"]  # (MAX_COILS,)
    return coil_data, coil_mask

def make_dataset(tfrecord_dir, batch_size=512, shuffle=False):
    files = tf.io.gfile.glob(str(Path(tfrecord_dir) / "*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No .tfrecord files found in: {tfrecord_dir}")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def sample_coil_currents(
    tfrecord_dir: str,
    current_col: int = 0,
    include_global_scalar: bool = False,
    max_samples: int | None = 200_000,
    batch_size: int = 1024,
    return_dataframe: bool = True,
):
    """
    Extracts a large sample of per-coil current scalars (and optional global scalar)
    across all TFRecords in `tfrecord_dir`.

    Args:
        current_col: which feature index within each coil row corresponds to current.
        include_global_scalar: also collect coil_data[-1, 0] per example (global scalar row).
        max_samples: cap total collected samples to control memory.
        return_dataframe: return a pandas DataFrame (else a 1D numpy array of currents).

    Returns:
        DataFrame with columns ["current"] (+ "global_scalar" if requested), or a numpy vector.
    """
    ds = make_dataset(tfrecord_dir, batch_size=batch_size, shuffle=False)

    per_coil_currents = []
    # global_scalars = [] if include_global_scalar else None

    collected = 0
    cap = max_samples if max_samples is not None else np.inf

    for coil_data_b, coil_mask_b in ds:
        # coil_data_b: (B, TOTAL_ROWS, FEATURES_PER_COIL)
        # coil_mask_b: (B, MAX_COILS)

        # Slice the N real coil rows (exclude last row) and take current_col
        coils_b = coil_data_b[:, :MAX_COILS, current_col]  # (B, MAX_COILS)
        mask_b = tf.cast(coil_mask_b, tf.bool)            # (B, MAX_COILS)

        # Apply mask to keep only valid coils
        valid_vals = tf.boolean_mask(coils_b, mask_b)     # (num_valid_in_batch,)

        # Append to list
        vals_np = valid_vals.numpy()
        per_coil_currents.append(vals_np)
        collected += vals_np.shape[0]

        # if include_global_scalar:
        #     # Take the scalar token at last row, first feature (index 0)
        #     # Mirrors scaler_loss usage: squared_error[:, -1, 0]
        #     gs = coil_data_b[:, -1, 0].numpy()  # (B,)
        #     global_scalars.append(gs)

        if collected >= cap:
            break

    # Concatenate
    all_currents = np.concatenate(per_coil_currents, axis=0)
    if max_samples is not None and all_currents.shape[0] > max_samples:
        # Downsample without replacement for a representative subset
        idx = np.random.choice(all_currents.shape[0], size=max_samples, replace=False)
        all_currents = all_currents[idx]

    if return_dataframe:
        df = pd.DataFrame({"current": all_currents})
        # if include_global_scalar and global_scalars:
        #     gs_all = np.concatenate(global_scalars, axis=0)
        #     if max_samples is not None and gs_all.shape[0] > max_samples:
        #         gs_all = gs_all[:max_samples]  # align simple case
        #     # If lengths mismatch (rare due to masking), align by min length
        #     n = min(len(df), len(gs_all))
        #     df = df.iloc[:n].copy()
        #     df["global_scalar"] = gs_all[:n]
        return df
    else:
        return all_currents

if __name__ == "__main__":
    # === Example usage ===
    tfrecord_dir = "surface_coil_tfrecords"  # change to your directory
    df = sample_coil_currents(
        tfrecord_dir,
        current_col=0,                 # <-- change if your current is another column
        include_global_scalar=True,    # also pull the last-row scalar if you store one
        max_samples=250_000,
        batch_size=2048,
        return_dataframe=True,
    )

    print(df.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))
    # Quick answers to: "are they all the same?"
    nunique = df["current"].nunique()
    print(f"\nUnique per-coil current values: {nunique}")
    if nunique <= 5:
        print("Distinct values:", sorted(df['current'].unique().tolist()))
