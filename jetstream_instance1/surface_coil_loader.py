import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
tf.constant(1.0)  # Trigger basic op
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from tensorflow.keras import layers
import os

MAX_COILS = 6
FEATURES_PER_COIL = 100
TOTAL_ROWS = MAX_COILS + 1  # N + 1
MAX_SETS = 441
TOTAL_SETS = MAX_SETS + 1
FEATURES_PER_SET = 4
AUTOTUNE = tf.data.AUTOTUNE

def parse_tfrecord_fn(example_proto):
    feature_description = {
    "ID": tf.io.FixedLenFeature([], tf.string),
    "coil_data": tf.io.FixedLenFeature([MAX_COILS * FEATURES_PER_COIL], tf.float32),
    "coil_mask": tf.io.FixedLenFeature([MAX_COILS], tf.int64),
    'surface_data': tf.io.FixedLenFeature([TOTAL_SETS * FEATURES_PER_SET], tf.float32),
    'surface_mask': tf.io.FixedLenFeature([MAX_SETS], tf.int64)
}
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    device_id = parsed["ID"]
    coil_data = tf.reshape(parsed["coil_data"], [MAX_COILS, FEATURES_PER_COIL])
    coil_mask = parsed["coil_mask"]
    surface_data = tf.reshape(parsed['surface_data'], [TOTAL_SETS, FEATURES_PER_SET])
    surface_mask = parsed['surface_mask']

    inputs = {
        "coil_data": coil_data,         # (N+1, D)
        "coil_mask": coil_mask,          # (N,)
        "surface_data": surface_data,         # (N+1, D)
        "surface_mask": surface_mask,
    }
    target = {'coil': coil_data, 'surface': surface_data}
    return inputs, target


def load_full_dataset(tfrecord_dir, batch_size=64, shuffle=True,
                      buffer_size=50000, repeat=False,
                      num_parallel_calls=tf.data.AUTOTUNE, _type='coil'):
    """
    Loads a tf.data.Dataset from TFRecord chunks in a directory.
    
    Returns: dataset of (coil_data, coil_mask)
    """
    tfrecord_files = tf.io.gfile.glob(str(Path(tfrecord_dir) / "*.tfrecord"))

    dataset = tf.data.TFRecordDataset(tfrecord_files,
                                      num_parallel_reads=num_parallel_calls)
    
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=num_parallel_calls)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def load_split_datasets(tfrecord_dir, batch_size=64, shuffle=True,
                       shuffle_buffer_size=50000, tfrecord_buffer_bytes= 8<<20, repeat=False, 
                       num_parallel_calls=AUTOTUNE,
                       train_frac=0.8, val_frac=0.1, max_records=None, seed=73):
    """
    Load coil datasets from TFRecord files and split into train/val/test datasets.
    
    Returns: (train_ds, val_ds, test_ds)
    """
    # Step 1: List TFRecord files
    # tfrecord_dir = Path(tfrecord_dir)
    # all_files = sorted(tf.io.gfile.glob(str(tfrecord_dir / "*.tfrecord")))

    # # Load all examples (optionally limit total count)
    # ds = tf.data.TFRecordDataset(all_files, buffer_size=buffer_bytes, 
    #                                       num_parallel_reads=num_parallel_calls)
    # if max_records is not None:
    #     ds = ds.take(max_records)

    # # Parse all examples to a list
    # # parsed_examples = list(raw_dataset.map(parse_tfrecord_fn))
    # ds = ds.map(parse_tfrecord_fn, num_parallel_calls=num_parallel_calls)

    # # Shuffle in-memory if needed
    # # if shuffle:
    # #     np.random.shuffle(parsed_examples)

    # total = len(ds)
    # n_train = int(total * train_frac)
    # n_val = int(total * val_frac)

    # train_data = ds[:n_train]
    # val_data = ds[n_train:n_train + n_val]
    # test_data = ds[n_train + n_val:]

    # def wrap_dataset(data, shuffle=False):
    #     def gen():
    #         for x in data:
    #             yield x

    #     input = {"coil_data": tf.TensorSpec(shape=(None, 100), dtype=tf.float32),
    #             "coil_mask": tf.TensorSpec(shape=(None,), dtype=tf.float32),
    #             "surface_data": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    #             "surface_mask": tf.TensorSpec(shape=(None,), dtype=tf.float32)}
    #     target = {'coil': tf.TensorSpec(shape=(None, 100), dtype=tf.float32), 
    #               'surface': tf.TensorSpec(shape=(None, 4), dtype=tf.float32)}
        
    #     ds = tf.data.Dataset.from_generator(
    #         gen,
    #         output_signature=(
    #             input,
    #             target
    #         )
    #     )

    #     if shuffle:
    #         ds = ds.shuffle(buffer_size=len(data))
    #     if repeat:
    #         ds = ds.repeat()
    #     ds = ds.batch(batch_size)
    #     ds = ds.prefetch(tf.data.AUTOTUNE)
    #     return ds

    tfrecord_dir = Path(tfrecord_dir)
    files = sorted(tf.io.gfile.glob(str(tfrecord_dir / "*.tfrecord")))
    if not files:
        raise FileNotFoundError(f"No TFRecord files in {tfrecord_dir}")

    # Base dataset: buffered TFRecord reads (fast path)
    base = tf.data.TFRecordDataset(
        files,
        buffer_size=tfrecord_buffer_bytes,
        num_parallel_reads=AUTOTUNE
    )

    if max_records is not None:
        base = base.take(int(max_records))

    # Parse → (features, targets)
    def parse_example(raw):
        # TODO: keep your existing parse_tfrecord_fn body here
        return parse_tfrecord_fn(raw)

    parsed = base.map(parse_example, num_parallel_calls=num_parallel_calls)

    # ---- COUNT EXAMPLES SAFELY (no len(ds)) ----
    # This does one linear pass to count examples.
    total = int(
        parsed.reduce(tf.constant(0, dtype=tf.int64),
                      lambda x, _: x + 1).numpy()
    )
    if total == 0:
        raise ValueError("Dataset is empty after parsing.")

    n_train = int(total * train_frac)
    n_val   = int(total * val_frac)
    n_test  = total - n_train - n_val

    # Re-create parsed pipeline for slicing (cheaper than caching huge sets)
    base = tf.data.TFRecordDataset(
        files,
        buffer_size=tfrecord_buffer_bytes,
        num_parallel_reads=AUTOTUNE
    ).map(parse_example, num_parallel_calls=num_parallel_calls)

    # Deterministic slicing: [0:n_train], [n_train:n_train+n_val], [rest]
    train_raw = base.take(n_train)
    val_raw   = base.skip(n_train).take(n_val)
    test_raw  = base.skip(n_train + n_val)

    # Post-process: shuffle (bounded), repeat, batch, prefetch
    def finalize(ds, do_shuffle):
        if shuffle and do_shuffle:
            # bounded buffer prevents minute-long warmup
            ds = ds.shuffle(shuffle_buffer_size,
                            reshuffle_each_iteration=True,
                            seed=seed)
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = finalize(train_raw, True)
    val_ds   = finalize(val_raw,   False)
    test_ds  = finalize(test_raw,  False)


    return train_ds, val_ds, test_ds

    
