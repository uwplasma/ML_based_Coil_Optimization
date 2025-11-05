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
LATENT_DIM = 64  # <-- set this to your encoder's output dimension

def parse_tfrecord_fn(example_proto):
    feature_description = {
        "coil_data": tf.io.FixedLenFeature([TOTAL_ROWS * FEATURES_PER_COIL], tf.float32),
        "coil_mask": tf.io.FixedLenFeature([MAX_COILS], tf.int64),
        "surface_data": tf.io.FixedLenFeature([TOTAL_SETS * FEATURES_PER_SET], tf.float32),
        "surface_mask": tf.io.FixedLenFeature([MAX_SETS], tf.int64),
        "coil_latent": tf.io.FixedLenFeature([LATENT_DIM], tf.float32)  # NEW
    }

    parsed = tf.io.parse_single_example(example_proto, feature_description)

    coil_data = tf.reshape(parsed["coil_data"], [TOTAL_ROWS, FEATURES_PER_COIL])
    coil_mask = parsed["coil_mask"]
    surface_data = tf.reshape(parsed["surface_data"], [TOTAL_SETS, FEATURES_PER_SET])
    surface_mask = parsed["surface_mask"]
    coil_latent = parsed["coil_latent"]  # already flat, shape: (LATENT_DIM,)

    inputs = {
        "coil_data": coil_data,      # (N+1, D)
        "coil_mask": coil_mask,      # (N,)
        "surface_data": surface_data,
        "surface_mask": surface_mask,
    }

    # Add coil_latent to the targets
    target = {
        "coil": coil_data,
        "surface": surface_data,
        "coil_latent": coil_latent
    }

    return inputs, target

def load_full_dataset(tfrecord_dir, batch_size=64, shuffle=True,
                      buffer_size=10000, repeat=False,
                      num_parallel_calls=tf.data.AUTOTUNE):
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
                       buffer_size=10000, repeat=False, num_parallel_calls=tf.data.AUTOTUNE,
                       train_frac=0.8, val_frac=0.1, max_records=None):
    """
    Load coil datasets from TFRecord files and split into train/val/test datasets.
    
    Returns: (train_ds, val_ds, test_ds)
    """
    # Step 1: List TFRecord files
    tfrecord_dir = Path(tfrecord_dir)
    all_files = sorted(tf.io.gfile.glob(str(tfrecord_dir / "*.tfrecord")))

    # Load all examples (optionally limit total count)
    raw_dataset = tf.data.TFRecordDataset(all_files)
    if max_records is not None:
        raw_dataset = raw_dataset.take(max_records)

    # Parse all examples to a list
    parsed_examples = list(raw_dataset.map(parse_tfrecord_fn))

    # Shuffle in-memory if needed
    if shuffle:
        np.random.shuffle(parsed_examples)

    total = len(parsed_examples)
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)

    train_data = parsed_examples[:n_train]
    val_data = parsed_examples[n_train:n_train + n_val]
    test_data = parsed_examples[n_train + n_val:]
    # print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")


    def wrap_dataset(data, shuffle=False):
        def gen():
            for x in data:
                yield x

        input = {"coil_data": tf.TensorSpec(shape=(None, 100), dtype=tf.float32),
                "coil_mask": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "surface_data": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                "surface_mask": tf.TensorSpec(shape=(None,), dtype=tf.float32)}
        target = {'coil': tf.TensorSpec(shape=(None, 100), dtype=tf.float32), 
                  'surface': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                  "coil_latent": tf.TensorSpec(shape=(None,), dtype=tf.float32)}
        
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                input,
                target
            )
        )

        if shuffle:
            ds = ds.shuffle(buffer_size=len(data))
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


    return wrap_dataset(train_data, shuffle=True), wrap_dataset(val_data), wrap_dataset(test_data)

    
