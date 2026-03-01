# %%
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
tf.constant(1.0)  # Trigger basic op
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import ray
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
tf.config.optimizer.set_jit(True)   # keep it on for A100 performance
# If startup time is a hassle, temporarily set False while debugging shape/mask issues.


# %%
%load_ext autoreload
%autoreload 2
from surface_coil_loader import load_full_dataset, load_split_datasets
from transformers import TransformerEncoder, TransformerDecoder
from CoilAutoEncoder import CoilAutoencoderModel

# %%
tfrecord_dir = Path("flat_surface_coil_tfrecords")

# %%
def build_coil_autoencoder(hp):
    encoder = TransformerEncoder(max_sets=6, features_per_set=100, name='coil', embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"], ff_dim=hp["ff_dim"], 
        num_sab_blocks=hp["sab_blocks"], dropout=hp["enc_dropout"])
    
    decoder = TransformerDecoder(name = 'coil', embed_dim=hp["embed_dim"], num_heads=hp["num_heads"], ff_dim=hp["ff_dim"],
        num_layers=hp["decoder_blocks"], max_sets=6, features_per_set=100, dropout=hp["dec_dropout"])
    
    autoencoder = CoilAutoencoderModel(
        encoder, decoder, hp["embed_dim"], hp['kl_target'], hp["kl_warmup_steps"],
        hp['kl_cap'], hp['kl_cap_warmup'], hp['kl_gamma']
    )
    return autoencoder, encoder, decoder

# %%
def build_model(hp):
    model, encoder, decoder = build_coil_autoencoder(hp)  
    total_steps = int((251_200 * 200) // hp["batch_size"])
    warmup_steps = int(total_steps * 0.04)  
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=hp["learning_rate"] / 25.0, 
                                                            decay_steps=total_steps-warmup_steps, 
                                                            warmup_target=hp['learning_rate'], 
                                                            warmup_steps=warmup_steps)
    
    optimizer = tf.keras.optimizers.Lion(learning_rate=lr_schedule, weight_decay=hp['weight_decay'], clipnorm=1.0)
    model.compile(optimizer=optimizer)
    return model, encoder, decoder

# %%
def prepare_callbacks(log_dir, model_name="coil_autoencoder"):
    return [
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint(
            filepath=os.path.join(log_dir, f"{model_name}_best.keras"),
            save_best_only=True,
            monitor='val_loss'
        )
    ]

# %%
# example_hp = {'batch_size': 512,
#  'embed_dim': 256,
#  'num_heads': 4,
#  'ff_dim': 1024,
#  'enc_dropout': 0.05,
#  'dec_dropout': 0.05,
#  'learning_rate': 9e-04,
#  'weight_decay': 5e-3,
#  'sab_blocks': 4,
#  'decoder_blocks': 4}
hp = {
  "embed_dim": 64,
  "num_heads": 8,
  "ff_dim": 768,
  "sab_blocks": 5,
  "decoder_blocks": 5,
  "enc_dropout": 0.05482322795208501,
  "dec_dropout": 0.198518591351031,
  "batch_size": 128,
  "learning_rate": 0.00012683792457697835,
  "weight_decay": 9.93354485855776e-06,
  "kl_target": 0.025,#0.10268862116710578,
  "kl_warmup_steps": 14000,
  "kl_cap": 10,#35.481023453263354,
  "kl_cap_warmup": 24000,
  "kl_gamma": 0.0007417943498667281
}

# %%
class SaveBestSubmodels(tf.keras.callbacks.Callback):
    """
    Saves encoder/decoder when `monitor` improves. Works even if `restore_best_weights=True`
    is used on the top-level model (weights are shared).
    """
    def __init__(self, encoder, decoder, log_dir, monitor="val_loss", mode="min"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.monitor = monitor
        self.log_dir = log_dir
        self.best = np.inf if mode == "min" else -np.inf
        self.cmp = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        self.save_dir = os.path.join(log_dir, "best")
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.cmp(current, self.best):
            self.best = current
            # Save submodels
            enc_path = os.path.join(self.save_dir, "encoder.best.keras")
            dec_path = os.path.join(self.save_dir, "decoder.best.keras")
            self.encoder.save(enc_path)
            self.decoder.save(dec_path)
            # Persist the score for reference
            with open(os.path.join(self.save_dir, "best_metric.json"), "w") as f:
                json.dump({self.monitor: float(self.best), "epoch": int(epoch)}, f)

# %%
def train_coil_autoencoder(
    hp, train_ds, val_ds, epochs, log_root="logs", use_wandb=False,
    # --- Early stopping knobs ---
    early_stopping=True,
    es_monitor="val_masked_recon",
    es_min_delta=0.0,
    es_patience=30,
    es_mode="min",
):
    
    def keep_coil_only(inputs, targets):
        return (
            {
                'coil_data': inputs['coil_data'],
                'coil_mask': inputs['coil_mask'],
            },
            targets['coil']
        )

    train_ds = train_ds.map(keep_coil_only, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(keep_coil_only, num_parallel_calls=tf.data.AUTOTUNE)

    model, encoder, decoder = build_model(hp)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, f"{run_id}_trial")
    os.makedirs(log_dir, exist_ok=True)

    callbacks = prepare_callbacks(log_dir)

    # EarlyStopping (global, not per-epoch) — restore the best weights seen on val
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_monitor,
                min_delta=es_min_delta,
                patience=es_patience,
                mode=es_mode,
                restore_best_weights=True
            )
        )
        # Save best encoder/decoder whenever the monitored metric improves
        callbacks.append(SaveBestSubmodels(
            encoder=encoder,
            decoder=decoder,
            log_dir=log_dir,
            monitor=es_monitor,
            mode=es_mode
        ))

    if use_wandb:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project="coil_autoencoder", config=hp)
        callbacks.append(WandbCallback(save_model=False))  # we save submodels ourselves

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Build submodels (ensures variables initialized)
    sample_inputs, _ = next(iter(train_ds))
    sample_latents = encoder(sample_inputs, training=False)
    _ = decoder(sample_latents, training=False)

    # Save "final" snapshot (early stopping may have restored best weights already)
    encoder.save(os.path.join(log_dir, "encoder.keras"))
    decoder.save(os.path.join(log_dir, "decoder.keras"))

    return history, encoder, decoder

# %%
train_ds, val_ds, test_ds = load_split_datasets(tfrecord_dir, batch_size=hp["batch_size"])

history, encoder, decoder = train_coil_autoencoder(
    hp=hp,
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=200,
    log_root="coil_logs",
    use_wandb=False
)

# %%
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.legend()
plt.show()

# %%
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.show()

# %%


# %%
batched_dataset = load_full_dataset(tfrecord_dir, batch_size = hp['batch_size'])

# %%
from transformers import SelfAttentionBlock, PoolingByMultiheadAttention
# encoder = tf.keras.models.load_model('coil_logs/20250812-152016_trial/encoder.keras', safe_mode=False)
saved_encoder = tf.keras.models.load_model(
    "coil_logs/20260211-204559_trial/encoder.keras",
    custom_objects={"SelfAttentionBlock": SelfAttentionBlock, 
                    'PoolingByMultiheadAttention': PoolingByMultiheadAttention},
    safe_mode=False, compile=False
)

# %%


# %%
for batch, _ in batched_dataset.take(1):
    coil_data = batch["coil_data"]
    coil_mask = batch["coil_mask"]
    surface_data = batch["surface_data"]  # assumes surface is already included in parse_tfrecord_fn
    surface_mask = batch['surface_mask']

    coil_latents = saved_encoder(
            {"coil_data": coil_data, "coil_mask": coil_mask},
            training=False
        )

# %%
coil_latents

# %%


# %%
import numpy as np
import tensorflow as tf

def _float_feature(value):
    """Returns a float_list from a tensor, numpy array, or list."""
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1).astype(np.float32)))

def _int64_feature(value):
    """Returns an int64_list from a tensor, numpy array, or list."""
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1).astype(np.int64)))

def serialize_example(coil_data, coil_mask, surface_data, surface_mask, coil_latent):
    feature = {
        "coil_data": _float_feature(coil_data),         # (TOTAL_ROWS, FEATURES_PER_COIL)
        "coil_mask": _int64_feature(coil_mask),         # (MAX_COILS,)
        "surface_data": _float_feature(surface_data),   # (TOTAL_SETS, FEATURES_PER_SET)
        "surface_mask": _int64_feature(surface_mask),   # (TOTAL_SETS,)
        "coil_latent": _float_feature(coil_latent),     # (latent_dim,)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# %%
def write_latent_augmented_tfrecord(dataset, encoder, output_dir, max_records_per_file=10000):
    os.makedirs(output_dir, exist_ok=True)
    file_count = 0
    record_count = 0
    writer = None

    for batch, _ in tqdm(dataset, desc="Encoding & Writing"):
        coil_data = batch["coil_data"]
        coil_mask = batch["coil_mask"]
        surface_data = batch["surface_data"]  # assumes surface is already included in parse_tfrecord_fn
        surface_mask = batch['surface_mask']

        # Predict latents
        coil_latents = encoder(
            {"coil_data": coil_data, "coil_mask": coil_mask},
            training=False
        )

        for i in range(coil_data.shape[0]):
            if writer is None or record_count >= max_records_per_file:
                if writer:
                    writer.close()
                tfrecord_path = os.path.join(output_dir, f"augmented_{file_count:03d}.tfrecord")
                writer = tf.io.TFRecordWriter(tfrecord_path)
                file_count += 1
                record_count = 0

            serialized = serialize_example(coil_data[i], coil_mask[i], surface_data[i], surface_mask[i], coil_latents[i])
            writer.write(serialized)
            record_count += 1

    if writer:
        writer.close()


# %%
latent_dir = Path('latents_tfrecords')
write_latent_augmented_tfrecord(batched_dataset, saved_encoder, latent_dir)


