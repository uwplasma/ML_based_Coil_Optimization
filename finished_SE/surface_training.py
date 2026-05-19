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
tf.config.optimizer.set_jit(True) 

# %%
%load_ext autoreload
%autoreload 2
from latent_loader import load_full_dataset, load_split_datasets
from transformers import TransformerEncoder, TransformerDecoder
from SurfaceEncoder_overhauled_perceiver import (
    build_surface_perceiver_encoder,
    build_surface_decoder,
    SurfaceVAEPerceiverModel,
)

# %%
tfrecord_dir = Path("latents_tfrecords")

# %%
def build_surface_autoencoder(hp, *, max_sets, features_per_set, coil_latent_dim, value_col):
    T = max_sets + 1
    encoder = build_surface_perceiver_encoder(
        max_sets=T,
        features_per_set=features_per_set,
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        ff_dim=hp["ff_dim"],
        num_latents=hp["num_latents"],
        num_blocks=hp["enc_blocks"],
        dropout=hp["enc_dropout"],
        name="surface",
    )
    decoder = build_surface_decoder(
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        ff_dim=hp["ff_dim"],
        num_layers=hp["dec_blocks"],
        max_sets=T,
        features_per_set=features_per_set,
        dropout=hp["dec_dropout"],
        name="surface",
    )

    model = SurfaceVAEPerceiverModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=hp["embed_dim"],
        coil_latent_dim=coil_latent_dim,
        value_col=value_col,
        meta_recon_weight=hp["meta_recon_weight"],
        align_weight=hp["align_weight"],
        align_type=hp["align_type"],
        kl_target=hp["kl_target"],
        kl_warmup_steps=hp["kl_warmup_steps"],
        kl_cap=hp["kl_cap"],
        kl_cap_warmup=hp["kl_cap_warmup"],
        kl_gamma=hp["kl_gamma"],
        sample_latent_train=True,
    )
    return model, encoder, decoder

# %%
def build_model(hp, *, epochs, num_examples_for_lr, max_sets, features_per_set, coil_latent_dim, value_col):
    model, encoder, decoder = build_surface_autoencoder(
        hp,
        max_sets=max_sets,
        features_per_set=features_per_set,
        coil_latent_dim=coil_latent_dim,
        value_col=value_col,
    )
    total_steps = int((num_examples_for_lr * epochs) // hp["batch_size"])
    total_steps = max(total_steps, 1)
    warmup_steps = int(total_steps * hp["warmup_frac"])

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=hp["learning_rate"] / 25.0, 
                                                            decay_steps=total_steps-warmup_steps, 
                                                            warmup_target=hp['learning_rate'], 
                                                            warmup_steps=warmup_steps)
    
    # WarmupCosine(
    #     max_lr=hp["learning_rate"],
    #     total_steps=total_steps,
    #     warmup_steps=warmup_steps,
    #     min_lr=hp["learning_rate"] / 25.0,
    # )

    try:
        optimizer = tf.keras.optimizers.Lion(
            learning_rate=lr_schedule, weight_decay=hp["weight_decay"], clipnorm=1.0
        )
    except Exception:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=hp["weight_decay"], clipnorm=1.0
        )

    model.compile(optimizer=optimizer)
    return model, encoder, decoder

# %%
hp = {
  "embed_dim": 64,
  "num_heads": 8,
  "ff_dim": 768,
  "num_latents": 96,
  "enc_blocks": 4,
  "dec_blocks": 2,
  "enc_dropout": 0.08739726153053011,
  "dec_dropout": 0.19333025998730616,
  "batch_size": 32,
  "learning_rate": 0.0001016162398740397,
  "weight_decay": 0.0013128150300032921,
  "warmup_frac": 0.060697802867789974,
  "kl_target": 0.011987564716964808,
  "kl_warmup_steps": 20000,
  "kl_cap_per_dim": 0.25240651983262236,
  "kl_cap_warmup": 24000,
  "kl_gamma": 0.0070295562199317735,
  "kl_cap": 16.15401726928783,
  "align_weight": 0.9890176129613788,
  "align_type": "cosine",
  "meta_recon_weight": 0.010773882055979482
}

# %%
def prepare_callbacks(log_dir, model_name="surface_autoencoder"):
    return [
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint(
            filepath=os.path.join(log_dir, f"{model_name}_best.keras"),
            save_best_only=True,
            monitor='val_loss'
        )
    ]

# %%
class SaveBestSubmodels(tf.keras.callbacks.Callback):
    """
    Saves encoder/decoder when `monitor` improves AND also saves q_encoder (encoder + DiagonalGaussian head).
    """
    def __init__(self, autoencoder, encoder, decoder, log_dir, monitor="val_loss", mode="min"):
        super().__init__()
        self.autoencoder = autoencoder      
        self.encoder = encoder
        self.decoder = decoder
        self.monitor = monitor
        self.log_dir = log_dir
        self.best = np.inf if mode == "min" else -np.inf
        self.cmp = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        self.save_dir = os.path.join(log_dir, "best")
        os.makedirs(self.save_dir, exist_ok=True)

        self.q_encoder = None

    def _build_q_encoder(self):
        if self.q_encoder is not None:
            return

        pooled = self.encoder.output  # typically (B, 1, D)

        z, mu, logvar = self.autoencoder.latent_head(pooled, training=False, sample=False)

        self.q_encoder = tf.keras.Model(
            inputs=self.encoder.inputs,
            outputs={"z": z, "mu": mu, "logvar": logvar},
            name="surface_q_encoder",
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.cmp(current, self.best):
            self.best = current

            enc_path = os.path.join(self.save_dir, "encoder.best.keras")
            dec_path = os.path.join(self.save_dir, "decoder.best.keras")
            q_path   = os.path.join(self.save_dir, "q_encoder.best.keras")

            self.encoder.save(enc_path)
            self.decoder.save(dec_path)

            self._build_q_encoder()
            self.q_encoder.save(q_path)

            with open(os.path.join(self.save_dir, "best_metric.json"), "w") as f:
                json.dump({self.monitor: float(self.best), "epoch": int(epoch)}, f)

# %%
def train_surface_autoencoder(
    hp,
    train_ds,
    val_ds,
    *,
    epochs,
    num_examples_for_lr,
    max_sets,
    features_per_set,
    coil_latent_dim=64,
    value_col=3,
    early_stopping=True,
    es_monitor="val_masked_recon",
    es_min_delta=0.0,
    es_patience=30,
    es_mode="min",
    log_root='logs'
):
    model, encoder, decoder = build_model(
        hp,
        epochs=epochs,
        num_examples_for_lr=num_examples_for_lr,
        max_sets=max_sets,
        features_per_set=features_per_set,
        coil_latent_dim=coil_latent_dim,
        value_col=value_col,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, f"{run_id}_trial")
    os.makedirs(log_dir, exist_ok=True)

    callbacks = prepare_callbacks(log_dir)

    callbacks = []
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_monitor,
                min_delta=es_min_delta,
                patience=es_patience,
                mode=es_mode,
                restore_best_weights=True,
            )
        )

        callbacks.append(SaveBestSubmodels(
            autoencoder=model,
            encoder=encoder,
            decoder=decoder,
            log_dir=log_dir,
            monitor=es_monitor,
            mode=es_mode
        ))

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks) #verbose=0 means no metric prints
    return history

# %%
train_ds, val_ds, test_ds = load_split_datasets(tfrecord_dir, batch_size=hp["batch_size"])

history, encoder, decoder = train_surface_autoencoder(
    hp=hp,
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=200,
    num_examples_for_lr=251_200,
    max_sets=441,
    features_per_set=4,
    log_root="surface_logs",
)

# %%
pd.DataFrame(history.history).plot()

# %%


# %%
model, encoder, decoder = build_model(
        hp,
        epochs=200,
        num_examples_for_lr=251_200,
        max_sets=441,
        features_per_set=4,
        coil_latent_dim=64,
        value_col=3,
    )

# %%
encoder.load_weights("surface_logs/20260308-181957_trial/best/q_encoder.best.keras")

# %%
batched_dataset = load_full_dataset(tfrecord_dir, batch_size = hp['batch_size'])

# %%
for batch, _ in batched_dataset.take(1):
    coil_data = batch["coil_data"]
    coil_mask = batch["coil_mask"]
    surface_data = batch["surface_data"]  # assumes surface is already included in parse_tfrecord_fn
    surface_mask = batch['surface_mask']

    surface_latents = encoder(
            {"surface_data": surface_data, "surface_mask": surface_mask},
            training=False
        )

# %%
surface_latents

# %%
encoder.save("surface_logs/20260308-181957_trial/best/rebuilt_q_encoder.best.keras")

# %%


# %%


# %%
from transformers import SelfAttentionBlock, PoolingByMultiheadAttention, DiagonalGaussian
from SurfaceEncoder_overhauled_perceiver import LatentTile, PerceiverBlock
# encoder = tf.keras.models.load_model('coil_logs/20250812-152016_trial/encoder.keras', safe_mode=False)
s_encoder = tf.keras.models.load_model(
    "surface_logs/20260308-181957_trial/best/rebuilt_q_encoder.best.keras",
    custom_objects={"SelfAttentionBlock": SelfAttentionBlock, 
                    'PoolingByMultiheadAttention': PoolingByMultiheadAttention,
                    'DiagonalGaussian': DiagonalGaussian,
                    'LatentTile': LatentTile,
                    'PerceiverBlock': PerceiverBlock},
    safe_mode=False, compile=False
)


