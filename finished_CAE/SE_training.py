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
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
%load_ext autoreload
%autoreload 2
from latent_loader import load_full_dataset, load_split_datasets
from transformers import TransformerEncoder, TransformerDecoder
from SurfaceEncoder import SurfaceEncoderModel

# %%
tfrecord_dir = Path("latents_tfrecords")


# %%
def build_encoder(max_sets, features_per_set, embed_dim, num_heads, ff_dim, num_sab_blocks, dropout):
    input_surface = tf.keras.Input(shape=(max_sets+1, features_per_set), name='surface_data')
    mask = tf.keras.Input(shape=(max_sets,), dtype=tf.float32, name='surface_mask')
    
    pooled, _ = TransformerEncoder(input_surface, mask, 
                                embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, 
                                dropout=dropout, num_sab_blocks=num_sab_blocks)
    
    return tf.keras.Model(inputs={"surface_data": input_surface, "surface_mask": mask}, outputs=pooled, name="surface_encoder")

# %%
def build_decoder(embed_dim, num_heads, ff_dim, num_layers, max_sets, features_per_set, dropout):
    encoded_input = tf.keras.Input(shape=(1, embed_dim), name='encoded_latent')
    
    decoded, _ = TransformerDecoder(encoded_input, embed_dim=embed_dim, num_heads=num_heads, 
                                    ff_dim=ff_dim, num_layers=num_layers, 
                                    max_sets=max_sets, features_per_set=features_per_set, 
                                    dropout=dropout)
    
    return tf.keras.Model(inputs=encoded_input, outputs=decoded, name="surface_decoder")

# %%
def build_surface_encoder(hp):
    encoder = build_encoder(
        max_sets=441, features_per_set=4, embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"], ff_dim=hp["ff_dim"], 
        num_sab_blocks=hp["sab_blocks"], dropout=hp["enc_dropout"]
    )
    
    decoder = build_decoder(
        embed_dim=hp["embed_dim"], num_heads=hp["num_heads"], ff_dim=hp["ff_dim"],
        num_layers=hp["decoder_blocks"], max_sets=441, features_per_set=4, dropout=hp["dec_dropout"]
    )
    
    model = SurfaceEncoderModel(encoder, decoder)
    return model, encoder, decoder

# %%
example_hp = {'batch_size': 64,
 'embed_dim': 64,
 'num_heads': 4,
 'ff_dim': 128,
 'enc_dropout': 0.03687998002549748,
 'dec_dropout': 0.14346229179997497,
 'learning_rate': 1.0024438630534578e-05,
 'weight_decay': 0.0060082332950099505,
 'sab_blocks': 1,
 'decoder_blocks': 1}

# %%
def build_model(hp):
    model, encoder, decoder = build_surface_encoder(hp)    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-7, decay_steps=423, 
                                                            warmup_target=hp['learning_rate'], warmup_steps=47)
    optimizer = tf.keras.optimizers.Lion(learning_rate=lr_schedule, weight_decay=hp['weight_decay'])
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
def train_surface_encoder(hp, train_ds, val_ds, epochs, steps_per_epoch, log_root="logs", use_wandb=False):
    model, encoder, decoder = build_model(hp)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, f"{run_id}_trial")
    os.makedirs(log_dir, exist_ok=True)

    # total_steps = epochs * steps_per_epoch

    callbacks = prepare_callbacks(log_dir)

    if use_wandb:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project="surface_encoder", config=hp)
        callbacks.append(WandbCallback())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    encoder.save(os.path.join(log_dir, "encoder.keras"))
    decoder.save(os.path.join(log_dir, "decoder.keras"))
    return history, encoder, decoder

# %%
train_ds, val_ds, test_ds = load_split_datasets(tfrecord_dir, batch_size=example_hp["batch_size"])

history, encoder, decoder = train_surface_encoder(
    hp=example_hp,
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=10,
    steps_per_epoch=47,
    log_root="surface_logs",
    use_wandb=False
)

# %%
pd.DataFrame(history.history).plot()

# %%
import keras

# %%
from transformers import SelfAttentionBlock, PoolingByMultiheadAttention
# encoder = tf.keras.models.load_model('coil_logs/20250812-152016_trial/encoder.keras', safe_mode=False)
encoder = tf.keras.models.load_model(
    "coil_logs/20250812-152016_trial/encoder.keras",
    custom_objects={"SelfAttentionBlock": SelfAttentionBlock, 
                    'PoolingByMultiheadAttention': PoolingByMultiheadAttention},
    safe_mode=False,
)


