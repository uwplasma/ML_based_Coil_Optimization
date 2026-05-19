
# SurfaceEncoder_overhauled_perceiver.py
# VAE-style surface encoder/decoder with:
# - Perceiver encoder (O(T*L) cross-attn)
# - Diagonal Gaussian latent head + KL warmup + capacity penalty (mirrors CoilAutoencoderModel)
# - Optional alignment loss to coil latents (cosine or MSE)
#
# Assumed surface tokens: (B, TOTAL_SETS=442, FEATURES_PER_SET=4)
# Assumed surface_mask:  (B, MAX_SETS=441) (mask for first 441 tokens; last token always valid metadata)

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
import keras

from transformers import DiagonalGaussian, PoolingByMultiheadAttention


def _masked_mean(e2: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Compute mean of e2 over masked entries."""
    mask = tf.cast(mask, e2.dtype)
    while mask.shape.rank < e2.shape.rank:
        mask = mask[..., tf.newaxis]
    num = tf.reduce_sum(e2 * mask)
    den = tf.reduce_sum(mask) + tf.cast(1e-8, e2.dtype)
    return num / den


def kl_diag_gaussian(mu: tf.Tensor, logvar: tf.Tensor, delta: float = 0.2) -> tf.Tensor:
    """
    KL( N(mu, diag(exp(logvar))) || N(0, I) ), averaged over batch.
    Mirrors your CoilAutoencoderModel clamp trick.
    """
    mu = tf.convert_to_tensor(mu)
    logvar = tf.convert_to_tensor(logvar)
    kl_j = 0.5 * (mu**2 + tf.exp(logvar) - logvar - 1.0)
    kl_j = tf.nn.relu(kl_j - delta) + delta
    kl_per = tf.reduce_sum(kl_j, axis=-1)
    return tf.reduce_mean(kl_per)


def cosine_align_loss(zs: tf.Tensor, zc: tf.Tensor) -> tf.Tensor:
    zs = tf.nn.l2_normalize(zs, axis=-1)
    zc = tf.stop_gradient(tf.nn.l2_normalize(zc, axis=-1))
    return 1.0 - tf.reduce_mean(tf.reduce_sum(zs * zc, axis=-1))


@keras.utils.register_keras_serializable(package="my_layers")
class LatentTile(layers.Layer):
    """Trainable latents (L, D) tiled to (B, L, D)."""
    def __init__(self, num_latents: int, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)

    def build(self, input_shape):
        self.latents = self.add_weight(
            shape=(self.num_latents, self.latent_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="learnable_latents",
        )
        super().build(input_shape)

    def call(self, x, training=False):
        b = tf.shape(x)[0]
        return tf.tile(self.latents[tf.newaxis, :, :], [b, 1, 1])

    def get_config(self):
        return {**super().get_config(), "num_latents": self.num_latents, "latent_dim": self.latent_dim}


@keras.utils.register_keras_serializable(package="my_layers")
class PerceiverBlock(layers.Layer):
    """
    Perceiver block: cross-attn (latents query inputs) -> FFN -> latent self-attn -> FFN.
    training has a default for functional use.
    """
    def __init__(self, latent_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)

        key_dim = max(1, self.latent_dim // max(1, self.num_heads))

        self.cross_attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim, dropout=self.dropout)
        self.cross_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_ffn = keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.latent_dim),
            layers.Dropout(self.dropout),
        ])
        self.cross_norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.self_attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim, dropout=self.dropout)
        self.self_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.self_ffn = keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.latent_dim),
            layers.Dropout(self.dropout),
        ])
        self.self_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, latents, inputs, attn_mask=None, training=False):
        cross_out = self.cross_attn(query=latents, value=inputs, key=inputs,
                                    attention_mask=attn_mask, training=training)
        lat = self.cross_norm1(latents + cross_out)
        lat = self.cross_norm2(lat + self.cross_ffn(lat, training=training))

        self_out = self.self_attn(query=lat, value=lat, key=lat, training=training)
        lat = self.self_norm1(lat + self_out)
        lat = self.self_norm2(lat + self.self_ffn(lat, training=training))
        return lat

    def get_config(self):
        return {
            **super().get_config(),
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        }


def build_surface_perceiver_encoder(
    *,
    max_sets: int,
    features_per_set: int,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    num_latents: int,
    num_blocks: int,
    dropout: float,
    name: str = "surface",
) -> tf.keras.Model:
    """
    Inputs:
      - surface_data: (B, max_sets, features_per_set)  where max_sets=442
      - surface_mask: (B, max_sets-1)                  where max_sets-1=441
    Output:
      - pooled: (B, 1, embed_dim)
    """
    data_key = f"{name}_data"
    mask_key = f"{name}_mask"
    model_name = f"{name}_perceiver_encoder"

    x_in = tf.keras.Input(shape=(max_sets, features_per_set), name=data_key)
    m_in = tf.keras.Input(shape=(max_sets - 1,), dtype=tf.float32, name=mask_key)

    x = layers.Dense(embed_dim, name=f"{name}_input_proj")(x_in)

    ones = layers.Lambda(lambda m: tf.ones_like(m[:, :1]), name=f"{name}_scalar_mask",
                         output_shape=(1,))(m_in)
    full_mask = layers.Concatenate(axis=1, name=f"{name}_full_mask")([m_in, ones])

    attn_mask = layers.Lambda(lambda m: tf.expand_dims(tf.cast(m, tf.bool), axis=1),
                              name=f"{name}_attn_mask", output_shape=(1,max_sets))(full_mask)

    lat = LatentTile(num_latents=num_latents, latent_dim=embed_dim, name=f"{name}_latents")(x)

    for i in range(num_blocks):
        lat = PerceiverBlock(
            latent_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"{name}_pblock_{i}",
        )(lat, x, attn_mask=attn_mask)

    lat = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_norm")(lat)

    pooled = PoolingByMultiheadAttention(embed_dim, num_heads, num_seeds=1, dropout=dropout, name=f"{name}_pma")(lat)
    return tf.keras.Model(inputs={data_key: x_in, mask_key: m_in}, outputs=pooled, name=model_name)


def build_surface_decoder(
    *,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    num_layers: int,
    max_sets: int,
    features_per_set: int,
    dropout: float,
    name: str = "surface",
) -> tf.keras.Model:
    """Uses your existing LearnedQueryDecoder-based TransformerDecoder."""
    from transformers import TransformerDecoder
    return TransformerDecoder(
        name=name,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_sets=max_sets,
        features_per_set=features_per_set,
        dropout=dropout,
    )


@keras.utils.register_keras_serializable(package="my_layers")
class SurfaceVAEPerceiverModel(tf.keras.Model):
    """VAE-style surface autoencoder + optional alignment to coil latents."""

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        latent_dim: int,
        *,
        coil_latent_dim: int,
        value_col: int = 3,
        meta_recon_weight: float = 0.0,
        align_weight: float = 0.0,
        align_type: str = "cosine",
        kl_target: float = 0.02,
        kl_warmup_steps: int = 10_000,
        kl_cap: float = 3.0,
        kl_cap_warmup: int = 15_000,
        kl_gamma: float = 1e-3,
        sample_latent_train: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.latent_head = DiagonalGaussian(latent_dim, name="latent_head")
        self.align_proj = layers.Dense(coil_latent_dim, name="align_proj")

        self.value_col = int(value_col)
        self.meta_recon_weight = float(meta_recon_weight)
        self.align_weight = float(align_weight)
        self.align_type = str(align_type)

        self.kl_target = float(kl_target)
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.sample_latent_train = bool(sample_latent_train)
        self._global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")

        self.kl_cap_final = tf.constant(kl_cap, tf.float32)
        self.kl_cap_warmup = tf.constant(kl_cap_warmup, tf.int64)
        self.kl_gamma = tf.constant(kl_gamma, tf.float32)

    def _kl_weight(self):
        t = tf.cast(tf.minimum(self._global_step, self.kl_warmup_steps), tf.float32)
        return (t / float(self.kl_warmup_steps)) * self.kl_target

    def _kl_capacity(self):
        t = tf.cast(tf.minimum(self._global_step, self.kl_cap_warmup), tf.float32)
        return (t / tf.cast(self.kl_cap_warmup, tf.float32)) * self.kl_cap_final

    def _align_loss(self, zs, zc):
        if self.align_weight <= 0.0:
            return tf.constant(0.0, tf.float32)
        if self.align_type.lower() == "mse":
            return tf.reduce_mean(tf.square(zs - tf.stop_gradient(zc)))
        return cosine_align_loss(zs, zc)

    def train_step(self, data):
        inputs, targets = data
        surface = targets["surface"]
        coil_latent = targets["coil_latent"]
        mask = tf.cast(inputs["surface_mask"], tf.float32)  # (B,441)

        with tf.GradientTape() as tape:
            h = self.encoder(
                {"surface_data": inputs["surface_data"], "surface_mask": inputs["surface_mask"]},
                training=True,
            )  # (B,1,D)
            z, mu, logvar = self.latent_head(h, training=True, sample=self.sample_latent_train)
            pred = self.decoder(z, training=True)  # (B,442,4)

            # Value recon: first 441 tokens, value_col only
            pred_val = pred[:, :-1, self.value_col]
            true_val = surface[:, :-1, self.value_col]
            val_loss = _masked_mean(tf.square(pred_val - true_val), mask)

            scaler_loss = tf.reduce_mean(tf.square(pred[:, -1, self.value_col] - surface[:, -1, self.value_col]))
            recon_loss = val_loss + scaler_loss

            # Optional tiny recon on the other 3 channels (keeps decoder stable)
            meta_loss = tf.constant(0.0, tf.float32)
            if self.meta_recon_weight > 0.0:
                meta_cols = [i for i in range(int(surface.shape[-1])) if i != self.value_col]
                if meta_cols:
                    pred_meta = tf.gather(pred[:, :-1, :], meta_cols, axis=-1)
                    true_meta = tf.gather(surface[:, :-1, :], meta_cols, axis=-1)
                    meta_loss = _masked_mean(tf.square(pred_meta - true_meta), mask)
                    recon_loss = recon_loss + self.meta_recon_weight * meta_loss

            kl = kl_diag_gaussian(mu, logvar)
            kl_w = self._kl_weight()
            C_t = self._kl_capacity()
            capacity_penalty = self.kl_gamma * tf.square(kl - C_t)

            mu2 = tf.squeeze(mu, axis=1) if mu.shape.rank == 3 else mu
            zs = self.align_proj(mu2)
            align = self._align_loss(zs, coil_latent)

            total_loss = recon_loss + kl_w * kl + capacity_penalty + self.align_weight * align

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._global_step.assign_add(1)

        unmasked_val_mse = tf.reduce_mean(tf.square(pred_val - true_val))
        return {
            "loss": total_loss,
            "masked_recon": val_loss,
            "scaler_loss": scaler_loss,
            "meta_loss": meta_loss,
            "KL": kl,
            "KL_weight": kl_w,
            "capacity_penalty": capacity_penalty,
            "capacity_weight": C_t,
            "align_loss": align,
            "unmasked_val_mse": unmasked_val_mse,
        }

    def test_step(self, data):
        inputs, targets = data
        surface = targets["surface"]
        coil_latent = targets["coil_latent"]
        mask = tf.cast(inputs["surface_mask"], tf.float32)

        h = self.encoder(
            {"surface_data": inputs["surface_data"], "surface_mask": inputs["surface_mask"]},
            training=False,
        )
        z, mu, logvar = self.latent_head(h, training=False, sample=False)
        pred = self.decoder(z, training=False)

        pred_val = pred[:, :-1, self.value_col]
        true_val = surface[:, :-1, self.value_col]
        val_loss = _masked_mean(tf.square(pred_val - true_val), mask)

        scaler_loss = tf.reduce_mean(tf.square(pred[:, -1, self.value_col] - surface[:, -1, self.value_col]))
        recon_loss = val_loss + scaler_loss

        meta_loss = tf.constant(0.0, tf.float32)
        if self.meta_recon_weight > 0.0:
            meta_cols = [i for i in range(int(surface.shape[-1])) if i != self.value_col]
            if meta_cols:
                pred_meta = tf.gather(pred[:, :-1, :], meta_cols, axis=-1)
                true_meta = tf.gather(surface[:, :-1, :], meta_cols, axis=-1)
                meta_loss = _masked_mean(tf.square(pred_meta - true_meta), mask)
                recon_loss = recon_loss + self.meta_recon_weight * meta_loss

        kl = kl_diag_gaussian(mu, logvar)
        kl_w = self._kl_weight()
        C_t = self._kl_capacity()
        capacity_penalty = self.kl_gamma * tf.square(kl - C_t)

        mu2 = tf.squeeze(mu, axis=1) if mu.shape.rank == 3 else mu
        zs = self.align_proj(mu2)
        align = self._align_loss(zs, coil_latent)

        total_loss = recon_loss + kl_w * kl + capacity_penalty + self.align_weight * align
        unmasked_val_mse = tf.reduce_mean(tf.square(pred_val - true_val))

        return {
            "loss": total_loss,
            "masked_recon": val_loss,
            "scaler_loss": scaler_loss,
            "meta_loss": meta_loss,
            "KL": kl,
            "KL_weight": kl_w,
            "capacity_penalty": capacity_penalty,
            "capacity_weight": C_t,
            "align_loss": align,
            "unmasked_val_mse": unmasked_val_mse,
        }
