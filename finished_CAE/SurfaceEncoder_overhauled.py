import tensorflow as tf
import keras
from keras.saving import serialize_keras_object, deserialize_keras_object

from transformers import DiagonalGaussian


def masked_mse(pred, target, mask):
    """Masked MSE matching CoilAutoEncoder.py semantics.

    Notes:
      - `mask` is assumed to weight *tokens* (not individual features).
      - If pred/target are (B,T,F) and mask is (B,T), we expand mask to (B,T,1).
    """
    e2 = tf.square(pred - target)
    if mask.shape.rank == 2 and e2.shape.rank == 3:
        mask = mask[..., tf.newaxis]
    mask = tf.cast(mask, tf.float32)
    num = tf.reduce_sum(mask * e2)
    den = tf.reduce_sum(mask) + 1e-8
    return num / den


def kl_diag_gaussian(mu, logvar, delta=0.2):
    """KL( N(mu, diag(exp(logvar))) || N(0, I) ) with a small free-bits clamp.

    This matches the CoilAutoEncoder.py structure:
      kl_j = 0.5*(mu^2 + exp(logvar) - logvar - 1)
      kl_j = relu(kl_j - delta) + delta
      return mean(sum(kl_j, -1))

    Works for mu/logvar ranks 2 or 3 (e.g. (B,D) or (B,1,D)).
    """
    mu = tf.convert_to_tensor(mu)
    logvar = tf.convert_to_tensor(logvar)
    kl_j = 0.5 * (mu ** 2 + tf.exp(logvar) - logvar - 1.0)
    kl_j = tf.nn.relu(kl_j - float(delta)) + float(delta)
    # sum over latent dim, then average over remaining batch/(token) dims
    return tf.reduce_mean(tf.reduce_sum(kl_j, axis=-1))


def cosine_align_loss(zs, zc, stopgrad_target=True):
    """1 - cosine similarity, averaged over batch."""
    zs = tf.nn.l2_normalize(zs, axis=-1)
    zc = tf.nn.l2_normalize(zc, axis=-1)
    if stopgrad_target:
        zc = tf.stop_gradient(zc)
    return 1.0 - tf.reduce_mean(tf.reduce_sum(zs * zc, axis=-1))


def _ensure_surface_3d(surface):
    """Coerce surface to shape (B,T,1) if given as (B,T)."""
    surface = tf.convert_to_tensor(surface)
    if surface.shape.rank == 2:
        surface = surface[..., tf.newaxis]
    return surface


def _ensure_latent_2d(z):
    """Coerce latent to (B,D) if given as (B,1,D)."""
    z = tf.convert_to_tensor(z)
    if z.shape.rank == 3 and z.shape[1] == 1:
        z = tf.squeeze(z, axis=1)
    return z


def _coerce_surface_mask(surface_mask, T, append_scalar_one=True):
    """Ensure a (B,T) float32 mask.

    If surface_mask comes in as (B,T-1) (common when the last token is a scalar/meta token),
    append a trailing 1 so the last token is always valid.
    """
    surface_mask = tf.cast(surface_mask, tf.float32)
    if not append_scalar_one:
        return surface_mask

    mask_T = tf.shape(surface_mask)[1]

    def _append_one():
        ones = tf.ones((tf.shape(surface_mask)[0], 1), tf.float32)
        return tf.concat([surface_mask, ones], axis=1)

    # If mask already matches T, keep it; if it matches T-1, append one; otherwise pass through.
    surface_mask = tf.cond(
        tf.equal(mask_T, T),
        lambda: surface_mask,
        lambda: tf.cond(tf.equal(mask_T, T - 1), _append_one, lambda: surface_mask),
    )
    return surface_mask


def _add_position_feature(surface_3d):
    """Augment surface tokens with a normalized position feature.

    If your surface data is just a 1-D coefficient vector, the vanilla SetTransformer-style
    encoder (no positional embeddings) cannot know which coefficient index it is seeing.

    We fix that by concatenating a normalized index i/(T-1) as a second feature:
      input token = [coeff_value, pos_norm]

    Returns shape (B,T,2).
    """
    B = tf.shape(surface_3d)[0]
    T = tf.shape(surface_3d)[1]

    # positions in [0,1]
    pos = tf.cast(tf.range(T), tf.float32) / tf.cast(tf.maximum(T - 1, 1), tf.float32)
    pos = tf.reshape(pos, (1, T, 1))
    pos = tf.tile(pos, (B, 1, 1))

    return tf.concat([surface_3d, pos], axis=-1)


@keras.utils.register_keras_serializable(package="my_layers")
class SurfaceEncoderModel(tf.keras.Model):
    """Surface autoencoder/VAE-style wrapper, mirroring CoilAutoencoderModel.

    Expected data (default pipeline in latent_loader.py):
      inputs: {
        'surface_data': (B,T) or (B,T,1),
        'surface_mask': (B,T) or (B,T-1),
        'coil_latent': (B,D)  (optional; can also be in targets)
      }
      targets: either
        - surface tensor (B,T) / (B,T,1)
        - or dict with keys {'surface': ..., 'coil_latent': ...}

    Loss:
      L = recon + kl_w * KL + gamma*(KL - C(t))^2 + align_w * align(z_s, z_c)

    Notes:
      - We apply alignment on the *mean* (mu) by default to avoid noisy gradients.
      - Alignment is optional and defaults to 0.
    """

    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        kl_target,
        kl_warmup_steps=10_000,
        kl_cap=3.0,
        kl_cap_warmup=15_000,
        kl_gamma=1e-3,
        sample_latent_train=True,
        # --- optional alignment to coil latents ---
        align_weight=0.0,
        align_metric="cosine",  # 'cosine' or 'mse'
        align_on="mu",          # 'mu' or 'z'
        stopgrad_align_target=True,
        # --- surface-specific helpers ---
        append_scalar_one_to_mask=True,
        add_position_feature=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.latent_head = DiagonalGaussian(int(latent_dim), name="latent_head")

        self.kl_target = float(kl_target)
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.sample_latent_train = bool(sample_latent_train)
        self._global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")

        self.kl_cap_final = tf.constant(float(kl_cap), tf.float32)
        self.kl_cap_warmup = tf.constant(int(kl_cap_warmup), tf.int64)
        self.kl_gamma = tf.constant(float(kl_gamma), tf.float32)

        self.align_weight = float(align_weight)
        self.align_metric = str(align_metric)
        self.align_on = str(align_on)
        self.stopgrad_align_target = bool(stopgrad_align_target)

        self.append_scalar_one_to_mask = bool(append_scalar_one_to_mask)
        self.add_position_feature = bool(add_position_feature)

    def _kl_weight(self):
        t = tf.cast(tf.minimum(self._global_step, self.kl_warmup_steps), tf.float32)
        return (t / float(self.kl_warmup_steps)) * self.kl_target

    def _kl_capacity(self):
        t = tf.cast(tf.minimum(self._global_step, self.kl_cap_warmup), tf.float32)
        return (t / tf.cast(self.kl_cap_warmup, tf.float32)) * self.kl_cap_final

    def call(self, inputs, training=False):
        # Encode -> sample -> decode (mainly for inference/debug)
        surface = _ensure_surface_3d(inputs["surface_data"])
        if self.add_position_feature:
            surface_in = _add_position_feature(surface)
        else:
            surface_in = surface

        T = tf.shape(surface)[1]
        mask = _coerce_surface_mask(inputs["surface_mask"], T, append_scalar_one=self.append_scalar_one_to_mask)

        h = self.encoder({"surface_data": surface_in, "surface_mask": mask}, training=training)
        z, mu, logvar = self.latent_head(h, training=training, sample=(training and self.sample_latent_train))
        recon = self.decoder(z, training=training)
        return recon

    def _alignment_loss(self, zs, zc):
        if self.align_metric.lower() == "mse":
            if self.stopgrad_align_target:
                zc = tf.stop_gradient(zc)
            return tf.reduce_mean(tf.square(zs - zc))
        # default cosine
        return cosine_align_loss(zs, zc, stopgrad_target=self.stopgrad_align_target)

    def train_step(self, data):
        inputs, targets = data

        # targets can be a dict (latent_loader.py) or raw surface tensor
        if isinstance(targets, dict):
            surface_target = targets.get("surface", targets)
            coil_latent_target = targets.get("coil_latent", None)
        else:
            surface_target = targets
            coil_latent_target = None

        # also allow coil_latent to be provided in inputs
        if coil_latent_target is None and isinstance(inputs, dict) and "coil_latent" in inputs:
            coil_latent_target = inputs["coil_latent"]

        surface_target = _ensure_surface_3d(surface_target)

        # Build/repair mask
        T = tf.shape(surface_target)[1]
        mask = _coerce_surface_mask(inputs["surface_mask"], T, append_scalar_one=self.append_scalar_one_to_mask)

        # Prepare surface input tokens
        surface_in = _ensure_surface_3d(inputs["surface_data"])
        if self.add_position_feature:
            surface_in = _add_position_feature(surface_in)

        with tf.GradientTape() as tape:
            h = self.encoder({"surface_data": surface_in, "surface_mask": mask}, training=True)
            z, mu, logvar = self.latent_head(h, training=True, sample=self.sample_latent_train)
            surface_recon = self.decoder(z, training=True)

            recon_loss = masked_mse(surface_recon, surface_target, mask)

            kl = kl_diag_gaussian(mu, logvar)
            kl_w = self._kl_weight()

            C_t = self._kl_capacity()
            capacity_penalty = self.kl_gamma * tf.square(kl - C_t)

            total_loss = recon_loss + kl_w * kl + capacity_penalty

            # Optional: align to coil latents
            align_loss = tf.constant(0.0, tf.float32)
            if self.align_weight > 0.0 and coil_latent_target is not None:
                zc = _ensure_latent_2d(coil_latent_target)
                zs = _ensure_latent_2d(mu if self.align_on == "mu" else z)
                align_loss = self._alignment_loss(zs, zc)
                total_loss = total_loss + tf.constant(self.align_weight, tf.float32) * align_loss

            # Reference metrics
            unmasked_mse = tf.reduce_mean(tf.square(surface_recon - surface_target))
            mae = tf.reduce_mean(tf.abs(surface_recon - surface_target))

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._global_step.assign_add(1)

        out = {
            "loss": total_loss,
            "masked_recon": recon_loss,
            "KL": kl,
            "KL_weight": kl_w,
            "capacity_penalty": capacity_penalty,
            "capacity_weight": C_t,
            "align_loss": align_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae,
        }
        return out

    def test_step(self, data):
        inputs, targets = data

        if isinstance(targets, dict):
            surface_target = targets.get("surface", targets)
            coil_latent_target = targets.get("coil_latent", None)
        else:
            surface_target = targets
            coil_latent_target = None

        if coil_latent_target is None and isinstance(inputs, dict) and "coil_latent" in inputs:
            coil_latent_target = inputs["coil_latent"]

        surface_target = _ensure_surface_3d(surface_target)
        T = tf.shape(surface_target)[1]
        mask = _coerce_surface_mask(inputs["surface_mask"], T, append_scalar_one=self.append_scalar_one_to_mask)

        surface_in = _ensure_surface_3d(inputs["surface_data"])
        if self.add_position_feature:
            surface_in = _add_position_feature(surface_in)

        h = self.encoder({"surface_data": surface_in, "surface_mask": mask}, training=False)
        z, mu, logvar = self.latent_head(h, training=False, sample=False)
        surface_recon = self.decoder(z, training=False)

        recon_loss = masked_mse(surface_recon, surface_target, mask)
        kl = kl_diag_gaussian(mu, logvar)
        kl_w = self._kl_weight()

        C_t = self._kl_capacity()
        capacity_penalty = self.kl_gamma * tf.square(kl - C_t)
        total_loss = recon_loss + kl_w * kl + capacity_penalty

        align_loss = tf.constant(0.0, tf.float32)
        if self.align_weight > 0.0 and coil_latent_target is not None:
            zc = _ensure_latent_2d(coil_latent_target)
            zs = _ensure_latent_2d(mu if self.align_on == "mu" else z)
            align_loss = self._alignment_loss(zs, zc)
            total_loss = total_loss + tf.constant(self.align_weight, tf.float32) * align_loss

        unmasked_mse = tf.reduce_mean(tf.square(surface_recon - surface_target))
        mae = tf.reduce_mean(tf.abs(surface_recon - surface_target))

        return {
            "loss": total_loss,
            "masked_recon": recon_loss,
            "KL": kl,
            "KL_weight": kl_w,
            "capacity_penalty": capacity_penalty,
            "capacity_weight": C_t,
            "align_loss": align_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae,
        }

    def get_config(self):
        return {
            "name": self.name,
            "encoder": serialize_keras_object(self.encoder),
            "decoder": serialize_keras_object(self.decoder),
            "latent_dim": int(self.latent_head.latent_dim),
            "kl_target": float(self.kl_target),
            "kl_warmup_steps": int(self.kl_warmup_steps),
            "kl_cap": float(self.kl_cap_final.numpy()),
            "kl_cap_warmup": int(self.kl_cap_warmup.numpy()),
            "kl_gamma": float(self.kl_gamma.numpy()),
            "sample_latent_train": bool(self.sample_latent_train),
            "align_weight": float(self.align_weight),
            "align_metric": self.align_metric,
            "align_on": self.align_on,
            "stopgrad_align_target": bool(self.stopgrad_align_target),
            "append_scalar_one_to_mask": bool(self.append_scalar_one_to_mask),
            "add_position_feature": bool(self.add_position_feature),
        }

    @classmethod
    def from_config(cls, config):
        enc = deserialize_keras_object(config.pop("encoder"))
        dec = deserialize_keras_object(config.pop("decoder"))
        name = config.pop("name", None)
        return cls(encoder=enc, decoder=dec, name=name, **config)
