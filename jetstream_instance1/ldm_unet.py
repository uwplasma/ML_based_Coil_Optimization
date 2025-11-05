# ldm_unet_training_scaffold.py
import tensorflow as tf
import math
import numpy as np
import os
from typing import Optional

# -------------------------
# Basic config / hyperparams
# -------------------------
BATCH_SIZE = 8
SEQ_LEN = 128         # L: sequence length for coil latents (set to your latent length)
LATENT_DIM = 4        # C: channels in coil latent
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4)  # shallow for PoC; scale up later
NUM_RES_BLOCKS = 2
TIME_EMB_DIM = 256
USE_CROSS_ATTENTION = True

EPOCHS = 10
STEPS_PER_EPOCH = 200
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
UNCOND_PROB = 0.12    # classifier-free guidance train-time masking prob
EMA_DECAY = 0.9999

# Mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Policy:", tf.keras.mixed_precision.global_policy())

# -------------------------
# Noise schedule (DDPM-style linear betas)
# -------------------------
def make_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

NUM_TIMESTEPS = 1000
betas = make_beta_schedule(NUM_TIMESTEPS)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

# convert to tensors
betas_t = tf.constant(betas, dtype=tf.float32)
alphas_cumprod_t = tf.constant(alphas_cumprod, dtype=tf.float32)
sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)
sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1 - alphas_cumprod), dtype=tf.float32)

# -------------------------
# Utilities
# -------------------------
def get_timestep_embedding(timesteps, dim):
    """
    Sinusoidal embedding for timesteps.
    timesteps: (B,) int32
    returns (B, dim) float32
    """
    timesteps = tf.cast(timesteps, tf.float32)
    half = dim // 2
    freqs = tf.exp(-math.log(10000.0) * tf.range(0, half, dtype=tf.float32) / float(half))
    args = tf.expand_dims(timesteps, -1) * tf.expand_dims(freqs, 0)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = tf.pad(emb, [[0,0],[0,1]])
    return emb

# -------------------------
# Simple FiLM + Cross-Attention Blocks (1D)
# -------------------------
class FiLMBlock(tf.keras.layers.Layer):
    def __init__(self, channels, time_emb_dim, cond_dim=None, name=None):
        super().__init__(name=name)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.act = tf.keras.layers.Activation("swish")
        self.conv1 = tf.keras.layers.Conv1D(channels, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(channels, 3, padding='same')
        self.to_scale_shift = tf.keras.layers.Dense(channels * 2)
        self.time_proj = tf.keras.layers.Dense(time_emb_dim, activation='swish')
        self.cond_proj = tf.keras.layers.Dense(time_emb_dim, activation='swish') if cond_dim is not None else None

    def call(self, x, t_emb, cond_emb=None):
        h = self.norm(x)
        h = self.act(h)
        h = self.conv1(h)
        te = self.time_proj(t_emb)
        if cond_emb is not None:
            ce = self.cond_proj(cond_emb)
            te = te + ce
        ss = self.to_scale_shift(te)
        scale, shift = tf.split(ss, 2, axis=-1)
        # expand to sequence length
        scale = tf.expand_dims(scale, axis=1)
        shift = tf.expand_dims(shift, axis=1)
        h = h * (1.0 + tf.cast(scale, h.dtype)) + tf.cast(shift, h.dtype)
        h = self.conv2(h)
        return h + x

class CrossAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, channels, num_heads=4, name=None):
        super().__init__(name=name)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels//num_heads)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x, cond_tokens):
        # x: (B, L, C)
        # cond_tokens: (B, S, E)
        if cond_tokens is None:
            return x
        att = self.mha(query=x, value=cond_tokens, key=cond_tokens)
        return self.norm(att + x)

# -------------------------
# UNet1D Model
# -------------------------
class UNet1D(tf.keras.Model):
    def __init__(self,
                 in_channels=LATENT_DIM,
                 out_channels=LATENT_DIM,
                 base_channels=BASE_CHANNELS,
                 channel_mults=CHANNEL_MULTS,
                 num_res_blocks=NUM_RES_BLOCKS,
                 time_emb_dim=TIME_EMB_DIM,
                 use_cross_attn=USE_CROSS_ATTENTION):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.time_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
        ])
        self.cond_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
        ])

        # stem
        self.stem = tf.keras.layers.Conv1D(base_channels, 3, padding='same')

        # down blocks
        self.downs = []
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = [FiLMBlock(out_ch, time_emb_dim, cond_dim=time_emb_dim) for _ in range(num_res_blocks)]
            attn = CrossAttentionBlock(out_ch) if (self.use_cross_attn and mult >= channel_mults[-1]) else None
            self.downs.append((blocks, attn))
        # bottleneck
        self.bottleneck1 = FiLMBlock(base_channels * channel_mults[-1], time_emb_dim, cond_dim=time_emb_dim)
        self.bottleneck_attn = CrossAttentionBlock(base_channels * channel_mults[-1]) if self.use_cross_attn else None
        self.bottleneck2 = FiLMBlock(base_channels * channel_mults[-1], time_emb_dim, cond_dim=time_emb_dim)

        # ups
        self.ups = []
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            blocks = [FiLMBlock(out_ch, time_emb_dim, cond_dim=time_emb_dim) for _ in range(num_res_blocks)]
            attn = CrossAttentionBlock(out_ch) if (self.use_cross_attn and mult >= channel_mults[-1]) else None
            self.ups.append((blocks, attn))

        # final
        self.final_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.final_act = tf.keras.layers.Activation("swish")
        self.final_conv = tf.keras.layers.Conv1D(base_channels, 3, padding='same')
        self.out_conv = tf.keras.layers.Conv1D(out_channels, 1, padding='same', dtype='float32')  # predict noise in float32

    def call(self, z, timesteps, cond_vector=None, cond_tokens=None, training=False):
        # z: (B, L, C)
        t_emb = get_timestep_embedding(timesteps, TIME_EMB_DIM)
        t_emb = self.time_mlp(t_emb)
        cond_proj = None
        if cond_vector is not None:
            cond_proj = self.cond_mlp(cond_vector)

        h = self.stem(z)
        skips = []
        for blocks, attn in self.downs:
            for b in blocks:
                h = b(h, t_emb, cond_proj)
            if attn is not None and cond_tokens is not None:
                h = attn(h, cond_tokens)
            skips.append(h)
            h = tf.keras.layers.AveragePooling1D(pool_size=2)(h)

        h = self.bottleneck1(h, t_emb, cond_proj)
        if self.bottleneck_attn is not None and cond_tokens is not None:
            h = self.bottleneck_attn(h, cond_tokens)
        h = self.bottleneck2(h, t_emb, cond_proj)

        for (blocks, attn), skip in zip(self.ups, reversed(skips)):
            # upsample by factor 2 (nearest)
            seq_len = tf.shape(h)[1] * 2
            h = tf.image.resize(tf.expand_dims(h, 2), [seq_len, tf.shape(h)[2]], method='nearest')[:, :, 0, :]
            # concat skip
            # if channels mismatch due to different shapes, conv can handle; we concat anyway
            # make sure shapes align
            if tf.shape(h)[1] != tf.shape(skip)[1]:
                # crop or pad
                minlen = tf.minimum(tf.shape(h)[1], tf.shape(skip)[1])
                h = h[:, :minlen, :]
                skip = skip[:, :minlen, :]
            h = tf.concat([h, skip], axis=-1)
            for b in blocks:
                h = b(h, t_emb, cond_proj)
            if attn is not None and cond_tokens is not None:
                h = attn(h, cond_tokens)

        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        out = self.out_conv(h)
        return out  # predicted noise

# -------------------------
# Simple fake dataset for PoC (replace this)
# -------------------------
def fake_dataset(batch_size=BATCH_SIZE, steps=10000):
    """
    Yields (surface_vector, cond_tokens, coil_latents)
    - surface_vector: (B, E) pooled cond vector
    - cond_tokens: (B, S, E) tokenized cond or None
    - coil_latents: (B, L, C)
    """
    E = 128
    S = 16
    while True:
        surface_vec = np.random.randn(batch_size, E).astype(np.float32)
        cond_tokens = np.random.randn(batch_size, S, E).astype(np.float32) if USE_CROSS_ATTENTION else None
        coil_latents = np.random.randn(batch_size, SEQ_LEN, LATENT_DIM).astype(np.float32)
        yield surface_vec, cond_tokens, coil_latents

# Wrap as tf.data
def make_tf_dataset():
    ds = tf.data.Dataset.from_generator(
        lambda: fake_dataset(BATCH_SIZE),
        output_types=(tf.float32, tf.float32 if USE_CROSS_ATTENTION else tf.float32, tf.float32),
        output_shapes=(
            (BATCH_SIZE, 128),
            (BATCH_SIZE, 16, 128) if USE_CROSS_ATTENTION else (BATCH_SIZE, 16, 128),
            (BATCH_SIZE, SEQ_LEN, LATENT_DIM)
        )
    )
    ds = ds.unbatch().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Model, optimizer, EMA
# -------------------------
strategy = tf.distribute.MirroredStrategy()  # change to MultiWorkerMirroredStrategy for multi-node
print("Num devices:", strategy.num_replicas_in_sync)

with strategy.scope():
    model = UNet1D()
    # base optimizer
    base_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(base_opt)
    # weight decay can be added via tf.keras.optimizers.experimental.AdamW in newer TF; keep simple here
    # EMA shadow vars
    ema_vars = [tf.Variable(v, trainable=False, dtype=v.dtype, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for v in model.trainable_variables]

# utility to update ema
@tf.function
def update_ema(ema_vars, model_vars, decay):
    for e, v in zip(ema_vars, model_vars):
        e.assign(e * decay + (1.0 - decay) * tf.cast(v, e.dtype))

# swap weights with EMA for eval/sampling
def load_ema_weights(model, ema_vars):
    orig = [v.read_value() for v in model.trainable_variables]
    for v, e in zip(model.trainable_variables, ema_vars):
        v.assign(tf.cast(e, v.dtype))
    return orig

def restore_weights(model, orig_vars):
    for v, o in zip(model.trainable_variables, orig_vars):
        v.assign(o)

# -------------------------
# Loss and training step
# -------------------------
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

@tf.function
def train_step(surface_vec, cond_tokens, clean_latents):
    """
    Performs one training step on a batch.
    """
    # sample random timesteps
    batch_size = tf.shape(clean_latents)[0]
    t = tf.random.uniform([batch_size], minval=0, maxval=NUM_TIMESTEPS, dtype=tf.int32)
    # sample noise
    noise = tf.random.normal(tf.shape(clean_latents), dtype=clean_latents.dtype)
    a_t = tf.gather(sqrt_alphas_cumprod, t)
    am1 = tf.gather(sqrt_one_minus_alphas_cumprod, t)
    a_t = tf.reshape(a_t, [batch_size, 1, 1])
    am1 = tf.reshape(am1, [batch_size, 1, 1])
    noisy = a_t * clean_latents + am1 * noise

    # classifier-free guidance masking: sometimes drop conditioning
    mask = tf.random.uniform([batch_size], 0, 1) < UNCOND_PROB
    # create cond vectors accordingly
    cond_vec_used = tf.where(tf.expand_dims(tf.cast(~mask, tf.float32), -1) == 1.0, surface_vec, tf.zeros_like(surface_vec))
    cond_tokens_used = None
    if USE_CROSS_ATTENTION and cond_tokens is not None:
        # drop tokens similarly
        cond_tokens_used = tf.where(tf.reshape(tf.cast(~mask, tf.float32), [batch_size, 1, 1]) == 1.0, cond_tokens, tf.zeros_like(cond_tokens))

    with tf.GradientTape() as tape:
        preds = model(noisy, t, cond_vector=cond_vec_used, cond_tokens=cond_tokens_used, training=True)
        # model may output float32; ensure same dtype
        preds = tf.cast(preds, noise.dtype)
        loss_per_elem = mse(noise, preds)
        loss = tf.reduce_mean(loss_per_elem)

        # scale the loss for mixed precision
        scaled_loss = opt.get_scaled_loss(loss)

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = opt.get_unscaled_gradients(scaled_grads)
    # gradient clipping/coalesce
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    # update EMA
    update_ema(ema_vars, model.trainable_variables, EMA_DECAY)
    return loss

# -------------------------
# Sampling (DDPM ancestral sampling) with classifier-free guidance
# -------------------------
def sample(model, cond_vector=None, cond_tokens=None, batch_size=1, guidance_scale=1.5):
    """
    Simple DDPM sampling loop using model to predict noise eps.
    cond_vector: (B, E) or None
    cond_tokens: (B, S, E) or None
    """
    x = tf.random.normal([batch_size, SEQ_LEN, LATENT_DIM], dtype=tf.float32)
    for t_ in reversed(range(NUM_TIMESTEPS)):
        t = tf.fill([batch_size], t_)  # vector
        eps_pred = model(x, t, cond_vector=cond_vector, cond_tokens=cond_tokens, training=False)
        # classifier-free guidance: combine cond and uncond predictions if cond provided
        if cond_vector is not None or cond_tokens is not None:
            # compute uncond prediction
            eps_uncond = model(x, t, cond_vector=tf.zeros_like(cond_vector) if cond_vector is not None else None,
                               cond_tokens=tf.zeros_like(cond_tokens) if cond_tokens is not None else None, training=False)
            eps_pred = eps_uncond + guidance_scale * (eps_pred - eps_uncond)
        # compute posterior mean (simple DDPM step)
        beta_t = betas[t_]
        alpha_t = alphas[t_]
        alpha_cum = alphas_cumprod[t_]
        alpha_cum_prev = alphas_cumprod_prev[t_]
        # following ddpm posterior formula: mean = 1/sqrt(alpha_t) * (x - beta_t/sqrt(1-alpha_cum) * eps_pred)
        coef1 = 1.0 / math.sqrt(alpha_t)
        coef2 = beta_t / math.sqrt(1.0 - alpha_cum)
        mean = coef1 * (x - coef2 * eps_pred)
        if t_ > 0:
            noise = tf.random.normal(tf.shape(x))
            sigma = math.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean
    return x

# -------------------------
# Training loop (PoC)
# -------------------------
def train():
    ds = make_tf_dataset()
    it = iter(ds)
    for epoch in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            surface_vec, cond_tokens, coil_latents = next(it)
            loss = train_step(surface_vec, cond_tokens if USE_CROSS_ATTENTION else None, coil_latents)
            if step % 50 == 0:
                print(f"Epoch {epoch} step {step} loss {loss.numpy():.6f}")
        # End epoch: save checkpoint and run a small sample using EMA weights
        ckpt_dir = "./checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        # save model (weights in float32)
        model.save_weights(os.path.join(ckpt_dir, f"model_epoch_{epoch}"))
        # sample with EMA weights
        orig = load_ema_weights(model, ema_vars)
        samples = sample(model, cond_vector=np.random.randn(1, 128).astype(np.float32) if True else None,
                         cond_tokens=np.random.randn(1, 16, 128).astype(np.float32) if USE_CROSS_ATTENTION else None,
                         batch_size=1, guidance_scale=1.3)
        # restore original weights
        restore_weights(model, orig)
        print(f"Epoch {epoch} sample shape: {samples.shape}")

if __name__ == "__main__":
    train()