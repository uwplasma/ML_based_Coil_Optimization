# ldm_unet_ddim_tfrecord.py
import tensorflow as tf
import numpy as np
import math
import os
from transformers import SelfAttentionBlock, PoolingByMultiheadAttention

# -------------------------
# USER-CONFIGURABLE CONSTANTS (set these to match your data / TFRecords)
# -------------------------
TFRECORD_FILES = ["mini_latents_tfrecords/augmented_000.tfrecord"]  # replace with your list
BATCH_SIZE = 8
SEQ_LEN = 128         # L: desired sequence length fed to UNet
LATENT_DIM = 64        # C: latent vector channel size
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4)
NUM_RES_BLOCKS = 2
TIME_EMB_DIM = 256
USE_CROSS_ATTENTION = True
EPOCHS = 10
STEPS_PER_EPOCH = 200
LEARNING_RATE = 2e-4
UNCOND_PROB = 0.12
EMA_DECAY = 0.9999
NUM_TIMESTEPS = 1000
DDIM_ETA = 0.0  # 0.0 => deterministic DDIM, >0 adds stochasticity (0<=eta<=1)
PREFETCH = tf.data.AUTOTUNE

# TFRecord schema constants used by parse_tfrecord_fn
MAX_COILS = 6
MAX_SETS = 441
TOTAL_ROWS = MAX_COILS + 1              # placeholder; set to your TOTAL_ROWS
FEATURES_PER_COIL = 100       # placeholder; set accordingly
TOTAL_SETS = MAX_SETS + 1              # placeholder; set to your TOTAL_SETS
FEATURES_PER_SET = 4        # placeholder; set accordingly


# -------------------------
# Mixed precision
# -------------------------
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# -------------------------
# Noise schedule (DDPM-style linear betas -> we use alphas_cumprod for DDIM)
# -------------------------
def make_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

betas = make_beta_schedule(NUM_TIMESTEPS)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)  # alpha_bar_t
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

betas_t = tf.constant(betas, dtype=tf.float32)
alphas_cumprod_t = tf.constant(alphas_cumprod, dtype=tf.float32)
sqrt_alphas_cumprod_t = tf.constant(sqrt_alphas_cumprod, dtype=tf.float32)
sqrt_one_minus_alphas_cumprod_t = tf.constant(sqrt_one_minus_alphas_cumprod, dtype=tf.float32)

# -------------------------
# TFRecord parse function (your supplied function, integrated)
# -------------------------
def parse_tfrecord_fn(example_proto):
    feature_description = {
        "coil_data": tf.io.FixedLenFeature([TOTAL_ROWS * FEATURES_PER_COIL], tf.float32),
        "coil_mask": tf.io.FixedLenFeature([MAX_COILS], tf.int64),
        "surface_data": tf.io.FixedLenFeature([TOTAL_SETS * FEATURES_PER_SET], tf.float32),
        "surface_mask": tf.io.FixedLenFeature([MAX_SETS], tf.int64),
        "coil_latent": tf.io.FixedLenFeature([LATENT_DIM], tf.float32)  # as provided
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    coil_data = tf.reshape(parsed["coil_data"], [TOTAL_ROWS, FEATURES_PER_COIL])
    coil_mask = parsed["coil_mask"]
    surface_data = tf.reshape(parsed["surface_data"], [TOTAL_SETS, FEATURES_PER_SET])
    surface_mask = parsed["surface_mask"]
    coil_latent = parsed["coil_latent"]

    inputs = {
        "coil_data": coil_data,      # (TOTAL_ROWS, FEATURES_PER_COIL)
        "coil_mask": coil_mask,      # (MAX_COILS,)
        "surface_data": surface_data,
        "surface_mask": surface_mask,
    }

    target = {
        "coil": coil_data,
        "surface": surface_data,
        "coil_latent": coil_latent
    }
    return inputs, target

# -------------------------
# Data pipeline: map parse -> format for model
# -------------------------
def make_dataset(tfrecord_files, batch_size=BATCH_SIZE, shuffle_buffer=2048):
    ds = tf.data.TFRecordDataset(tfrecord_files)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=PREFETCH)

    def _format(inputs, target):
        # target['coil_latent'] shape currently: (LATENT_DIM,) as per your parse function
        coil_latent = target["coil_latent"]  # (LATENT_DIM,)

        # Two possible cases are handled:
        # 1) coil_latent length == SEQ_LEN * LATENT_DIM (flattened sequence): reshape to (SEQ_LEN, LATENT_DIM)
        # 2) coil_latent length == LATENT_DIM: treat as global vector, expand to sequence by repeating
        # Because parse_tfrecord_fn guaranteed shape [LATENT_DIM], we check the actual vector size and act accordingly.
        # NOTE: If your TFRecord truly stores a flattened sequence, modify the parse function to store length SEQ_LEN*LATENT_DIM
        flat_len = tf.shape(coil_latent)[0]

        # We implement the following logic: if flat_len == SEQ_LEN*LATENT_DIM -> reshape; elif flat_len==LATENT_DIM -> tile
        # Because parse defined fixed size LATENT_DIM, two cases cannot both be handled without changing TFRecord creation.
        # Here we implement the conservative expansion:
        def reshape_to_seq():
            seq = tf.reshape(coil_latent, [SEQ_LEN, LATENT_DIM])  # user must ensure this matches underlying layout
            return seq

        def tile_to_seq():
            # expand and tile across sequence positions
            seq = tf.expand_dims(coil_latent, axis=0)  # (1, LATENT_DIM)
            seq = tf.tile(seq, [SEQ_LEN, 1])           # (SEQ_LEN, LATENT_DIM)
            return seq

        # safe branch: if flat_len equals LATENT_DIM (expected given your parse), tile_to_seq is used
        coil_seq = tf.cond(tf.equal(flat_len, LATENT_DIM),
                           true_fn=tile_to_seq,
                           false_fn=reshape_to_seq)

        # cast everything
        coil_seq = tf.cast(coil_seq, tf.float32)

        # surface features go to surface encoder (we pass them through as-is)
        surface_data = inputs["surface_data"]
        surface_mask = inputs["surface_mask"]

        # final formatting
        # return: (surface_data, surface_mask, cond placeholder), coil_seq
        # cond placeholders are computed in training loop by the surface encoder so dataset only provides raw surface_data
        return (surface_data, surface_mask, coil_seq), coil_seq

    ds = ds.map(_format, num_parallel_calls=PREFETCH)
    ds = ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(PREFETCH)
    return ds

# -------------------------
# Minimal surface encoder (placeholder)
# Replace with your real surface encoder; it must output:
#  - pooled vector cond_vector: shape (B, E)
#  - token sequence cond_tokens: shape (B, S, E_token)
# We'll build a small MLP + projection to tokens; this is only a starting point.
# -------------------------
def load_surface_encoder():
    return tf.keras.models.load_model("coil_logs/20250812-152016_trial/encoder.keras", #change the file to the correct 
    custom_objects={"SelfAttentionBlock": SelfAttentionBlock,                          #surface_log
                    'PoolingByMultiheadAttention': PoolingByMultiheadAttention},
    safe_mode=False)

def make_surface_encoder(pooled_dim=128, token_dim=128, token_len=16):
    surface_in = tf.keras.Input(shape=(TOTAL_SETS, FEATURES_PER_SET), name="surface_in")
    mask_in = tf.keras.Input(shape=(MAX_SETS,), dtype=tf.int32, name="mask_in")

    # simple per-set MLP
    x = tf.keras.layers.Dense(256, activation="swish")(surface_in)
    x = tf.keras.layers.Dense(256, activation="swish")(x)   # (B, TOTAL_SETS, 256)

    # pooled vector (mean pooling respecting mask)
    # make mask float
    mask_float = tf.cast(tf.expand_dims(mask_in, -1), tf.float32)  # (B, MAX_SETS, 1)
    # if shapes mismatch (TOTAL_SETS vs MAX_SETS) user should ensure they match or adjust above
    pooled = tf.reduce_sum(x * mask_float, axis=1) / (tf.reduce_sum(mask_float, axis=1) + 1e-6)
    pooled = tf.keras.layers.Dense(pooled_dim, activation="swish", name="cond_vector")(pooled)

    # tokens: project per-set features to token_dim and optionally reduce to token_len
    token_proj = tf.keras.layers.Dense(token_dim, activation=None)(x)  # (B, TOTAL_SETS, token_dim)
    # if TOTAL_SETS != token_len, simple linear projection to token_len (learned)
    if TOTAL_SETS != token_len:
        # transpose -> (B, token_dim, TOTAL_SETS), dense to token_len, transpose back
        t = tf.transpose(token_proj, [0, 2, 1])
        t = tf.keras.layers.Dense(token_len)(t)
        cond_tokens = tf.transpose(t, [0, 2, 1])  # (B, token_len, token_dim)
    else:
        cond_tokens = token_proj

    return tf.keras.Model(inputs=[surface_in, mask_in], outputs=[pooled, cond_tokens], name="surface_encoder")

# -------------------------
# UNet model is the same as previously provided, compacted here to focus on TFRecord + DDIM changes.
# (Use the UNet1D/FiLM/CrossAttention definitions from the previous scaffold.)
# For brevity we reuse function/class definitions (you should paste the UNet1D, FiLMBlock, CrossAttentionBlock code here).
# -------------------------
# --- Paste UNet1D, FiLMBlock, CrossAttentionBlock, get_timestep_embedding here ---
# For brevity in this snippet, assume we've already defined them as in prior file.

# To keep this file runnable, minimal versions are supplied below:
def get_timestep_embedding(timesteps, dim):
    timesteps = tf.cast(timesteps, tf.float32)
    half = dim // 2
    freqs = tf.exp(-math.log(10000.0) * tf.range(0, half, dtype=tf.float32) / float(half))
    args = tf.expand_dims(timesteps, -1) * tf.expand_dims(freqs, 0)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = tf.pad(emb, [[0,0],[0,1]])
    return emb

# Minimal FiLM + CrossAttention and UNet1D classes (copy full definitions from earlier scaffold in production)
class FiLMBlock(tf.keras.layers.Layer):
    def __init__(self, channels, time_emb_dim, cond_dim=None):
        super().__init__()
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
        scale = tf.expand_dims(scale, axis=1)
        shift = tf.expand_dims(shift, axis=1)
        h = h * (1.0 + tf.cast(scale, h.dtype)) + tf.cast(shift, h.dtype)
        h = self.conv2(h)
        return h + x

class CrossAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels//num_heads)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
    def call(self, x, cond_tokens):
        if cond_tokens is None:
            return x
        att = self.mha(query=x, value=cond_tokens, key=cond_tokens)
        return self.norm(att + x)

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
        self.time_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
        ])
        self.cond_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
        ])
        self.stem = tf.keras.layers.Conv1D(base_channels, 3, padding='same')
        self.downs = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = [FiLMBlock(out_ch, time_emb_dim, cond_dim=time_emb_dim) for _ in range(num_res_blocks)]
            attn = CrossAttentionBlock(out_ch) if (use_cross_attn and mult >= channel_mults[-1]) else None
            self.downs.append((blocks, attn))
        self.bneck1 = FiLMBlock(base_channels * channel_mults[-1], time_emb_dim, cond_dim=time_emb_dim)
        self.bneck_attn = CrossAttentionBlock(base_channels * channel_mults[-1]) if use_cross_attn else None
        self.bneck2 = FiLMBlock(base_channels * channel_mults[-1], time_emb_dim, cond_dim=time_emb_dim)
        self.ups = []
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            blocks = [FiLMBlock(out_ch, time_emb_dim, cond_dim=time_emb_dim) for _ in range(num_res_blocks)]
            attn = CrossAttentionBlock(out_ch) if (use_cross_attn and mult >= channel_mults[-1]) else None
            self.ups.append((blocks, attn))
        self.final_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.final_act = tf.keras.layers.Activation("swish")
        self.final_conv = tf.keras.layers.Conv1D(base_channels, 3, padding='same')
        self.out_conv = tf.keras.layers.Conv1D(out_channels, 1, padding='same', dtype='float32')
    def call(self, z, timesteps, cond_vector=None, cond_tokens=None, training=False):
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
        h = self.bneck1(h, t_emb, cond_proj)
        if self.bneck_attn is not None and cond_tokens is not None:
            h = self.bneck_attn(h, cond_tokens)
        h = self.bneck2(h, t_emb, cond_proj)
        for (blocks, attn), skip in zip(self.ups, reversed(skips)):
            seq_len = tf.shape(h)[1] * 2
            h = tf.image.resize(tf.expand_dims(h, 2), [seq_len, tf.shape(h)[2]], method='nearest')[:, :, 0, :]
            if tf.shape(h)[1] != tf.shape(skip)[1]:
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
        return out

# -------------------------
# Prepare strategy, models, optimizer, EMA
# -------------------------
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet = UNet1D()
    surface_encoder = make_surface_encoder(pooled_dim=128, token_dim=128, token_len=16)
    base_opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(base_opt)

    # EMA shadow variables:
    ema_vars = [tf.Variable(v, trainable=False, dtype=v.dtype, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
                for v in unet.trainable_variables + surface_encoder.trainable_variables]

@tf.function
def update_ema(ema_vars, model_vars, decay):
    for e, v in zip(ema_vars, model_vars):
        e.assign(e * decay + (1.0 - decay) * tf.cast(v, e.dtype))

def load_ema_weights(unet, surface_encoder, ema_vars):
    model_vars = unet.trainable_variables + surface_encoder.trainable_variables
    orig = [v.read_value() for v in model_vars]
    for v, e in zip(model_vars, ema_vars):
        v.assign(tf.cast(e, v.dtype))
    return orig

def restore_weights(unet, surface_encoder, orig_vars):
    model_vars = unet.trainable_variables + surface_encoder.trainable_variables
    for v, o in zip(model_vars, orig_vars):
        v.assign(o)

# -------------------------
# Training loss and step
# -------------------------
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

@tf.function
def train_step(surface_data, surface_mask, clean_latents):
    # surface_data: (B, TOTAL_SETS, FEATURES_PER_SET)
    # surface_mask: (B, MAX_SETS)
    # clean_latents: (B, SEQ_LEN, LATENT_DIM)
    batch_size = tf.shape(clean_latents)[0]
    t = tf.random.uniform([batch_size], 0, NUM_TIMESTEPS, dtype=tf.int32)

    # sample noise and produce noisy input per DDPM forward
    noise = tf.random.normal(tf.shape(clean_latents), dtype=clean_latents.dtype)
    a_t = tf.gather(sqrt_alphas_cumprod_t, t)
    am1 = tf.gather(sqrt_one_minus_alphas_cumprod_t, t)
    a_t = tf.reshape(a_t, [batch_size, 1, 1])
    am1 = tf.reshape(am1, [batch_size, 1, 1])
    noisy = a_t * clean_latents + am1 * noise

    # compute conditioning embeddings from surface encoder
    cond_vector, cond_tokens = surface_encoder([surface_data, surface_mask], training=True)

    # classifier-free masking
    mask = tf.random.uniform([batch_size], 0.0, 1.0) < UNCOND_PROB
    cond_vector_used = tf.where(tf.expand_dims(tf.cast(~mask, tf.float32), -1) == 1.0, cond_vector, tf.zeros_like(cond_vector))
    cond_tokens_used = None
    if USE_CROSS_ATTENTION:
        cond_tokens_used = tf.where(tf.reshape(tf.cast(~mask, tf.float32), [batch_size, 1, 1]) == 1.0,
                                    cond_tokens,
                                    tf.zeros_like(cond_tokens))

    with tf.GradientTape() as tape:
        preds = unet(noisy, t, cond_vector=cond_vector_used, cond_tokens=cond_tokens_used, training=True)
        preds = tf.cast(preds, noise.dtype)
        loss_per_elem = mse(noise, preds)
        loss = tf.reduce_mean(loss_per_elem)
        scaled_loss = opt.get_scaled_loss(loss)

    scaled_grads = tape.gradient(scaled_loss, unet.trainable_variables + surface_encoder.trainable_variables)
    grads = opt.get_unscaled_gradients(scaled_grads)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    opt.apply_gradients(zip(grads, unet.trainable_variables + surface_encoder.trainable_variables))

    # update EMA for both models' variables
    update_ema(ema_vars, unet.trainable_variables + surface_encoder.trainable_variables, EMA_DECAY)
    return loss

# -------------------------
# DDIM sampler
# -------------------------
def ddim_sample(unet, cond_vector=None, cond_tokens=None, batch_size=1, guidance_scale=1.0, eta=DDIM_ETA):
    # Precompute necessary arrays
    alphas_cum = alphas_cumprod  # numpy array
    seq = list(range(0, NUM_TIMESTEPS))
    x = np.random.randn(batch_size, SEQ_LEN, LATENT_DIM).astype(np.float32)

    for i in reversed(range(NUM_TIMESTEPS)):
        t = np.full((batch_size,), i, dtype=np.int32)
        t_tf = tf.convert_to_tensor(t, dtype=tf.int32)
        # predict epsilon
        eps = unet(x, t_tf, cond_vector=cond_vector, cond_tokens=cond_tokens, training=False).numpy()
        if cond_vector is not None or cond_tokens is not None:
            # unconditioned preds
            eps_uncond = unet(x, t_tf, cond_vector=np.zeros_like(cond_vector) if cond_vector is not None else None,
                              cond_tokens=np.zeros_like(cond_tokens) if cond_tokens is not None else None, training=False).numpy()
            eps = eps_uncond + guidance_scale * (eps - eps_uncond)

        a_t = alphas_cum[i]
        sqrt_a_t = math.sqrt(a_t)
        sqrt_1_a_t = math.sqrt(1.0 - a_t)
        # estimate x0
        x0_pred = (x - sqrt_1_a_t * eps) / sqrt_a_t

        if i == 0:
            x = x0_pred
            break

        a_prev = alphas_cum[i - 1]
        # DDIM deterministic update (eta=0)
        sigma = eta * np.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        # direction pointing to x_t
        dir_xt = np.sqrt(1 - a_prev - sigma**2) * eps
        noise = sigma * np.random.randn(*x.shape).astype(np.float32) if sigma > 0 else 0.0
        x = np.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x

# -------------------------
# Training loop
# -------------------------
def train():
    ds = make_dataset(TFRECORD_FILES, batch_size=BATCH_SIZE)
    it = iter(ds)
    for epoch in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            (surface_data, surface_mask, coil_latents), _ = next(it)
            loss = train_step(surface_data, surface_mask, coil_latents)
            if step % 50 == 0:
                tf.print("Epoch", epoch, "step", step, "loss", loss)
        # checkpointing & sample with EMA
        ckpt_dir = "./checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        unet.save_weights(os.path.join(ckpt_dir, f"unet_epoch_{epoch}"))
        surface_encoder.save_weights(os.path.join(ckpt_dir, f"surfenc_epoch_{epoch}"))
        # sample using EMA weights
        orig = load_ema_weights(unet, surface_encoder, ema_vars)
        cond_vec = np.random.randn(1, 128).astype(np.float32)
        cond_tokens = np.random.randn(1, 16, 128).astype(np.float32) if USE_CROSS_ATTENTION else None
        samples = ddim_sample(unet, cond_vector=cond_vec, cond_tokens=cond_tokens, batch_size=1, guidance_scale=1.2, eta=DDIM_ETA)
        restore_weights(unet, surface_encoder, orig)
        tf.print("Epoch", epoch, "sample shape", np.shape(samples))

if __name__ == "__main__":
    train()
