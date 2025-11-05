import tensorflow as tf
tf.constant(1.0)  # Trigger basic op
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from tensorflow.keras import layers
import keras

@tf.keras.utils.register_keras_serializable(package="my_layers")
class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim//num_heads,
            dropout=dropout
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout)
        ])
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, mask=None, training=False):
        attn_mask = None

        if mask is not None:
            # MultiHeadAttention expects (B, T) → (B, 1, T)
            attn_mask = mask[:, tf.newaxis, :]
            
        attn_out = self.attn(x, x, attention_mask=attn_mask, training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + ffn_out)
    
    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Future-proof: allows ignoring extra keys if constructor changes
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="my_layers")
class PoolingByMultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, num_seeds=1, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_seeds = int(num_seeds)
        self.dropout_rate = dropout

        # Seed vectors
        self.seed_vectors = self.add_weight(
            shape=(self.num_seeds, self.embed_dim),
            initializer='random_normal',
            trainable=True,
            name='pma_seed_vectors'
        )

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        B = tf.shape(inputs)[0]
        seed_tiled = tf.tile(tf.expand_dims(self.seed_vectors, axis=0), [B, 1, 1])  # (B, num_seeds, D)
        return self.mha(query=seed_tiled, value=inputs, key=inputs, training=training)
    
    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_seeds": self.num_seeds,
            "dropout": self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@tf.keras.utils.register_keras_serializable(package="my_layers")
class LatentSlots(tf.keras.layers.Layer):
    def __init__(self, num_slots: int, dim: int, **kw):
        super().__init__(**kw)
        self.num_slots = num_slots
        self.proj = tf.keras.layers.Dense(num_slots * dim, activation=None)

    def call(self, z):  # z: (B, D)
        x = self.proj(z)
        return tf.reshape(x, (-1, self.num_slots, x.shape[-1] // self.num_slots))  # (B, K, D)
    
@tf.keras.utils.register_keras_serializable(package="my_layers")
class LatentTile(layers.Layer):
    """
    Holds learnable latents (L, D) and tiles them to (B, L, D) at call-time.
    """
    def __init__(self, num_latents, latent_dim, initializer="truncated_normal", name="latent_tile", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.initializer = initializer

    def build(self, input_shape):
        # create a trainable (L, D) latent matrix
        self.latents = self.add_weight(
            shape=(self.num_latents, self.latent_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="learnable_latents",
        )
        super().build(input_shape)

    def call(self, x, training=False):
        # x is any tensor with batch dimension, so tile latents to batch
        b = tf.shape(x)[0]
        lat_tiled = tf.tile(tf.expand_dims(self.latents, axis=0), [b, 1, 1])  # (B, L, D)
        return lat_tiled
    
    @classmethod
    def from_config(cls, config):
        # Future-proof: allows ignoring extra keys if constructor changes
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="my_layers")
class PerceiverBlock(layers.Layer):
    """
    One Perceiver block: cross-attention (latents query inputs) -> latent self-attn -> FFN.
    Inputs:
      - latents : (B, L, D)
      - inputs  : (B, N, D)
      - attn_mask: (B, 1, N) or None
    """
    def __init__(self, latent_dim, num_heads, ff_dim, dropout=0.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)

        # cross-attention: latents query inputs
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.latent_dim // max(1, self.num_heads), dropout=self.dropout
        )
        self.cross_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_ffn = keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.latent_dim),
            layers.Dropout(self.dropout)
        ])
        self.cross_norm2 = layers.LayerNormalization(epsilon=1e-6)

        # latent self-attention + FFN
        self.self_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.latent_dim // max(1, self.num_heads), dropout=self.dropout
        )
        self.self_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.self_ffn = keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.latent_dim),
            layers.Dropout(self.dropout)
        ])
        self.self_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, latents, inputs, training, attn_mask=None):
        # Cross-attention: queries=latents, keys/values=inputs
        cross_out = self.cross_attn(query=latents, value=inputs, key=inputs, attention_mask=attn_mask, training=training)
        lat = latents + cross_out
        lat = self.cross_norm1(lat)
        lat = lat + self.cross_ffn(lat, training=training)
        lat = self.cross_norm2(lat)

        # Latent self-attention
        self_attn_out = self.self_attn(query=lat, value=lat, key=lat, training=training)
        lat = lat + self_attn_out
        lat = self.self_norm1(lat)
        lat = lat + self.self_ffn(lat, training=training)
        lat = self.self_norm2(lat)

        return lat
    
    @classmethod
    def from_config(cls, config):
        # Future-proof: allows ignoring extra keys if constructor changes
        return cls(**config)

    
def TransformerEncoder(max_sets, features_per_set, name, embed_dim, num_heads, ff_dim, num_sab_blocks, 
                       dropout=0.1):
    
    data_type = name+'_data'
    mask_type = name+'_mask'
    model_name = name+'_encoder'

    input = tf.keras.Input(shape=(max_sets, features_per_set), name=data_type)
    mask = tf.keras.Input(shape=(max_sets,), dtype=tf.float32, name=mask_type)
    
    # Project input coils to embed_dim
    x = layers.Dense(embed_dim)(input)

    # ones = keras.layers.Lambda(lambda x: tf.ones_like(x[:, :1]), output_shape=(1,))(mask)  # shape (B, 1)
    # full_mask = keras.layers.Concatenate(axis=1)([mask, ones])

    sab_blocks = [
            SelfAttentionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_sab_blocks)
        ]

    for sab in sab_blocks:
        x = sab(x, mask=mask)


    # Pool with PMA
    PMA = PoolingByMultiheadAttention(embed_dim, num_heads, dropout=dropout)
    pooled = PMA(x) # (B, num_seeds, embed_dim)

    
    return tf.keras.Model(inputs={data_type: input, mask_type: mask}, outputs=[pooled, x], name=model_name)

def PerceiverEncoder(
    max_sets,
    features_per_set,
    name,
    embed_dim,
    num_heads,
    ff_dim,
    num_latents=32,
    num_blocks=3,
    dropout=0.1,
):
    """
    Returns a functional Keras Model matching your TransformerEncoder API:
      inputs: { name+'_data': Input(shape=(max_sets+1, features_per_set)),
                name+'_mask': Input(shape=(max_sets,)) }
      outputs: [pooled, tokens] where tokens shape (B, num_latents, embed_dim)
    """
    data_type = name + "_data"
    mask_type = name + "_mask"
    model_name = name + "_encoder"

    # Inputs
    input_data = tf.keras.Input(shape=(max_sets + 1, features_per_set), name=data_type)
    mask = tf.keras.Input(shape=(max_sets,), dtype=tf.float32, name=mask_type)

    # Input projection
    x = layers.Dense(embed_dim, name=f"{name}_input_proj")(input_data)  # (B, N+1, D)

    # Build attention mask for MHA: convert (B, N) -> (B, 1, N) (Keras MHA expects shape broadcastable)
    # We'll keep the final scalar token included: make a scalar-one appended to mask as you previously did.
    ones = layers.Lambda(lambda m: tf.ones_like(m[:, :1]), output_shape=(1,), name=f"{name}_scalar_mask")(mask)
    full_mask = layers.Concatenate(axis=1, name=f"{name}_full_mask")([mask, ones])  # (B, N+1)
    attn_mask = layers.Lambda(lambda m: tf.expand_dims(tf.cast(m, tf.bool), axis=1), name=f"{name}_attn_mask")(full_mask)
    # print("attn_mask shape:", attn_mask.shape)  # None,1,442 (query_len is not present, it's broadcastable)
    # attn_mask shape: (B, 1, N+1) -> broadcastable to (B, num_queries, N+1)

    # Create latent tile layer (adds trainable latents and tiles to batch inside call)
    latent_tile = LatentTile(num_latents=num_latents, latent_dim=embed_dim, name=f"{name}_latent_tile")
    latents = latent_tile(x)  # (B, L, D)

    # Apply multiple Perceiver blocks
    lat = latents
    for i in range(num_blocks):
        block = PerceiverBlock(latent_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, name=f"{name}_pblock_{i}")
        lat = block(lat, x, attn_mask, training=None)  # (B, L, D)

    # final normalization
    lat = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_norm")(lat)

    PMA = PoolingByMultiheadAttention(embed_dim, num_heads, dropout=dropout)
    pooled = PMA(x) # (B, num_seeds, embed_dim)

    return tf.keras.Model(
        inputs={data_type: input_data, mask_type: mask},
        outputs=[pooled, lat],
        name=model_name
    )

@tf.keras.utils.register_keras_serializable(package="my_layers")
class LearnedQueryDecoder(tf.keras.layers.Layer):
    def __init__(self, max_sets, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.max_sets = max_sets
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.learned_queries = self.add_weight(
            shape=(max_sets, embed_dim), initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), 
            trainable=True, name='learned_queries')                                                        

        self.attn_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, 
                                      key_dim=embed_dim//num_heads, 
                                      dropout=dropout)
            for _ in range(num_layers)
        ]

        self.norm1_layers = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.norm2_layers = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]

        self.ffn_layers = [
            tf.keras.Sequential([
                layers.Dense(ff_dim, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                # layers.Dropout(dropout)
            ])
            for _ in range(num_layers)
        ]

        # FiLM from z
        self.film_g1 = [tf.keras.layers.Dense(embed_dim) for _ in range(num_layers)]
        self.film_b1 = [tf.keras.layers.Dense(embed_dim) for _ in range(num_layers)]
        self.film_g2 = [tf.keras.layers.Dense(embed_dim) for _ in range(num_layers)]
        self.film_b2 = [tf.keras.layers.Dense(embed_dim) for _ in range(num_layers)]

    def _film(self, x, z, g_layer, b_layer):
        gamma = g_layer(z)[:, None, :]  # (B,1,D)
        beta  = b_layer(z)[:, None, :]
        return x * (1.0 + gamma) + beta

    def call(self, tokens, z, training=False):
        tf.debugging.assert_rank(tokens, 3)
        tf.debugging.assert_greater(tf.shape(tokens)[1], 1, "Decoder got T_ctx=1; pass full token sequence.")

        B = tf.shape(tokens)[0]
        x = tf.tile(self.learned_queries[tf.newaxis, :, :], [B, 1, 1])

        for i in range(self.num_layers):
            # Cross-attention
            attn_out, attn_weights = self.attn_layers[i](
                x, tokens, return_attention_scores=True, training=training
            )
            x = self.norm1_layers[i](x + self._film(attn_out, z, self.film_g1[i], self.film_b1[i]))

            # Feed-forward network
            ff_out = self.ffn_layers[i](x, training=training)
            x = self.norm2_layers[i](x + self._film(ff_out, z, self.film_g2[i], self.film_b2[i]))
        
        return x  # shape: (B, max_coils, D)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_sets": self.max_sets,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
def TransformerDecoder(name, embed_dim, num_heads, ff_dim, num_layers, 
                       max_sets=6, features_per_set=100, dropout=0.1, num_slots=8):
    
    decoder_name = name+'_decoder'
    encoded_tokens = tf.keras.Input(shape=(None, embed_dim), name='encoded_tokens')
    encoded_latent = tf.keras.Input(shape=(1, embed_dim),        name="encoded_latent")

    # squeeze latent to (B,D) inside the graph
    latent = layers.Reshape((embed_dim,), name="squeeze_latent")(encoded_latent)   # (B,D)

    # slots = LatentSlots(num_slots=num_slots, dim=embed_dim, name="latent_slots")(encoded_input)  # (B, K, D)
    
    decoder = LearnedQueryDecoder(max_sets, embed_dim, num_heads, ff_dim, num_layers, dropout)
    decoded = decoder(encoded_tokens, latent)

    # Project back to original coil feature dimension
    output = layers.Dense(features_per_set)(decoded)
    
    return tf.keras.Model(inputs=[encoded_tokens, encoded_latent], outputs=output, name=decoder_name)