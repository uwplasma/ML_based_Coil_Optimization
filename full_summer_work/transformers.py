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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # Nothing dynamic here, but we call build() on sub-layers for safety
        # self.attn.build([input_shape, input_shape, input_shape])
        # self.norm1.build(input_shape)
        # self.ffn.build(input_shape)
        # self.norm2.build(input_shape)
        super().build(input_shape)

    def call(self, x, mask=None, training=False):
        if mask is not None:
            # MultiHeadAttention expects (B, T) → (B, 1, T)
            attn_mask = mask[:, tf.newaxis, :]
        else:
            attn_mask = None

        attn_out = self.attn(x, x, attention_mask=attn_mask, training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + ffn_out)

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
        # Seed vectors depend only on embed_dim
        # self.seed_vectors = self.add_weight(
        #     shape=(self.num_seeds, self.embed_dim),
        #     initializer="random_normal",
        #     trainable=True,
        #     name="pma_seed_vectors"
        # )
        # self.mha.build([tf.TensorShape([None, self.num_seeds, self.embed_dim]), input_shape])
        super().build(input_shape)

    def call(self, inputs, training=False):
        B = tf.shape(inputs)[0]
        seed_tiled = tf.tile(tf.expand_dims(self.seed_vectors, axis=0), [B, 1, 1])  # (B, num_seeds, D)
        return self.mha(query=seed_tiled, value=inputs, key=inputs, training=training)

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
    
def TransformerEncoder(input, mask, embed_dim, num_heads, ff_dim, num_sab_blocks, 
                       dropout=0.1):
    
    
    # Project input coils to embed_dim
    x = layers.Dense(embed_dim)(input)

    ones = keras.layers.Lambda(lambda x: tf.ones_like(x[:, :1]), output_shape=(1,))(mask)  # shape (B, 1)
    full_mask = keras.layers.Concatenate(axis=1)([mask, ones])

    sab_blocks = [
            SelfAttentionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_sab_blocks)
        ]

    for sab in sab_blocks:
        x = sab(x, mask=full_mask)


    # Pool with PMA
    PMA = PoolingByMultiheadAttention(embed_dim, num_heads, dropout=dropout)
    pooled = PMA(x) # (B, num_seeds, embed_dim)

    return pooled, x # pooled for FiLM, x for cross-attention (in UNet LDM training)

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
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
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

    def call(self, encoded_set, training=False):
        B = tf.shape(encoded_set)[0]
        x = tf.tile(self.learned_queries[tf.newaxis, :, :], [B, 1, 1])

        for i in range(self.num_layers):
            # Cross-attention
            attn_out, attn_weights = self.attn_layers[i](
                x, encoded_set, return_attention_scores=True, training=training
            )
            x = self.norm1_layers[i](x + attn_out)

            # Feed-forward network
            ff_out = self.ffn_layers[i](x, training=training)
            x = self.norm2_layers[i](x + ff_out)
        
        return x, attn_weights  # shape: (B, max_coils, D)
    
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

    
def TransformerDecoder(x, embed_dim, num_heads, ff_dim, num_layers, 
                       max_sets=6, features_per_set=100, dropout=0.1):
    
    decoder = LearnedQueryDecoder(max_sets+1, embed_dim, num_heads, ff_dim, num_layers, dropout)
    decoded, attention_weights = decoder(x)

    # Project back to original coil feature dimension
    output = layers.Dense(features_per_set)(decoded)
    return output, attention_weights