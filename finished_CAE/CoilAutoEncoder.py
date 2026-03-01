import tensorflow as tf
import keras
from keras.saving import serialize_keras_object, deserialize_keras_object
from transformers import DiagonalGaussian

def masked_mse(pred, target, mask):
    e2 = tf.square(pred - target)
    num = tf.reduce_sum(mask * e2)
    den = tf.reduce_sum(mask) + 1e-8
    return num / den

def kl_diag_gaussian(mu, logvar):
    # KL( N(mu, diag(exp(logvar))) || N(0, I) )
    # = 0.5 * sum( mu^2 + exp(logvar) - 1 - logvar )
    delta = 0.2
    kl_j = 0.5*(mu**2 + tf.exp(logvar) - logvar - 1.)   # [B,d]
    kl_j = tf.nn.relu(kl_j - delta) + delta             # clamp from below
    kl_batch = tf.reduce_mean(tf.reduce_sum(kl_j, -1))
    return kl_batch


@keras.utils.register_keras_serializable(package="my_layers")
class CoilAutoencoderModel(tf.keras.Model):
    def __init__(self, encoder, decoder, latent_dim, kl_target, kl_warmup_steps=10_000, kl_cap=3, kl_cap_warmup=15000, kl_gamma=1e-3,
                 sample_latent_train=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.latent_head = DiagonalGaussian(latent_dim, name="latent_head")
        self.kl_target = float(kl_target)
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.sample_latent_train = bool(sample_latent_train)
        self._global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")

        self.kl_cap_final   = tf.constant(kl_cap, tf.float32)   # nats target (tune)
        self.kl_cap_warmup  = tf.constant(kl_cap_warmup, tf.int64)   # steps to reach C_final
        self.kl_gamma       = tf.constant(kl_gamma, tf.float32)  # small penalty weight

    def call(self, inputs, training=False):
        latent = self.encoder(inputs, training=training)
        decoded = self.decoder(latent, training=training)
        return decoded
    
    def _kl_weight(self):
        # linear 0 → kl_target over warmup steps (clamped)
        t = tf.cast(tf.minimum(self._global_step, self.kl_warmup_steps), tf.float32)
        w = (t / float(self.kl_warmup_steps)) * self.kl_target
        return w

    def _kl_capacity(self):
        t = tf.cast(tf.minimum(self._global_step, self.kl_cap_warmup), tf.float32)
        return (t / tf.cast(self.kl_cap_warmup, tf.float32)) * self.kl_cap_final

    def train_step(self, data):
        
        inputs, targets = data
        coil_mask = tf.cast(inputs["coil_mask"][..., tf.newaxis], tf.float32)
        # targets = targets['coil']

        with tf.GradientTape() as tape:
            latent = self.encoder(inputs, training=True)
            z, mu, logvar = self.latent_head(latent, training=True, sample=self.sample_latent_train)
            predictions = self.decoder(z, training=True)

            # 🔹 Masked MSE
            squared_error = tf.square(predictions - targets)
            # coil_loss = tf.reduce_mean(squared_error * coil_mask)
            coil_loss = masked_mse(predictions, targets, coil_mask)
            # scaler_loss = tf.reduce_mean(squared_error[:, -1, 0])  # Only first feature of last token

            kl = kl_diag_gaussian(mu, logvar)             # (B,)
            # kl = tf.reduce_mean(kl_per_example)                        # scalar
            kl_w = self._kl_weight()                                   # scalar
            # kl_term   = kl_w * kl
            # kl_capped = tf.minimum(kl_term, 0.3 * tf.stop_gradient(coil_loss))
            # total_loss = coil_loss + kl_capped

            C_t = self._kl_capacity()
            capacity_penalty = self.kl_gamma * tf.square(kl - C_t)
            total_loss = coil_loss + kl_w * kl + capacity_penalty

            # total_loss = coil_loss #+ scaler_loss * 0.25

            # 🔸 Optional: Unmasked MSE and MAE for reference
            unmasked_mse = tf.reduce_mean(squared_error)
            mae = tf.reduce_mean(tf.abs(predictions - targets))

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._global_step.assign_add(1)
        # self.metric_kl.update_state(kl)
        # self.metric_klw.update_state(kl_w)

        return {
            "loss": total_loss,
            'masked_recon': coil_loss,
            'KL': kl,
            'KL_weight': kl_w,
            'capacity_penalty': capacity_penalty,
            'capacity_weight': C_t,
            "unmasked_mse": unmasked_mse,
            "mae": mae
            }
    
    def test_step(self, data):
        inputs, targets = data
        coil_mask = tf.cast(inputs["coil_mask"][..., tf.newaxis], tf.float32)
        # targets = targets['coil']
        latent = self.encoder(inputs, training=False)
        z, mu, logvar = self.latent_head(latent, training=False, sample=False)
        predictions = self.decoder(z, training=False)

        # 🔹 Masked MSE
        squared_error = tf.square(predictions - targets)
        # coil_loss = tf.reduce_mean(squared_error * coil_mask)
        coil_loss = masked_mse(predictions, targets, coil_mask)
        # scaler_loss = tf.reduce_mean(squared_error[:, -1, 0])  # Only first feature of last token

        kl = kl_diag_gaussian(mu, logvar)
        kl_w = self._kl_weight()  # you can also fix to kl_target for val if you prefer
        # kl_term   = kl_w * kl
        # kl_capped = tf.minimum(kl_term, 0.3 * tf.stop_gradient(coil_loss))
        # total_loss = coil_loss + kl_capped

        C_t = self._kl_capacity()
        capacity_penalty = self.kl_gamma * tf.square(kl - C_t)
        total_loss = coil_loss + kl_w * kl + capacity_penalty

        # total_loss = coil_loss #+ scaler_loss * 0.1

        # 🔸 Optional: Unmasked MSE and MAE for reference
        unmasked_mse = tf.reduce_mean(squared_error)
        mae = tf.reduce_mean(tf.abs(predictions - targets))

        return {
            "loss": total_loss,
            'masked_recon': coil_loss,
            'KL': kl,
            'KL_weight': kl_w,
            'capacity_penalty': capacity_penalty,
            'capacity_weight': C_t,
            "unmasked_mse": unmasked_mse,
            "mae": mae
            }
    
    def get_config(self):
        # Avoid super().get_config() to dodge MRO/namespace issues.
        return {
            "name": self.name,
            "encoder": serialize_keras_object(self.encoder),
            "decoder": serialize_keras_object(self.decoder),
        }

    @classmethod
    def from_config(cls, config):
        enc = deserialize_keras_object(config.pop("encoder"))
        dec = deserialize_keras_object(config.pop("decoder"))
        # pass along name if present (Keras will set it too)
        name = config.pop("name", None)
        return cls(encoder=enc, decoder=dec, name=name, **config)