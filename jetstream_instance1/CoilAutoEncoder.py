import tensorflow as tf

def masked_mse(pred, target, mask):
    e2 = tf.square(pred - target)
    num = tf.reduce_sum(mask * e2)
    den = tf.reduce_sum(mask) + 1e-8
    return num / den

class CoilAutoencoderModel(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        latent, tokens = self.encoder(inputs, training=training)
        decoded = self.decoder([tokens, latent], training=training)
        return decoded

    def train_step(self, data):
        
        inputs, targets = data
        coil_mask = tf.cast(inputs["coil_mask"][..., tf.newaxis], tf.float32)
        targets = targets['coil']

        with tf.GradientTape() as tape:
            latent, tokens = self.encoder(inputs, training=True)
            predictions = self.decoder([tokens, latent], training=True)

            # 🔹 Masked MSE
            squared_error = tf.square(predictions - targets)
            coil_loss = tf.reduce_mean(squared_error * coil_mask)
            # coil_loss = masked_mse(predictions[:, :-1, :], targets[:, :-1, :], coil_mask)
            # scaler_loss = tf.reduce_mean(squared_error[:, -1, 0])  # Only first feature of last token

            total_loss = coil_loss #+ scaler_loss * 0.25

            # 🔸 Optional: Unmasked MSE and MAE for reference
            unmasked_mse = tf.reduce_mean(squared_error)
            mae = tf.reduce_mean(tf.abs(predictions - targets))

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": total_loss,
            # "scaler_loss": scaler_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae
            }
    
    def test_step(self, data):
        inputs, targets = data
        coil_mask = tf.cast(inputs["coil_mask"][..., tf.newaxis], tf.float32)
        targets = targets['coil']
        latent, tokens = self.encoder(inputs, training=False)
        predictions = self.decoder([tokens, latent], training=False)

        # 🔹 Masked MSE
        squared_error = tf.square(predictions - targets)
        coil_loss = tf.reduce_mean(squared_error * coil_mask)
        # coil_loss = masked_mse(predictions[:, :-1, :], targets[:, :-1, :], coil_mask)
        # scaler_loss = tf.reduce_mean(squared_error[:, -1, 0])  # Only first feature of last token

        total_loss = coil_loss #+ scaler_loss * 0.1

        # 🔸 Optional: Unmasked MSE and MAE for reference
        unmasked_mse = tf.reduce_mean(squared_error)
        mae = tf.reduce_mean(tf.abs(predictions - targets))

        return {
            "loss": total_loss,
            #"scaler_loss": scaler_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae
            }