import tensorflow as tf

class CoilAutoencoderModel(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        latent, tokens = self.encoder(inputs, training=training)
        decoded = self.decoder(latent, training=training)
        return decoded

    def train_step(self, data):
        inputs, targets = data
        coil_mask = tf.cast(inputs["coil_mask"][..., tf.newaxis], tf.float32)
        targets = targets['coil']

        with tf.GradientTape() as tape:
            latent, tokens = self.encoder(inputs, training=True)
            predictions = self.decoder(latent, training=True)

            # 🔹 Masked MSE
            squared_error = tf.square(predictions - targets)
            coil_loss = tf.reduce_mean(squared_error[:, :-1, :] * coil_mask)
            scaler_loss = tf.reduce_mean(squared_error[:, -1, 0])  # Only first feature of last token

            total_loss = coil_loss + scaler_loss

            # 🔸 Optional: Unmasked MSE and MAE for reference
            unmasked_mse = tf.reduce_mean(squared_error[:, :-1, :])
            mae = tf.reduce_mean(tf.abs(predictions - targets)[:, :-1, :])

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": total_loss,
            "coil_loss": coil_loss,
            "scaler_loss": scaler_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae
            }
    
    def test_step(self, data):
        inputs, targets = data
        coil_mask = tf.cast(inputs["coil_mask"][..., tf.newaxis], tf.float32)
        targets = targets['coil']
        latent, tokens = self.encoder(inputs, training=False)
        predictions = self.decoder(latent, training=False)

        # 🔹 Masked MSE
        squared_error = tf.square(predictions - targets)
        coil_loss = tf.reduce_mean(squared_error[:, :-1, :] * coil_mask)
        scaler_loss = tf.reduce_mean(squared_error[:, -1, 0])  # Only first feature of last token

        total_loss = coil_loss + scaler_loss

        # 🔸 Optional: Unmasked MSE and MAE for reference
        unmasked_mse = tf.reduce_mean(squared_error[:, :-1, :])
        mae = tf.reduce_mean(tf.abs(predictions - targets)[:, :-1, :])

        return {
            "loss": total_loss,
            "coil_loss": coil_loss,
            "scaler_loss": scaler_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae
            }