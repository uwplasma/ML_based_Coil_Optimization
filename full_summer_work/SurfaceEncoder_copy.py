import tensorflow as tf

class SurfaceEncoderModel(tf.keras.Model):
    def __init__(self, encoder, decoder, lambda_coil=0.5):
        """
        encoder: surface encoder network
        decoder: surface decoder network
        lambda_coil: weight for coil latent loss in total loss
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_coil = lambda_coil

    def call(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def train_step(self, data):
        # Unpack data
        inputs, targets = data
        
        # Surface inputs
        surface_mask = tf.cast(inputs["surface_mask"][..., tf.newaxis], tf.float32)
        
        # Targets
        surface_target = targets["surface"]           # (B, N+1, D)
        coil_latent_target = targets["coil_latent"]   # (B, latent_dim)

        with tf.GradientTape() as tape:
            # Encode surface
            encoded_latent, tokens = self.encoder({"surface_data": inputs["surface_data"], "surface_mask": inputs["surface_mask"]}, 
                                          training=True)
            
            # Decode to reconstruct surface
            surface_recon = self.decoder(encoded_latent, training=True)

            # --- 1️⃣ Surface reconstruction loss ---
            surf_sq_err = tf.square(surface_recon - surface_target)
            surface_loss = tf.reduce_mean(surf_sq_err[:, :-1, :] * surface_mask)
            scaler_loss = tf.reduce_mean(surf_sq_err[:, -1, 0])  # first feature of last token
            recon_loss = surface_loss + scaler_loss

            # --- 2️⃣ Coil latent matching loss ---
            coil_latent_loss = tf.reduce_mean(tf.square(encoded_latent - coil_latent_target))

            # --- 3️⃣ Total loss (weighted sum) ---
            total_loss = recon_loss + self.lambda_coil * coil_latent_loss

            # Optional metrics
            unmasked_mse = tf.reduce_mean(surf_sq_err[:, :-1, :])
            mae = tf.reduce_mean(tf.abs(surface_recon - surface_target)[:, :-1, :])

        # Backpropagation
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "coil_latent_loss": coil_latent_loss,
            "scaler_loss": scaler_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae
        }

    def test_step(self, data):
        inputs, targets = data
        surface_mask = tf.cast(inputs["surface_mask"][..., tf.newaxis], tf.float32)
        
        surface_target = targets["surface"]
        coil_latent_target = targets["coil_latent"]

        encoded_latent, tokens = self.encoder({"surface_data": inputs["surface_data"], "surface_mask": inputs["surface_mask"]}, 
                                      training=False)
        surface_recon = self.decoder(encoded_latent, training=False)

        # Surface loss
        surf_sq_err = tf.square(surface_recon - surface_target)
        surface_loss = tf.reduce_mean(surf_sq_err[:, :-1, :] * surface_mask)
        scaler_loss = tf.reduce_mean(surf_sq_err[:, -1, 0])
        recon_loss = surface_loss + scaler_loss

        # Coil latent loss
        coil_latent_loss = tf.reduce_mean(tf.square(encoded_latent - coil_latent_target))

        total_loss = recon_loss + self.lambda_coil * coil_latent_loss

        unmasked_mse = tf.reduce_mean(surf_sq_err[:, :-1, :])
        mae = tf.reduce_mean(tf.abs(surface_recon - surface_target)[:, :-1, :])

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "coil_latent_loss": coil_latent_loss,
            "scaler_loss": scaler_loss,
            "unmasked_mse": unmasked_mse,
            "mae": mae
        }
