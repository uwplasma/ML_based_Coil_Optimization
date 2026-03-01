# phase1_optuna_hpo.py  (drop-in replacement)

import os
import json
import argparse
import logging
from pathlib import Path
import datetime as dt  # <-- avoid collision with from datetime import ...

import numpy as np
import tensorflow as tf
import optuna
from tensorflow.keras import layers

# Your modules
from surface_coil_loader import load_split_datasets
from transformers import TransformerEncoder, TransformerDecoder
from CoilAutoEncoder import CoilAutoencoderModel

logging.getLogger("absl").setLevel(logging.ERROR)
tf.config.optimizer.set_jit(True)


# ------------------------------
# Utils
# ------------------------------
def set_tf_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def seed_everything(seed: int):
    import random
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_log_dir(root: Path, study_name: str, trial_number: int) -> Path:
    # Fixed: use dt.datetime.now()
    tstamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{study_name}_trial{trial_number:03d}_{tstamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def best_val_metric(history, candidates=("val_masked_mse", "val_loss")):
    for k in candidates:
        if k in history.history:
            return k, float(np.min(history.history[k]))
    for k in history.history.keys():
        if k.startswith("val_"):
            return k, float(np.min(history.history[k]))
    return "loss", float(np.min(history.history.get("loss", [np.inf])))


# ------------------------------
# LR schedule (warmup + cosine)
# ------------------------------
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup to max_lr for warmup_steps, then cosine decay to min_lr over
    (total_steps - warmup_steps).
    """
    def __init__(self, max_lr, total_steps, warmup_steps=0, min_lr=0.0, name=None):
        super().__init__()
        self.max_lr = tf.convert_to_tensor(max_lr, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(tf.maximum(warmup_steps, 0), tf.float32)
        self.decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        self.min_lr = tf.convert_to_tensor(min_lr, tf.float32)
        self.name = name or "WarmupCosine"

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Warmup phase
        def warm():
            return self.max_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        # Cosine phase
        def cosine():
            t = (step - self.warmup_steps) / self.decay_steps
            t = tf.clip_by_value(t, 0.0, 1.0)
            cos = 0.5 * (1.0 + tf.cos(np.pi * t))
            return self.min_lr + (self.max_lr - self.min_lr) * cos
        return tf.where(step < self.warmup_steps, warm(), cosine())

    def get_config(self):
        return {
            "max_lr": float(self.max_lr.numpy()),
            "total_steps": int(self.total_steps.numpy()),
            "warmup_steps": int(self.warmup_steps.numpy()),
            "min_lr": float(self.min_lr.numpy()),
            "name": self.name,
        }


# ------------------------------
# Model builders
# ------------------------------
def build_coil_autoencoder(hp):
    encoder = TransformerEncoder(
        max_sets=6,
        features_per_set=100,
        name="coil",
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        ff_dim=hp["ff_dim"],
        num_sab_blocks=hp["sab_blocks"],
        dropout=hp["enc_dropout"],
    )

    decoder = TransformerDecoder(
        name="coil",
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        ff_dim=hp["ff_dim"],
        num_layers=hp["decoder_blocks"],
        max_sets=6,
        features_per_set=100,
        dropout=hp["dec_dropout"],
    )

    autoencoder = CoilAutoencoderModel(
        encoder, decoder, hp["embed_dim"], hp['kl_target'], hp["kl_warmup_steps"],
        hp['kl_cap'], hp['kl_cap_warmup'], hp['kl_gamma']
    )
    return autoencoder, encoder, decoder


def build_model(hp):
    model, encoder, decoder = build_coil_autoencoder(hp)

    # If you know your dataset size, compute total_steps precisely:
    # total_steps = steps_per_epoch * epochs
    # For now, keep your original idea but correct the math:
    # (251200 examples * 200 epochs) / batch_size
    total_steps = int((251_200 * 200) // hp["batch_size"])
    warmup_steps = int(total_steps * 0.04)

    lr_schedule = WarmupCosine(
        max_lr=hp["learning_rate"],
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=hp["learning_rate"] / 25.0,
    )

    # Lion may not exist in some tf-keras versions; fall back to AdamW.
    try:
        optimizer = tf.keras.optimizers.Lion(
            learning_rate=lr_schedule, weight_decay=hp["weight_decay"], clipnorm=1.0
        )
    except Exception:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=hp["weight_decay"], clipnorm=1.0
        )

    model.compile(optimizer=optimizer)
    return model, encoder, decoder


def train_coil_autoencoder(
    hp,
    train_ds,
    val_ds,
    epochs,
    early_stopping=True,
    es_monitor="val_loss",
    es_min_delta=0.0,
    es_patience=10,
    es_mode="min",
):
    def keep_coil_only(inputs, targets):
        return (
            {"coil_data": inputs["coil_data"], "coil_mask": inputs["coil_mask"]},
            targets["coil"],
        )

    train_ds = train_ds.map(keep_coil_only, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(keep_coil_only, num_parallel_calls=tf.data.AUTOTUNE)

    model, encoder, decoder = build_model(hp)

    callbacks = []
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_monitor,
                min_delta=es_min_delta,
                patience=es_patience,
                mode=es_mode,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )
    return history, encoder, decoder


# ------------------------------
# Hyperparameters
# ------------------------------
def suggest_hparams(trial: optuna.trial.Trial):
    hp = {}

    # First sample embed_dim, then choose heads that divide it.
    hp["embed_dim"] = trial.suggest_categorical("embed_dim", [64, 128, 256])
    possible_heads = [h for h in (4, 8, 12) if hp["embed_dim"] % h == 0]
    # Always non-empty given the choices above; but be safe:
    if not possible_heads:
        possible_heads = [4]
    hp["num_heads"] = trial.suggest_categorical("num_heads", possible_heads)

    hp["ff_dim"] = trial.suggest_categorical("ff_dim", [256, 384, 512, 768, 1024])

    # Use distinct names (no collisions)
    hp["sab_blocks"] = trial.suggest_int("sab_blocks", 2, 6)
    hp["decoder_blocks"] = trial.suggest_int("decoder_blocks", 2, 6)

    hp["enc_dropout"] = trial.suggest_float("enc_dropout", 0.0, 0.3)
    hp["dec_dropout"] = trial.suggest_float("dec_dropout", 0.0, 0.3)

    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    hp["kl_target"] = trial.suggest_float("kl_target", 5e-3, 0.1, log=True)
    hp["kl_warmup_steps"] = trial.suggest_int("kl_warmup_steps", 2000, 20000, step=2000)

    hp["kl_cap_per_dim"] = trial.suggest_float("kl_cap", 0.05, 0.3)
    hp["kl_cap_warmup"] = trial.suggest_int("kl_cap_warmup", 10000, 30000, step=2000)
    hp["kl_gamma"] = trial.suggest_float("kl_gamma", 1e-5, 1e-2, log=True)

    hp['kl_cap'] = hp["kl_cap_per_dim"] * hp["embed_dim"]

    return hp


# ------------------------------
# Objective
# ------------------------------
def make_objective(args):
    tfrecord_dir = args.tfrecord_dir
    log_root = Path(args.log_root)
    epochs = args.epochs
    seed = args.seed

    def objective(trial: optuna.trial.Trial):
        tf.keras.backend.clear_session()
        seed_everything(seed + trial.number)
        set_tf_memory_growth()

        hp = suggest_hparams(trial)
        trial_log_dir = make_log_dir(log_root, args.study_name, trial.number)

        train_ds, val_ds, _ = load_split_datasets(
            tfrecord_dir=tfrecord_dir, batch_size=hp["batch_size"]
        )

        history, encoder, decoder = train_coil_autoencoder(
            hp=hp, train_ds=train_ds, val_ds=val_ds, epochs=epochs
        )

        monitor, score = best_val_metric(
            history, candidates=(args.es_monitor, "val_masked_mse", "val_loss")
        )

        with open(trial_log_dir / "hparams.json", "w") as f:
            json.dump(hp, f, indent=2)
        with open(trial_log_dir / "score.json", "w") as f:
            json.dump({"monitor": monitor, "score": score}, f, indent=2)

        trial.set_user_attr("monitor", monitor)
        trial.set_user_attr("log_dir", str(trial_log_dir))
        return score

    return objective


# ------------------------------
# Main
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord_dir", type=str, required=True,
                   help="Directory with TFRecords. Use '~/canon_curr_surface_coil_tfrecords' (note the slash).")
    p.add_argument("--log_root", type=str, default="coil_logs/optuna_runs",
                   help="Where to write trial logs.")
    p.add_argument("--study_name", type=str, default="coil_ae_hpo")
    p.add_argument("--storage", type=str, default=None,
                   help="e.g. sqlite:///coil_hpo.db (optional)")
    p.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=73)
    # For best_val_metric candidate
    p.add_argument("--es_monitor", type=str, default="val_masked_mse")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,           # None => in-memory; or "sqlite:///coil_hpo.db"
        direction=args.direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            ),
            patience=2
        ),
    )

    objective = make_objective(args)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\n== Best trial ==")
    bt = study.best_trial
    print("  value:", bt.value)
    print("  params:")
    for k, v in bt.params.items():
        print(f"    {k}: {v}")
    print("  user_attrs:", bt.user_attrs)

    out = {
        "best_value": bt.value,
        "best_params": bt.params,
        "best_user_attrs": bt.user_attrs,
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
    }
    with open(Path(args.log_root) / f"{args.study_name}_best.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
