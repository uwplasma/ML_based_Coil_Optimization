
# phase1_surface_optuna_hpo.py
# Optuna HPO for Surface Encoder (VAE + Perceiver encoder).
# Template derived from phase1_optuna_hpo.py.

from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
import datetime as dt

import numpy as np
import tensorflow as tf
import optuna

from SurfaceEncoder_overhauled_perceiver import (
    build_surface_perceiver_encoder,
    build_surface_decoder,
    SurfaceVAEPerceiverModel,
)

logging.getLogger("absl").setLevel(logging.ERROR)
tf.config.optimizer.set_jit(True)


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
    tstamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{study_name}_trial{trial_number:03d}_{tstamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def best_val_metric(history, candidates=("val_masked_recon", "val_loss")):
    for k in candidates:
        if k in history.history:
            return k, float(np.min(history.history[k]))
    for k in history.history.keys():
        if k.startswith("val_"):
            return k, float(np.min(history.history[k]))
    return "loss", float(np.min(history.history.get("loss", [np.inf])))


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
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

        def warm():
            return self.max_lr * (step / tf.maximum(self.warmup_steps, 1.0))

        def cosine():
            t = (step - self.warmup_steps) / self.decay_steps
            t = tf.clip_by_value(t, 0.0, 1.0)
            cos = 0.5 * (1.0 + tf.cos(np.pi * t))
            return self.min_lr + (self.max_lr - self.min_lr) * cos

        return tf.where(step < self.warmup_steps, warm(), cosine())


def parse_latent_tfrecord(example_proto, *, max_sets, features_per_set, coil_latent_dim):
    T = max_sets + 1
    feat = {
        "surface_data": tf.io.FixedLenFeature([T * features_per_set], tf.float32),
        "surface_mask": tf.io.FixedLenFeature([max_sets], tf.int64),
        "coil_latent": tf.io.FixedLenFeature([coil_latent_dim], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feat)

    surface_data = tf.reshape(parsed["surface_data"], [T, features_per_set])
    surface_mask = tf.cast(parsed["surface_mask"], tf.float32)
    coil_latent = parsed["coil_latent"]

    inputs = {"surface_data": surface_data, "surface_mask": surface_mask}
    targets = {"surface": surface_data, "coil_latent": coil_latent}
    return inputs, targets


def load_split_datasets(
    tfrecord_dir: str | Path,
    *,
    batch_size: int,
    max_sets: int,
    features_per_set: int,
    coil_latent_dim: int,
    train_frac=0.8,
    val_frac=0.1,
    shuffle=True,
    shuffle_buffer_size=50_000,
    tfrecord_buffer_bytes=8 << 20,
    repeat=False,
    num_parallel_calls=tf.data.AUTOTUNE,
    max_records=None,
    seed=73,
    num_examples_hint: int | None = None,
):
    tfrecord_dir = Path(tfrecord_dir)
    files = sorted(tf.io.gfile.glob(str(tfrecord_dir / "*.tfrecord")))
    if not files:
        raise FileNotFoundError(f"No TFRecord files in {tfrecord_dir}")

    base = tf.data.TFRecordDataset(
        files,
        buffer_size=tfrecord_buffer_bytes,
        num_parallel_reads=num_parallel_calls,
    )
    if max_records is not None and max_records > 0:
        base = base.take(int(max_records))

    parsed = base.map(
        lambda raw: parse_latent_tfrecord(
            raw, max_sets=max_sets, features_per_set=features_per_set, coil_latent_dim=coil_latent_dim
        ),
        num_parallel_calls=num_parallel_calls,
    )

    if num_examples_hint is None:
        total = int(parsed.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy())
        if total <= 0:
            raise ValueError("Dataset is empty after parsing.")
    else:
        total = int(num_examples_hint)

    n_train = int(total * train_frac)
    n_val = int(total * val_frac)

    # Recreate pipeline for slicing deterministically
    base2 = tf.data.TFRecordDataset(
        files,
        buffer_size=tfrecord_buffer_bytes,
        num_parallel_reads=num_parallel_calls,
    )
    if max_records is not None and max_records > 0:
        base2 = base2.take(int(max_records))

    parsed2 = base2.map(
        lambda raw: parse_latent_tfrecord(
            raw, max_sets=max_sets, features_per_set=features_per_set, coil_latent_dim=coil_latent_dim
        ),
        num_parallel_calls=num_parallel_calls,
    )

    train_raw = parsed2.take(n_train)
    val_raw = parsed2.skip(n_train).take(n_val)
    test_raw = parsed2.skip(n_train + n_val)

    def finalize(ds, do_shuffle):
        if shuffle and do_shuffle:
            ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True, seed=seed)
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    return finalize(train_raw, True), finalize(val_raw, False), finalize(test_raw, False)


def build_surface_autoencoder(hp, *, max_sets, features_per_set, coil_latent_dim, value_col):
    T = max_sets + 1
    encoder = build_surface_perceiver_encoder(
        max_sets=T,
        features_per_set=features_per_set,
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        ff_dim=hp["ff_dim"],
        num_latents=hp["num_latents"],
        num_blocks=hp["enc_blocks"],
        dropout=hp["enc_dropout"],
        name="surface",
    )
    decoder = build_surface_decoder(
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        ff_dim=hp["ff_dim"],
        num_layers=hp["dec_blocks"],
        max_sets=T,
        features_per_set=features_per_set,
        dropout=hp["dec_dropout"],
        name="surface",
    )

    model = SurfaceVAEPerceiverModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=hp["embed_dim"],
        coil_latent_dim=coil_latent_dim,
        value_col=value_col,
        meta_recon_weight=hp["meta_recon_weight"],
        align_weight=hp["align_weight"],
        align_type=hp["align_type"],
        kl_target=hp["kl_target"],
        kl_warmup_steps=hp["kl_warmup_steps"],
        kl_cap=hp["kl_cap"],
        kl_cap_warmup=hp["kl_cap_warmup"],
        kl_gamma=hp["kl_gamma"],
        sample_latent_train=True,
    )
    return model


def build_model(hp, *, epochs, num_examples_for_lr, max_sets, features_per_set, coil_latent_dim, value_col):
    model = build_surface_autoencoder(
        hp,
        max_sets=max_sets,
        features_per_set=features_per_set,
        coil_latent_dim=coil_latent_dim,
        value_col=value_col,
    )
    total_steps = int((num_examples_for_lr * epochs) // hp["batch_size"])
    total_steps = max(total_steps, 1)
    warmup_steps = int(total_steps * hp["warmup_frac"])

    lr_schedule = WarmupCosine(
        max_lr=hp["learning_rate"],
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=hp["learning_rate"] / 25.0,
    )

    try:
        optimizer = tf.keras.optimizers.Lion(
            learning_rate=lr_schedule, weight_decay=hp["weight_decay"], clipnorm=1.0
        )
    except Exception:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=hp["weight_decay"], clipnorm=1.0
        )

    model.compile(optimizer=optimizer)
    return model


def train_surface_autoencoder(
    hp,
    train_ds,
    val_ds,
    *,
    epochs,
    num_examples_for_lr,
    max_sets,
    features_per_set,
    coil_latent_dim,
    value_col,
    early_stopping=True,
    es_monitor="val_masked_recon",
    es_min_delta=0.0,
    es_patience=10,
    es_mode="min",
):
    model = build_model(
        hp,
        epochs=epochs,
        num_examples_for_lr=num_examples_for_lr,
        max_sets=max_sets,
        features_per_set=features_per_set,
        coil_latent_dim=coil_latent_dim,
        value_col=value_col,
    )

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

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1) #verbose=0 means no metric prints
    return history


def suggest_hparams(trial: optuna.trial.Trial):
    hp = {}
    hp["embed_dim"] = 64 #trial.suggest_categorical("embed_dim", [64, 128, 256])
    possible_heads = [h for h in (4, 8, 12) if hp["embed_dim"] % h == 0]
    hp["num_heads"] = trial.suggest_categorical("num_heads", possible_heads or [4])

    hp["ff_dim"] = trial.suggest_categorical("ff_dim", [256, 384, 512, 768, 1024])
    hp["num_latents"] = trial.suggest_categorical("num_latents", [16, 32, 64, 96])
    hp["enc_blocks"] = trial.suggest_int("enc_blocks", 2, 6)
    hp["dec_blocks"] = trial.suggest_int("dec_blocks", 2, 6)
    hp["enc_dropout"] = trial.suggest_float("enc_dropout", 0.0, 0.3)
    hp["dec_dropout"] = trial.suggest_float("dec_dropout", 0.0, 0.3)

    hp["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hp["warmup_frac"] = trial.suggest_float("warmup_frac", 0.02, 0.08)

    hp["kl_target"] = trial.suggest_float("kl_target", 5e-3, 0.1, log=True)
    hp["kl_warmup_steps"] = trial.suggest_int("kl_warmup_steps", 2000, 20000, step=2000)
    hp["kl_cap_per_dim"] = trial.suggest_float("kl_cap_per_dim", 0.05, 0.3)
    hp["kl_cap_warmup"] = trial.suggest_int("kl_cap_warmup", 10_000, 30_000, step=2000)
    hp["kl_gamma"] = trial.suggest_float("kl_gamma", 1e-5, 1e-2, log=True)
    hp["kl_cap"] = hp["kl_cap_per_dim"] * hp["embed_dim"]

    hp["align_weight"] = trial.suggest_float("align_weight", 0.0, 1.0)
    hp["align_type"] = trial.suggest_categorical("align_type", ["cosine", "mse"])
    hp["meta_recon_weight"] = trial.suggest_float("meta_recon_weight", 0.0, 0.05)

    return hp


def make_objective(args):
    tfrecord_dir = args.tfrecord_dir
    log_root = Path(args.log_root)
    epochs = args.epochs
    seed = args.seed

    max_sets = args.max_sets
    features_per_set = args.features_per_set
    coil_latent_dim = args.coil_latent_dim
    value_col = args.value_col

    num_examples_hint = args.num_examples if args.num_examples > 0 else None
    num_examples_for_lr = args.num_examples if args.num_examples > 0 else args.default_num_examples_for_lr

    def objective(trial: optuna.trial.Trial):
        tf.keras.backend.clear_session()
        seed_everything(seed + trial.number)
        set_tf_memory_growth()

        hp = suggest_hparams(trial)
        trial_log_dir = make_log_dir(log_root, args.study_name, trial.number)

        train_ds, val_ds, _ = load_split_datasets(
            tfrecord_dir=tfrecord_dir,
            batch_size=hp["batch_size"],
            max_sets=max_sets,
            features_per_set=features_per_set,
            coil_latent_dim=coil_latent_dim,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            shuffle=True,
            shuffle_buffer_size=args.shuffle_buffer,
            repeat=False,
            seed=seed,
            max_records=args.max_records if args.max_records > 0 else None,
            num_examples_hint=num_examples_hint,
        )

        history = train_surface_autoencoder(
            hp=hp,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=epochs,
            num_examples_for_lr=num_examples_for_lr,
            max_sets=max_sets,
            features_per_set=features_per_set,
            coil_latent_dim=coil_latent_dim,
            value_col=value_col,
            early_stopping=True,
            es_monitor=args.es_monitor,
            es_patience=args.es_patience,
        )

        monitor, score = best_val_metric(history, candidates=(args.es_monitor, "val_masked_recon", "val_loss"))

        with open(trial_log_dir / "hparams.json", "w") as f:
            json.dump(hp, f, indent=2)
        with open(trial_log_dir / "score.json", "w") as f:
            json.dump({"monitor": monitor, "score": score}, f, indent=2)

        trial.set_user_attr("monitor", monitor)
        trial.set_user_attr("log_dir", str(trial_log_dir))
        return score

    return objective


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord_dir", type=str, required=True,
                   help="Directory with latent-augmented TFRecords (surface_data, surface_mask, coil_latent).")
    p.add_argument("--log_root", type=str, default="surface_logs/optuna_runs")
    p.add_argument("--study_name", type=str, default="surface_ae_hpo")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=73)

    p.add_argument("--max_sets", type=int, default=441)
    p.add_argument("--features_per_set", type=int, default=4)
    p.add_argument("--coil_latent_dim", type=int, default=64)
    p.add_argument("--value_col", type=int, default=3)

    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--shuffle_buffer", type=int, default=50_000)
    p.add_argument("--max_records", type=int, default=0)

    p.add_argument("--num_examples", type=int, default=0,
                   help="If >0, used as dataset size hint for splitting AND LR schedule steps.")
    p.add_argument("--default_num_examples_for_lr", type=int, default=251_200)

    p.add_argument("--es_monitor", type=str, default="val_masked_recon")
    p.add_argument("--es_patience", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1),
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
