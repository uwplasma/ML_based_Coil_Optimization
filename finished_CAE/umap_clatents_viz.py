import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib
from matplotlib.colors import Normalize, LogNorm


# custom layers so load_model works
from transformers import SelfAttentionBlock, PoolingByMultiheadAttention, LearnedQueryDecoder
from surface_coil_loader import load_full_dataset

CUSTOM_OBJECTS = {
    "SelfAttentionBlock": SelfAttentionBlock,
    "PoolingByMultiheadAttention": PoolingByMultiheadAttention,
    "LearnedQueryDecoder": LearnedQueryDecoder,
}

# ---------------- Repro ----------------
def fix_seeds(seed):
    if seed is None:
        return
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        import umap  # type: ignore
        umap.random.seed(seed)
    except Exception:
        pass

# ------------- Models I/O -------------
def load_encoder_decoder(models_dir: Path):
    enc = tf.keras.models.load_model(
        str(models_dir / "encoder.keras"),
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,
        compile=False,
    )
    dec_path = models_dir / "decoder.keras"
    dec = None
    if dec_path.exists():
        dec = tf.keras.models.load_model(
            str(dec_path),
            custom_objects=CUSTOM_OBJECTS,
            safe_mode=False,
            compile=False,
        )
    return enc, dec

def encoder_to_latent(encoder, batch_inputs):
    enc_out = encoder(batch_inputs, training=False)
    if isinstance(enc_out, (tuple, list)) and len(enc_out) == 2:
        z, tokens = enc_out
    else:
        z, tokens = enc_out, None
    z = tf.convert_to_tensor(z)
    if z.shape.rank == 3 and z.shape[1] == 1:
        z = tf.squeeze(z, axis=1)  # (B,D)
    return z.numpy(), tokens

def forward_predict(decoder, tokens, z):
    if decoder is None:
        return None
    z = tf.convert_to_tensor(z)

    def _coerce_z(z, expected_shape):
        dims = len(expected_shape)
        if dims == 3:
            if z.shape.rank == 2:
                z = tf.expand_dims(z, axis=1)  # (B,1,D)
            elif z.shape.rank == 3 and z.shape[1] != 1:
                z = z[:, :1, :]
        elif dims == 2:
            if z.shape.rank == 3 and z.shape[1] == 1:
                z = tf.squeeze(z, axis=1)      # (B,D)
        return z

    if isinstance(decoder.inputs, (list, tuple)) and len(decoder.inputs) == 2:
        z = _coerce_z(z, decoder.inputs[1].shape)
        return decoder([tokens, z], training=False)

    z = _coerce_z(z, decoder.inputs[0].shape)
    return decoder(z, training=False)

# --------------- Metrics ---------------
def per_sample_norm_mse(pred, target, mask):
    per_coil_mse = tf.reduce_mean(tf.square(pred - target), axis=-1)  # (B,K)
    mask = tf.cast(mask, tf.float32)
    num = tf.reduce_sum(per_coil_mse * mask, axis=-1)
    den = tf.reduce_sum(mask, axis=-1) + 1e-8
    return (num / den).numpy()

# --------------- UMAP full --------------
def fit_or_load_umap(Z, seed, n_neighbors, min_dist, metric, model_in: Path | None, model_out: Path):
    try:
        from umap.umap_ import UMAP  # robust across versions
    except Exception as e:
        raise RuntimeError("umap-learn is required. pip install umap-learn") from e

    if model_in and model_in.exists():
        um = joblib.load(model_in.as_posix())
        emb = um.transform(Z)  # transform ALL points
        return um, emb, "umap_loaded"

    um = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,  # set to None for parallelism (non-deterministic)
        n_components=2,
    )
    emb = um.fit_transform(Z)  # fit + embed ALL points

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(um, model_out.as_posix())
    return um, emb, "umap_full"

# ------------- Plot helpers -------------
def robust_norm(values, mode="pct", p_low=1.0, p_high=99.0, log=False, eps=1e-12):
    """Return a matplotlib norm for values with robust bounds."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Normalize()  # fallback

    if log or mode == "log":
        # Use percentiles to set bounds, floor vmin at eps
        vmin = max(np.percentile(v, p_low), eps)
        vmax = max(np.percentile(v, p_high), vmin * 10.0)  # avoid identical bounds
        return LogNorm(vmin=vmin, vmax=vmax)

    # default: linear with percentile clipping
    vmin = np.percentile(v, p_low)
    vmax = np.percentile(v, p_high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = v.min(), v.max()
        if vmin == vmax:
            vmax = vmin + 1e-9
    return Normalize(vmin=vmin, vmax=vmax)

def professional_axes(ax, title=None):
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Embedding 1", fontsize=11)
    ax.set_ylabel("Embedding 2", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def discrete_palette(unique_vals):
    cmap = plt.get_cmap("tab20")
    return {v: cmap(i % cmap.N) for i, v in enumerate(sorted(unique_vals))}

def plot_fig_A(emb, coil_counts, method, out_base):
    figA_png = Path(out_base).with_name(Path(out_base).stem + "_FigA.png")
    figA_pdf = figA_png.with_suffix(".pdf")

    N = emb.shape[0]
    idx = np.arange(N)

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    uvals = np.unique(coil_counts)
    cmap = discrete_palette(uvals)
    colors = np.array([cmap[c] for c in coil_counts[idx]])
    ax.scatter(emb[idx, 0], emb[idx, 1], c=colors, s=6, alpha=0.75, linewidths=0)

    legend_elems = [Line2D([0], [0], marker='o', linestyle='',
                           markerfacecolor=cmap[v], markeredgecolor='none',
                           markersize=6, label=f'{int(v)} coils') for v in uvals]
    ax.legend(handles=legend_elems, title="# of Coils", fontsize=9, title_fontsize=10,
              loc="best", frameon=True, framealpha=0.9)

    professional_axes(ax, f"CAE Latent Space — {method.upper()} (N={N})\nColor = number of valid coils")
    plt.tight_layout()
    fig.savefig(figA_png, dpi=300); fig.savefig(figA_pdf)
    plt.close(fig)
    print(f"Saved {figA_png} and {figA_pdf}")

def plot_fig_B(emb, recon_mse, method, out_base,
               cbar_mode="pct", cbar_p_low=1.0, cbar_p_high=99.0, cbar_log=False):
    figB_png = Path(out_base).with_name(Path(out_base).stem + "_FigB.png")
    figB_pdf = figB_png.with_suffix(".pdf")

    N = emb.shape[0]
    idx = np.arange(N)

    # Build a robust color normalization
    norm = robust_norm(recon_mse, mode=cbar_mode, p_low=cbar_p_low, p_high=cbar_p_high, log=cbar_log)

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    sc = ax.scatter(emb[idx, 0], emb[idx, 1],
                    c=recon_mse[idx], s=6, alpha=0.75, linewidths=0,
                    cmap="viridis", norm=norm)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized reconstruction MSE", fontsize=10)

    # Optional: print chosen bounds for quick sanity
    if isinstance(norm, LogNorm):
        print(f"[FigB] Log scale color range vmin={norm.vmin:.3e}, vmax={norm.vmax:.3e}")
        subtitle = "Color = normalized recon MSE (log scale)"
    else:
        print(f"[FigB] Linear color range vmin={norm.vmin:.3e}, vmax={norm.vmax:.3e}")
        subtitle = "Color = normalized recon MSE (percentile-clipped)"

    professional_axes(ax, f"CAE Latent Space — {method.upper()} (N={N})\n{subtitle}")
    plt.tight_layout()
    fig.savefig(figB_png, dpi=300); fig.savefig(figB_pdf)
    plt.close(fig)
    print(f"Saved {figB_png} and {figB_pdf}")


def plot_fig_C_facets_by_nfp(emb, coil_counts, nfp, method, out_base):
    if nfp is None:
        print("Figure C skipped: no NFP available in dataset.")
        return
    figC_png = Path(out_base).with_name(Path(out_base).stem + "_FigC.png")
    figC_pdf = figC_png.with_suffix(".pdf")

    u_nfp = sorted(np.unique(nfp))
    k = len(u_nfp)
    ncols = min(3, k)
    nrows = int(np.ceil(k / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5*ncols/2.2, 6.0*nrows/2.2), dpi=150, squeeze=False)
    palette = discrete_palette(np.unique(coil_counts))

    for i, n in enumerate(u_nfp):
        ax = axes[i // ncols, i % ncols]
        ids = np.where(nfp == n)[0]
        colors = np.array([palette[c] for c in coil_counts[ids]])
        ax.scatter(emb[ids, 0], emb[ids, 1], c=colors, s=5, alpha=0.75, linewidths=0)
        professional_axes(ax, f"NFP = {n}")

    # blank unused axes
    for j in range(i+1, nrows*ncols):
        axes[j // ncols, j % ncols].axis("off")

    # shared legend
    legend_elems = [Line2D([0], [0], marker='o', linestyle='',
                           markerfacecolor=palette[v], markeredgecolor='none',
                           markersize=6, label=f'{int(v)} coils')
                    for v in sorted(np.unique(coil_counts))]
    fig.legend(handles=legend_elems, title="# of Coils", fontsize=9, title_fontsize=10,
               loc="lower center", ncol=min(6, len(legend_elems)), frameon=False)

    fig.suptitle(f"CAE Latent Space — {method.upper()} Faceted by NFP", fontsize=13, y=0.99)
    plt.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(figC_png, dpi=300); fig.savefig(figC_pdf)
    plt.close(fig)
    print(f"Saved {figC_png} and {figC_pdf}")

def export_csv(emb, coil_counts, recon_mse, nfp, out_csv):
    cols = ["x", "y", "coil_count", "recon_mse"]
    data = [emb[:, 0], emb[:, 1], coil_counts, recon_mse]
    if nfp is not None:
        cols.append("nfp"); data.append(nfp)
    arr = np.column_stack(data)
    header = ",".join(cols)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")
    print(f"Saved CSV to {out_csv}")

# ---------------- Main -----------------
def main(args):
    seed = None if (args.seed is not None and args.seed < 0) else args.seed
    fix_seeds(seed)

    models_dir = Path(args.models_dir)
    out_base   = Path(args.out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    encoder, decoder = load_encoder_decoder(models_dir)
    ds = load_full_dataset(Path(args.tfrecord_dir), batch_size=args.batch_size)

    Z_list, cnt_list, nfp_list, mse_list = [], [], [], []
    for batch_inputs, targets in ds:
        enc_inputs = {"coil_data": batch_inputs["coil_data"], "coil_mask": batch_inputs["coil_mask"]}
        z_np, tokens = encoder_to_latent(encoder, enc_inputs)
        Z_list.append(z_np)

        counts = tf.reduce_sum(tf.cast(batch_inputs["coil_mask"], tf.float32), axis=1).numpy().astype(int)
        cnt_list.append(counts)

        # NFP: prefer explicit field; else derive from surface_data last token's value column (index 3)
        nfp_vals = None
        for key in ["nfp", "NFP", "surface_nfp", "surface_meta_nfp"]:
            if key in batch_inputs:
                nfp_vals = np.asarray(batch_inputs[key]).astype(int)
                break
        if nfp_vals is None and "surface_data" in batch_inputs:
            surf = batch_inputs["surface_data"].numpy()  # (B, T, 4) with cols [m, n, flag, value]
            cand = np.array(surf[:, -1])*5
            cand = cand.astype(int)   # use the last token's 'value'
            # sanity: keep integers in [1, 10]; if mostly valid, accept; else drop
            valid = (cand >= 1) & (cand <= 5)
            if valid.mean() > 0.8:
                nfp_vals = cand
        if nfp_vals is not None:
            nfp_list.append(nfp_vals)

        if decoder is not None:
            preds = forward_predict(decoder, tokens, z=tf.convert_to_tensor(z_np))
            pred_np = preds.numpy()
            target_np = targets["coil"].numpy()
            K_mask = batch_inputs["coil_mask"].numpy().shape[1]
            if pred_np.shape[1] == K_mask + 1:
                pred_np = pred_np[:, :-1, :]
                target_np = target_np[:, :-1, :]
            mse = per_sample_norm_mse(pred_np, target_np, batch_inputs["coil_mask"])
            mse_list.append(mse)

        if args.max_batches > 0 and len(Z_list) >= args.max_batches:
            break

    Z = np.concatenate(Z_list, axis=0)
    coil_counts = np.concatenate(cnt_list, axis=0)
    nfp = np.concatenate(nfp_list, axis=0) if nfp_list else None
    recon_mse = np.concatenate(mse_list, axis=0) if mse_list else np.full((Z.shape[0],), np.nan)

    # standardize latents
    Zs = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-8)

    # UMAP: full fit/transform over entire dataset; save/reuse model
    model_in  = Path(args.umap_model_in) if args.umap_model_in else None
    model_out = Path(args.umap_model_out)
    um, emb2, method = fit_or_load_umap(
        Zs, seed=seed, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric,
        model_in=model_in, model_out=model_out
    )
    print(f"Embedding method: {method}, emb shape={emb2.shape}")
    print(f"UMAP model saved/used: {model_out}")

    # Figures
    plot_fig_A(emb2, coil_counts, method, out_base=args.out_base)
    if not np.all(np.isnan(recon_mse)):
        plot_fig_B(emb2, recon_mse, method, out_base=args.out_base)
    else:
        print("Figure B skipped: decoder not found or recon MSE not computed.")
    plot_fig_C_facets_by_nfp(emb2, coil_counts, nfp, method, out_base=args.out_base)

    # CSV
    export_csv(emb2, coil_counts, recon_mse, nfp, out_csv=str(Path(args.out_base).with_suffix(".csv")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir",   type=str, required=True)
    ap.add_argument("--tfrecord_dir", type=str, required=True)
    ap.add_argument("--batch_size",   type=int, default=512)
    ap.add_argument("--max_batches",  type=int, default=-1)

    # UMAP + reproducibility
    ap.add_argument("--seed",         type=int, default=-1)  # -1 => no seed (parallel)
    ap.add_argument("--n_neighbors",  type=int, default=30)
    ap.add_argument("--min_dist",     type=float, default=0.1)
    ap.add_argument("--metric",       type=str, default="euclidean")
    ap.add_argument("--umap_model_in",  type=str, default="")
    ap.add_argument("--umap_model_out", type=str, default="umap_latents.joblib")

    # Outputs
    ap.add_argument("--out_base",     type=str, default="latents_umap")
    args = ap.parse_args()
    main(args)
