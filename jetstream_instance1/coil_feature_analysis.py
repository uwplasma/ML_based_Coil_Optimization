import tensorflow as tf
import numpy as np

# ---------- helper: pull coil tensor + mask from your (inputs, targets) ----------
def _extract_coil_and_mask(inputs, targets):
    """
    Returns:
      x:     [B, Q, D] float32 (coil feature tensor)
      mask:  [B, Q]    float32 (1 for valid token, 0 for pad)
    """
    if isinstance(targets, dict):
        x = tf.cast(targets["coil"], tf.float32)
    else:
        x = tf.cast(targets, tf.float32)
    mask = tf.cast(inputs["coil_mask"], tf.float32)
    return x, mask

# ---------- analyzer ----------
def analyze_coil_feature_stats(dataset, D=100, max_batches=None,
                               sigma_floor=1e-3, mean_abs_thresh=1e4,
                               const_range_tol=1e-12):
    sum_x  = tf.zeros([D], tf.float64)
    sum_x2 = tf.zeros([D], tf.float64)
    sum_abs_x = tf.zeros([D], tf.float64)
    zeros_cnt = tf.zeros([D], tf.float64)
    n_tokens  = tf.zeros([],  tf.float64)

    min_v = tf.fill([D], tf.constant(np.inf,  dtype=np.float64))
    max_v = tf.fill([D], tf.constant(-np.inf, dtype=np.float64))

    for i, (inputs, targets) in enumerate(dataset):
        if max_batches is not None and i >= max_batches:
            break
        x, mask = _extract_coil_and_mask(inputs, targets)      # x:[B,Q,D], mask:[B,Q]
        x64 = tf.cast(x, tf.float64)
        m   = tf.cast(mask, tf.float64)[..., tf.newaxis]       # [B,Q,1]

        sum_x     += tf.reduce_sum(x64 * m, axis=[0, 1])
        sum_x2    += tf.reduce_sum((x64 * x64) * m, axis=[0, 1])
        sum_abs_x += tf.reduce_sum(tf.abs(x64) * m, axis=[0, 1])
        zeros_cnt += tf.reduce_sum(tf.cast(tf.equal(x64, 0.0), tf.float64) * m, axis=[0, 1])

        # masked min/max
        valid_min = tf.where(m==1.0, x64, tf.fill(tf.shape(x64), tf.constant(np.inf,  dtype=np.float64)))
        valid_max = tf.where(m==1.0, x64, tf.fill(tf.shape(x64), tf.constant(-np.inf, dtype=np.float64)))
        min_v = tf.minimum(min_v, tf.reduce_min(valid_min, axis=[0, 1]))
        max_v = tf.maximum(max_v, tf.reduce_max(valid_max, axis=[0, 1]))

        # ✔️ Correct: count TOKENS, not elements — no "/ D"
        n_tokens += tf.reduce_sum(m)

    # finalize
    n_safe = tf.maximum(n_tokens, 1.0)
    mu   = sum_x / n_safe
    ex2  = sum_x2 / n_safe
    var  = tf.maximum(ex2 - tf.square(mu), 0.0)
    sigma = tf.maximum(tf.sqrt(var), sigma_floor)

    mean_abs = sum_abs_x / n_safe
    rng      = max_v - min_v
    frac_zero = zeros_cnt / n_safe

    # convert to numpy
    mu_np        = mu.numpy().astype(np.float64)
    sigma_np     = sigma.numpy().astype(np.float64)
    min_np       = min_v.numpy().astype(np.float64)
    max_np       = max_v.numpy().astype(np.float64)
    mean_abs_np  = mean_abs.numpy().astype(np.float64)
    frac_zero_np = frac_zero.numpy().astype(np.float64)
    rng_np       = rng.numpy().astype(np.float64)

    # flags
    huge_mean_abs_idx = np.where(mean_abs_np > mean_abs_thresh)[0].tolist()
    sigma_floored_idx = np.where(sigma_np <= (sigma_floor + 1e-12))[0].tolist()
    const_like_idx    = np.where(rng_np <= const_range_tol)[0].tolist()

    # quick human-friendly summaries
    top_by_mean_abs = np.argsort(-np.abs(mu_np))[:10].tolist()
    top_by_max_abs  = np.argsort(-np.maximum(np.abs(min_np), np.abs(max_np)))[:10].tolist()

    print(f"Scanned ~{int(n_tokens.numpy())} valid tokens per feature")
    print("Top 10 features by |mean|:", top_by_mean_abs)
    print("Top 10 features by max |value|:", top_by_max_abs)
    print("Features with floored σ (≈ constants):", sigma_floored_idx)
    print("Features with (max-min)≈0:", const_like_idx)
    print("Features with very large mean|x|:", huge_mean_abs_idx)

    stats = {
        "mu": mu_np,
        "sigma": sigma_np,
        "min": min_np,
        "max": max_np,
        "range": rng_np,
        "mean_abs": mean_abs_np,
        "frac_zero": frac_zero_np,
        "flags": {
            "huge_mean_abs": huge_mean_abs_idx,
            "sigma_floored": sigma_floored_idx,
            "constant_like": const_like_idx,
        },
        "top10_abs_mean": top_by_mean_abs,
        "top10_abs_max": top_by_max_abs,
    }

    # Optional: return a pandas DataFrame if available
    try:
        import pandas as pd
        df = pd.DataFrame({
            "feature": np.arange(D, dtype=int),
            "mu": mu_np,
            "sigma": sigma_np,
            "min": min_np,
            "max": max_np,
            "range": rng_np,
            "mean_abs": mean_abs_np,
            "frac_zero": frac_zero_np,
            "sigma_floored": np.isin(np.arange(D), sigma_floored_idx),
            "constant_like": np.isin(np.arange(D), const_like_idx),
            "huge_mean_abs": np.isin(np.arange(D), huge_mean_abs_idx),
        })
        stats["summary_df"] = df
    except Exception:
        pass

    return stats
