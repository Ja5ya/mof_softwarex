"""
run_xai_batch.py
================
XAI pipeline for six psychiatric ICD-10 codes.

Attribution methods
-------------------
1. SG   : Input Gradient Saliency  -- d(output)/d(input) at input level
           Primary gradient method. Bypasses GAP attenuation because
           gradients are taken at the input tensor, not a Conv feature map.
           Wagner et al. (2024) found plain saliency was the only gradient
           method that consistently passed sanity checks for ECG deep learning.

2. GC   : Grad-CAM on last Conv layer before GAP
           Retained for comparison with SG and consistency with prior work.

3. SHAP : DeepExplainer, background set of 50 samples
           Degenerate samples (raw_max < SHAP_DEGEN_MAX) are excluded from
           ALL f_E components.

4. LIME : Segment-mode Ridge regression, 200 segments / 300 perturbations
           Excluded from compactness (fixed-segment artefact produces
           uniform 0.525 across all codes). Included in continuity and
           contrastivity.

VarGrad and IG removed
-----------------------
Both methods were tested in the previous pipeline iteration. Neither
improved on SG or GC for continuity, compactness, or contrastivity.
VarGrad produced lower continuity than single-pass Grad-CAM, inconsistent
with its design intent, likely due to GAP diffusing the gradient variance
signal. IG showed no meaningful advantage over SG. Both are therefore
dropped to reduce computational cost and simplify the pipeline.

f_E components (three, equal weights 1/3 each)
----------------------------------------------
  continuity    : Jaccard top-k stability under noise
                  pooled over SG + GC + non-degenerate SHAP + LIME
  compactness   : fraction of timesteps below 10% of max attribution
                  averaged over SG + GC + non-degenerate SHAP
                  (LIME excluded: fixed-segment structure gives uninformative
                  uniform value across all codes)
  contrastivity : normalised L2 distance between psychiatric/normal maps
                  averaged over SG + GC + non-degenerate SHAP + LIME

AOPC correctness: removed from f_E
------------------------------------
All three gradient methods tested (GC, VarGrad, IG) produced negative AOPC
under mean-fill deletion, meaning model confidence increases rather than
decreases when timesteps are deleted. This reflects the GAP attenuation
problem and distributional artefacts, not faithfulness of the attributions.
Correctness is assessed qualitatively via the MPRT sanity check instead.

MPRT sanity check
-----------------
One-time diagnostic per code. Randomises all model weights, recomputes
SG and GC for 20 samples, measures Jaccard overlap between original and
randomised attributions. Low Jaccard = attribution sensitive to model = PASS.
Reference: Adebayo et al. (2018); Wagner et al. (2024).

Usage
-----
    python run_xai_batch.py --code F329
    python run_xai_batch.py --code all --n_samples 200 --shap_bg 50
    python run_xai_batch.py --code F329 --skip_sanity
    # f_E for every hyper_sweep run (arch × lr_dr), 200 samples each by default:
    python run_xai_batch.py --hyper_sweep_fe --code all
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from ecg_utils import demographic_match
from mof.paths import dataset_dir, shared_dir
from pipeline_config import PipelineConfig, add_config_arg, load_config_or_snapshot

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — resolved from config in main()
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
MODELS_PARENT = os.environ.get("MOF_MODELS_PARENT", str(_SRC_DIR))
if MODELS_PARENT not in sys.path:
    sys.path.insert(0, MODELS_PARENT)

DATA_ROOT = ""
PIPE_CFG: PipelineConfig | None = None

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SAMPLES      = 200
TOP_K          = 500
NOISE_SIGMA    = 0.05
NOISE_REPS     = 5
ATTR_THRESH    = 0.1
LIME_SEGS      = 200
LIME_SAMPS     = 300
SHAP_BG        = 50
SHAP_DEGEN_MAX = 0.001
FS             = 500
_SPLIT_RANDOM_STATE = 42

SANITY_N               = 20
SANITY_PASS_THRESHOLD  = 0.30


# ===========================================================================
# TF / SHAP import
# ===========================================================================
def _import_tf_and_shap():
    import tensorflow as tf
    import shap
    tf.get_logger().setLevel("ERROR")
    return tf, shap


# ===========================================================================
# Data loading
# ===========================================================================
def load_test_fold(code, pipe_cfg: PipelineConfig):
    from sklearn.model_selection import train_test_split

    data_dir = dataset_dir(pipe_cfg, code)
    shared = shared_dir(pipe_cfg)

    test_pkl = data_dir / "test_data.pkl"
    X_npy = data_dir / "X_test.npy"

    if test_pkl.is_file():
        with open(test_pkl, "rb") as f:
            td = pickle.load(f)
        X_test, y_test = (td if isinstance(td, (list, tuple)) and len(td) == 2
                          else (td["X"], td["y"]))
    elif X_npy.is_file():
        X_test = np.load(X_npy)
        y_test = np.load(data_dir / "y_test.npy")
    else:
        with open(data_dir / "recs_psych.pkl", "rb") as f:
            recs_psych = pickle.load(f)
        with open(shared / "recs_normal.pkl", "rb") as f:
            recs_normal = pickle.load(f)
        matched_p, matched_n = demographic_match(
            recs_psych, recs_normal, pipe_cfg.demographic_matching
        )
        all_recs = matched_p + matched_n
        signals  = np.array([r["signal"] for r in all_recs], dtype=np.float32)
        labels   = np.array([r["label"]  for r in all_recs], dtype=np.int64)
        X_all    = np.ascontiguousarray(np.transpose(signals, (0, 2, 1)))
        y_int    = labels.astype(np.int64)
        _, X_test, _, yi_test = train_test_split(
            X_all, y_int, test_size=0.2, stratify=y_int,
            random_state=_SPLIT_RANDOM_STATE,
        )
        y_test = yi_test

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).ravel()
    print(f"  [{code}] X_test={X_test.shape}, class balance={np.bincount(y_test)}")
    return X_test, y_test


def resolve_run_dir(code, cfg, pipe_cfg: PipelineConfig):
    """cfg: best-config entry {'arch','lr','dr'} or sweep {'arch','run_dir_name'}."""
    arch = cfg["arch"]
    if "run_dir_name" in cfg:
        sub = cfg["run_dir_name"]
    else:
        sub = f"lr{cfg['lr']}_dr{cfg['dr']}"
    return os.path.join(pipe_cfg.output_root, "hyper_sweep", code, arch, sub)


def discover_sweep_runs(code, pipe_cfg: PipelineConfig, only_arch=None, only_run_dir=None):
    """All trained runs under hyper_sweep/<code> that have test_predictions.npz."""
    base = os.path.join(pipe_cfg.output_root, "hyper_sweep", code)
    runs = []
    if not os.path.isdir(base):
        return runs
    for arch in sorted(os.listdir(base)):
        if only_arch is not None and arch != only_arch:
            continue
        arch_path = os.path.join(base, arch)
        if not os.path.isdir(arch_path):
            continue
        for name in sorted(os.listdir(arch_path)):
            if only_run_dir is not None and name != only_run_dir:
                continue
            run_path = os.path.join(arch_path, name)
            if not os.path.isdir(run_path):
                continue
            if os.path.isfile(os.path.join(run_path, "test_predictions.npz")):
                runs.append({"arch": arch, "run_dir_name": name})
    return runs


def _parse_dropout_from_run_dir(run_dir_name):
    """e.g. lr1e-3_dr0.5 ->0.5 (for model_factory when only sweep folder is known)."""
    if "_dr" not in run_dir_name:
        raise ValueError(f"Cannot parse dropout from {run_dir_name!r}")
    return float(run_dir_name.split("_dr", 1)[1])


def load_predictions(code, cfg, pipe_cfg: PipelineConfig):
    run_dir = resolve_run_dir(code, cfg, pipe_cfg)
    preds_path = os.path.join(run_dir, "test_predictions.npz")
    z          = np.load(preds_path)
    y_true     = np.asarray(z["y_true"]).flatten().astype(np.int32)
    for key in ("y_score", "y_pred", "y_proba"):
        if key in z.files:
            return y_true, np.asarray(z[key]).flatten()
    raise KeyError(f"No probability key in {preds_path}: {z.files}")


def load_model(code, tf, cfg, pipe_cfg: PipelineConfig):
    run_dir = resolve_run_dir(code, cfg, pipe_cfg)
    dr = float(cfg["dr"]) if "dr" in cfg else _parse_dropout_from_run_dir(
        cfg["run_dir_name"]
    )
    ecg = pipe_cfg.ecg
    for fname in ("model.keras", "model.h5"):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            print(f"  [{code}] Loading model from {fname}")
            m = tf.keras.models.load_model(fpath, compile=False)
            m.trainable = False
            return m
    print(f"  [{code}] Rebuilding from weights ...")
    from models.model_factory import create_model
    tf.keras.backend.clear_session()
    m = create_model(
        cfg["arch"],
        input_shape=(ecg.signal_len, ecg.n_leads),
        num_classes=1,
        dropout_rate=dr,
    )
    m.load_weights(os.path.join(run_dir, "best_val_auc.weights.h5"))
    m.trainable = False
    return m


def select_samples(y_true, y_pred_proba, n=N_SAMPLES, seed=42):
    rng    = np.random.default_rng(seed)
    tp_idx = np.where((y_true == 1) & (y_pred_proba >= 0.75))[0]
    tn_idx = np.where((y_true == 0) & (y_pred_proba <= 0.25))[0]
    bd_idx = np.where(np.abs(y_pred_proba - 0.5) <= 0.10)[0]

    quota    = n // 3
    selected = []
    for pool in [tp_idx, tn_idx, bd_idx]:
        k = min(quota, len(pool))
        if k > 0:
            selected.extend(rng.choice(pool, k, replace=False).tolist())

    remaining = n - len(selected)
    if remaining > 0:
        used     = set(selected)
        leftover = [i for i in np.concatenate([tp_idx, tn_idx, bd_idx])
                    if i not in used]
        extra    = min(remaining, len(leftover))
        if extra > 0:
            selected.extend(rng.choice(leftover, extra, replace=False).tolist())

    selected = list(dict.fromkeys(selected))[:n]
    print(f"    TP pool={len(tp_idx)}, TN pool={len(tn_idx)}, BD pool={len(bd_idx)}"
          f" -> selected {len(selected)}")
    return selected


# ===========================================================================
# Shared utilities
# ===========================================================================
def normalise_attr(attr):
    mx = np.max(np.abs(attr))
    return np.abs(attr) / mx if mx > 0 else np.abs(attr)


def _attr_1d(x):
    x = np.asarray(x)
    return np.abs(x).mean(axis=-1).ravel() if x.ndim > 1 else np.abs(x).ravel()


def topk_jaccard(a, b, k=TOP_K):
    a, b  = _attr_1d(a), _attr_1d(b)
    k     = min(int(k), len(a), len(b))
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 0.0


# ===========================================================================
# Attribution methods
# ===========================================================================

def get_input_gradient_saliency(model, x_sample, tf):
    """
    Plain input gradient saliency: |d(output)/d(input)|.
    Primary gradient method. Operates at input level, bypasses GAP.
    """
    x_var = tf.Variable(tf.cast(x_sample[np.newaxis], tf.float32))
    with tf.GradientTape() as tape:
        preds = model(x_var, training=False)
        loss  = preds[:, 0]
    grads = tape.gradient(loss, x_var)
    if grads is None:
        raise RuntimeError("SG: gradients are None.")
    sg = np.abs(grads.numpy()[0])       # (5000, 12)
    return normalise_attr(sg.mean(axis=-1))


def get_gradcam(model, x_sample, tf, layer_name=None):
    """Grad-CAM on last Conv layer before GAP. Retained for comparison."""
    from scipy.ndimage import zoom

    last_conv = None
    if layer_name:
        last_conv = model.get_layer(layer_name)
    else:
        for name in ("spatial_conv",):
            try:
                last_conv = model.get_layer(name)
                break
            except ValueError:
                pass
        if last_conv is None:
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D,
                                      tf.keras.layers.Conv1D)):
                    last_conv = layer
                    break
    if last_conv is None:
        raise ValueError("No Conv layer found for GradCAM.")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output]
    )
    x = tf.cast(x_sample[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError("GradCAM: gradients are None.")

    if len(conv_out.shape) == 4:
        pooled = tf.reduce_mean(grads, axis=(1, 2))
        cam    = tf.reduce_sum(conv_out * pooled[:, tf.newaxis, tf.newaxis, :],
                               axis=-1)
        cam    = tf.squeeze(cam, axis=-1)[0].numpy()
        if cam.ndim == 2:
            cam = cam.mean(axis=-1)
    else:
        pooled = tf.reduce_mean(grads, axis=1)
        cam    = tf.reduce_sum(conv_out * pooled[:, tf.newaxis, :],
                               axis=-1)[0].numpy()

    tlen = int(x_sample.shape[0])
    if len(cam) != tlen:
        cam = zoom(cam, tlen / len(cam), order=1)
    return normalise_attr(np.maximum(cam, 0.0))


def get_shap_deep(model, x_sample, X_background, shap_lib):
    """DeepExplainer SHAP. Returns (attr_1d, raw_max, raw_mean)."""
    explainer = shap_lib.DeepExplainer(
        model, X_background.astype(np.float32)
    )
    shap_vals = explainer.shap_values(
        x_sample[np.newaxis].astype(np.float32)
    )
    raw = shap_vals[0] if isinstance(shap_vals, (list, tuple)) else shap_vals
    sv  = np.asarray(raw)
    while sv.ndim > 2 and sv.shape[0] == 1:
        sv = sv[0]
    if sv.ndim == 1:
        sv = sv[:, np.newaxis]
    attr_1d  = np.abs(sv).mean(axis=-1).ravel().astype(np.float32)
    raw_max  = float(np.max(np.abs(attr_1d)))
    raw_mean = float(np.mean(np.abs(attr_1d)))
    return attr_1d, raw_max, raw_mean


def get_lime(model, x_sample):
    """Segment-mode LIME with Ridge regression.

    Uses the sample's time length (works for MIMIC 5000 and beat-length inputs).
    """
    from sklearn.linear_model import Ridge

    T = int(x_sample.shape[0])
    n_segs = min(LIME_SEGS, T)
    seg_len = max(1, T // n_segs)
    segments = np.array([min(t // seg_len, n_segs - 1) for t in range(T)])
    x0 = x_sample.copy().astype(np.float32)

    np.random.seed(42)
    Z = np.random.binomial(1, 0.5, size=(LIME_SAMPS, n_segs))
    preds = np.zeros(LIME_SAMPS, dtype=np.float32)
    for start in range(0, LIME_SAMPS, 32):
        end = min(start + 32, LIME_SAMPS)
        batch = []
        for i in range(start, end):
            x_p = x0.copy()
            for seg_id in range(n_segs):
                if Z[i, seg_id] == 0:
                    x_p[segments == seg_id, :] = 0.0
            batch.append(x_p)
        preds[start:end] = model.predict(
            np.stack(batch, axis=0), verbose=0
        ).flatten()

    distances = np.sqrt(np.sum((Z - 1.0) ** 2, axis=1))
    kernel_w = np.exp(-distances ** 2 / (2 * 0.25 ** 2))
    reg = Ridge(alpha=1.0)
    reg.fit(Z, preds, sample_weight=kernel_w)
    lime_1d = np.array([reg.coef_[segments[t]] for t in range(T)])
    return normalise_attr(lime_1d)


# ===========================================================================
# Co-12 metrics
# ===========================================================================
def continuity(attr_fn, x_sample, n_repeats=NOISE_REPS, sigma=NOISE_SIGMA):
    base   = attr_fn(x_sample)
    scores = []
    for _ in range(n_repeats):
        noisy  = x_sample + np.random.normal(0, sigma, x_sample.shape).astype(np.float32)
        scores.append(topk_jaccard(base, attr_fn(noisy)))
    return float(np.mean(scores)), float(np.std(scores)), base


def compactness(attr_1d, thresh_frac=ATTR_THRESH):
    a      = _attr_1d(attr_1d)
    thresh = thresh_frac * float(np.max(a))
    return 1.0 - float(np.mean(a > thresh))


def contrastivity(attr_pos, attr_neg):
    diff = attr_pos.astype(float) - attr_neg.astype(float)
    return min(float(np.linalg.norm(diff)) / np.sqrt(len(attr_pos)), 1.0)


def coherence_all(sg, gc, sh, lm):
    """Pairwise Jaccard over all four methods (diagnostic only)."""
    pairs = {
        "sg_gc": topk_jaccard(sg, gc),
        "sg_sh": topk_jaccard(sg, sh),
        "sg_lm": topk_jaccard(sg, lm),
        "gc_sh": topk_jaccard(gc, sh),
        "gc_lm": topk_jaccard(gc, lm),
        "sh_lm": topk_jaccard(sh, lm),
    }
    return float(np.mean(list(pairs.values()))), pairs


# ===========================================================================
# MPRT Sanity Check
# ===========================================================================
def run_sanity_check(model, X_test, tf, n_samples=SANITY_N, seed=42):
    """
    Model Parameter Randomisation Test for SG and GC.
    Low Jaccard between original and randomised attributions = PASS.
    Reference: Adebayo et al. (2018); Wagner et al. (2024).
    """
    rng     = np.random.default_rng(seed)
    n_draw  = min(n_samples, len(X_test))
    indices = rng.choice(len(X_test), n_draw, replace=False)

    print(f"\n  Running MPRT sanity check (N={n_draw} samples) ...")

    # Attributions on trained model
    sg_orig, gc_orig = [], []
    for idx in indices:
        x = X_test[idx]
        T = int(x.shape[0])
        try:
            sg_orig.append(get_input_gradient_saliency(model, x, tf))
        except Exception:
            sg_orig.append(np.zeros(T, dtype=np.float32))
        try:
            gc_orig.append(get_gradcam(model, x, tf))
        except Exception:
            gc_orig.append(np.zeros(T, dtype=np.float32))

    # Randomise all model weights
    rand_model = tf.keras.models.clone_model(model)
    new_weights = [
        np.random.default_rng(seed + 1).normal(0, 0.01, w.shape).astype(w.dtype)
        for w in model.get_weights()
    ]
    rand_model.set_weights(new_weights)
    rand_model.trainable = False

    # Attributions on randomised model
    sg_rand, gc_rand = [], []
    for idx in indices:
        x = X_test[idx]
        T = int(x.shape[0])
        try:
            sg_rand.append(get_input_gradient_saliency(rand_model, x, tf))
        except Exception:
            sg_rand.append(np.zeros(T, dtype=np.float32))
        try:
            gc_rand.append(get_gradcam(rand_model, x, tf))
        except Exception:
            gc_rand.append(np.zeros(T, dtype=np.float32))

    sg_j = [topk_jaccard(o, r) for o, r in zip(sg_orig, sg_rand)]
    gc_j = [topk_jaccard(o, r) for o, r in zip(gc_orig, gc_rand)]

    results = {}
    for name, scores in [("SG", sg_j), ("GC", gc_j)]:
        mean_j = float(np.mean(scores))
        std_j  = float(np.std(scores))
        passed = mean_j < SANITY_PASS_THRESHOLD
        status = "PASS" if passed else "FAIL"
        print(f"    MPRT [{name}]: mean_jaccard={mean_j:.3f} (+-{std_j:.3f}) "
              f"threshold={SANITY_PASS_THRESHOLD} -> {status}")
        if not passed:
            print(f"    WARNING: {name} is insensitive to model weights. "
                  f"Do not use for clinical interpretation.")
        results[name] = {
            "mean_jaccard": mean_j,
            "std_jaccard":  std_j,
            "passed":       passed,
        }

    return results


# ===========================================================================
# f_E (three components)
# ===========================================================================
def compute_f_E(continuity_val, compactness_val, contrastivity_val,
                w_cont=1/3, w_comp=1/3, w_contr=1/3):
    """
    f_E = weighted mean of continuity, compactness, contrastivity.

    Each component pools across all four methods (SG, GC, SHAP, LIME)
    where valid:
      continuity    : SG + GC + non-degenerate SHAP + LIME
      compactness   : SG + GC + non-degenerate SHAP  (LIME excluded)
      contrastivity : SG + GC + non-degenerate SHAP + LIME

    LIME excluded from compactness because its fixed 200-segment structure
    produces a structurally determined value (~0.525) invariant across codes.
    """
    assert abs(w_cont + w_comp + w_contr - 1.0) < 1e-5, "Weights must sum to 1.0"
    return (w_cont  * continuity_val +
            w_comp  * compactness_val +
            w_contr * contrastivity_val)


# ===========================================================================
# Per-code pipeline
# ===========================================================================
def run_code(code, tf, shap_lib, pipe_cfg: PipelineConfig,
             n_samples=N_SAMPLES, shap_bg=SHAP_BG,
             run_sanity=True, sweep_cfg=None):
    model_cfg = (
        sweep_cfg if sweep_cfg is not None else pipe_cfg.best_xai_config(code)
    )
    data_root = pipe_cfg.output_root
    cfg_tag = (f"{model_cfg['arch']}/{model_cfg['run_dir_name']}"
               if sweep_cfg is not None else
               f"{model_cfg['arch']}/lr{model_cfg['lr']}_dr{model_cfg['dr']}")

    print(f"\n{'='*60}")
    print(f"  CODE = {code}  cfg = {cfg_tag}  "
          f"(n_samples={n_samples}, shap_bg={shap_bg})")
    print(f"{'='*60}")

    if sweep_cfg is not None:
        out_dir = os.path.join(
            data_root, "xai_results", "hyper_sweep_fe",
            code, sweep_cfg["arch"], sweep_cfg["run_dir_name"],
        )
    else:
        out_dir = os.path.join(data_root, "xai_results", code)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"xai_raw_results_{code}.pkl")

    if os.path.exists(raw_path):
        with open(raw_path, "rb") as f:
            results = pickle.load(f)
        print(f"  Resuming: {len(results)} samples already done.")
    else:
        results = []

    done_indices = {r["sample_idx"] for r in results}

    X_test, y_test       = load_test_fold(code, pipe_cfg)
    y_true_pred, y_proba = load_predictions(code, model_cfg, pipe_cfg)
    model                = load_model(code, tf, model_cfg, pipe_cfg)

    n_test = len(X_test)
    effective_shap_bg = min(shap_bg, n_test)
    if effective_shap_bg < shap_bg:
        print(
            f"  WARNING: shap_bg={shap_bg} > n_test={n_test}; "
            f"using shap_bg={effective_shap_bg}"
        )

    np.random.seed(42)
    bg_idx = np.random.choice(n_test, effective_shap_bg, replace=False)
    X_bg   = X_test[bg_idx]
    print(f"  SHAP background: {X_bg.shape}")

    # -----------------------------------------------------------------------
    # MPRT sanity check (once per code, before main loop)
    # -----------------------------------------------------------------------
    sanity_results = {}
    if run_sanity:
        sanity_results = run_sanity_check(model, X_test, tf)
        sanity_path = os.path.join(out_dir, f"sanity_check_{code}.json")
        with open(sanity_path, "w") as f:
            json.dump(sanity_results, f, indent=2)
        print(f"  Sanity check saved: {sanity_path}\n")

    # -----------------------------------------------------------------------
    # Main attribution loop
    # -----------------------------------------------------------------------
    selected = select_samples(y_true_pred, y_proba, n=min(n_samples, n_test))
    todo     = [i for i in selected if i not in done_indices]
    print(f"  {len(todo)} samples to process (of {len(selected)} total)")

    def sg_fn(x): return get_input_gradient_saliency(model, x, tf)
    def gc_fn(x): return get_gradcam(model, x, tf)
    def lm_fn(x): return get_lime(model, x)

    for i, idx in enumerate(todo):
        x   = X_test[idx]
        yt  = int(y_true_pred[idx])
        yp  = float(y_proba[idx])
        lbl = "Psychiatric" if yt == 1 else "Normal"
        print(f"\n  Sample {i+1}/{len(todo)} | idx={idx} | {lbl} | pred={yp:.3f}")

        # --- [1] Input Gradient Saliency ---
        print("    [1/4] SG (Input Gradient Saliency) ...")
        cont_sg, std_sg, sg_attr = continuity(sg_fn, x)
        comp_sg = compactness(sg_attr)
        print(f"          cont={cont_sg:.3f} (+-{std_sg:.3f})  comp={comp_sg:.3f}")

        # --- [2] Grad-CAM ---
        print("    [2/4] GC (Grad-CAM) ...")
        cont_gc, std_gc, gc_attr = continuity(gc_fn, x)
        comp_gc = compactness(gc_attr)
        print(f"          cont={cont_gc:.3f} (+-{std_gc:.3f})  comp={comp_gc:.3f}")

        # --- [3] SHAP ---
        print("    [3/4] SHAP (DeepExplainer) ...")
        sh_degen = False
        sh_raw_max = sh_raw_mean = 0.0
        cont_sh = std_sh = comp_sh = None
        sh_attr = np.zeros(int(x.shape[0]), dtype=np.float32)
        try:
            raw_attr, sh_raw_max, sh_raw_mean = get_shap_deep(model, x, X_bg, shap_lib)
            if sh_raw_max < SHAP_DEGEN_MAX:
                sh_degen = True
                sh_attr  = raw_attr
                print(f"          DEGENERATE (raw_max={sh_raw_max:.4g}) "
                      f"-- excluded from ALL f_E components")
            else:
                sh_attr = raw_attr
                def sh_fn_bound(x_in):
                    a, _, _ = get_shap_deep(model, x_in, X_bg, shap_lib)
                    return a
                cont_sh, std_sh, _ = continuity(sh_fn_bound, x)
                comp_sh = compactness(sh_attr)
                print(f"          cont={cont_sh:.3f} (+-{std_sh:.3f})  "
                      f"comp={comp_sh:.3f}  raw_max={sh_raw_max:.4g}")
        except Exception as e:
            sh_degen = True
            print(f"          SHAP failed: {e} -- excluded from ALL f_E components")

        # --- [4] LIME ---
        print("    [4/4] LIME ...")
        cont_lm, std_lm, lm_attr = continuity(lm_fn, x)
        comp_lm = compactness(lm_attr)
        print(f"          cont={cont_lm:.3f} (+-{std_lm:.3f})  "
              f"comp={comp_lm:.3f} (ref only, excluded from f_E compactness)")

        # --- Coherence across all four methods (diagnostic) ---
        coher_score, coher_detail = coherence_all(sg_attr, gc_attr, sh_attr, lm_attr)
        print(f"          Coherence (diagnostic)={coher_score:.3f}")

        results.append({
            "code": code, "sample_idx": idx,
            "true_label": yt, "pred_proba": yp,
            # Attribution vectors
            "sg_attr": sg_attr,
            "gc_attr": gc_attr,
            "sh_attr": sh_attr,
            "lm_attr": lm_attr,
            # Continuity: all four methods
            "continuity_sg": cont_sg, "continuity_sg_std": std_sg,
            "continuity_gc": cont_gc, "continuity_gc_std": std_gc,
            "continuity_sh": cont_sh, "continuity_sh_std": std_sh,
            "continuity_lm": cont_lm, "continuity_lm_std": std_lm,
            # Compactness: SG + GC + SHAP (LIME is reference only)
            "compactness_sg": comp_sg,
            "compactness_gc": comp_gc,
            "compactness_sh": comp_sh,   # None if degenerate
            "compactness_lm": comp_lm,   # stored for reference
            # SHAP flags
            "shap_degenerate": sh_degen,
            "shap_raw_max":    sh_raw_max,
            "shap_raw_mean":   sh_raw_mean,
            # Coherence (diagnostic)
            "coherence":        coher_score,
            "coherence_detail": coher_detail,
        })

        with open(raw_path, "wb") as f:
            pickle.dump(results, f)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------

    # ── Continuity: SG + GC + non-degenerate SH + LM ──────────────────────
    cont_sg_vals = [r["continuity_sg"] for r in results]
    cont_gc_vals = [r["continuity_gc"] for r in results]
    cont_sh_vals = [r["continuity_sh"] for r in results
                    if not r["shap_degenerate"] and r["continuity_sh"] is not None]
    cont_lm_vals = [r["continuity_lm"] for r in results]
    cont_valid   = cont_sg_vals + cont_gc_vals + cont_sh_vals + cont_lm_vals
    continuity_agg = float(np.mean(cont_valid)) if cont_valid else None

    # ── Compactness: SG + GC + non-degenerate SH  (LIME excluded) ─────────
    comp_sg_vals = [r["compactness_sg"] for r in results]
    comp_gc_vals = [r["compactness_gc"] for r in results]
    comp_sh_vals = [r["compactness_sh"] for r in results
                    if r["compactness_sh"] is not None]
    comp_valid   = comp_sg_vals + comp_gc_vals + comp_sh_vals
    compactness_agg = float(np.mean(comp_valid)) if comp_valid else None

    # ── Contrastivity: SG + GC + non-degenerate SH + LM ──────────────────
    pos_all = [r for r in results if r["true_label"] == 1]
    neg_all = [r for r in results if r["true_label"] == 0]
    ct_sg = ct_gc = ct_sh = ct_lm = ct_mean = None
    if pos_all and neg_all:
        ct_sg = float(np.mean([
            contrastivity(p["sg_attr"], n["sg_attr"])
            for p in pos_all for n in neg_all
        ]))
        ct_gc = float(np.mean([
            contrastivity(p["gc_attr"], n["gc_attr"])
            for p in pos_all for n in neg_all
        ]))
        pos_sh = [r for r in pos_all if not r["shap_degenerate"]]
        neg_sh = [r for r in neg_all if not r["shap_degenerate"]]
        if pos_sh and neg_sh:
            ct_sh = float(np.mean([
                contrastivity(p["sh_attr"], n["sh_attr"])
                for p in pos_sh for n in neg_sh
            ]))
        ct_lm = float(np.mean([
            contrastivity(p["lm_attr"], n["lm_attr"])
            for p in pos_all for n in neg_all
        ]))
        ct_vals = [v for v in [ct_sg, ct_gc, ct_sh, ct_lm] if v is not None]
        ct_mean = float(np.mean(ct_vals))
        print(f"\n  Contrastivity: SG={ct_sg:.4f}  GC={ct_gc:.4f}  "
              f"SH={'N/A' if ct_sh is None else f'{ct_sh:.4f}'}  "
              f"LM={ct_lm:.4f}  mean={ct_mean:.4f}")
    else:
        print(f"  WARNING [{code}]: No TP+TN pairs.")

    # ── f_E ────────────────────────────────────────────────────────────────
    valid = [v for v in [continuity_agg, compactness_agg, ct_mean] if v is not None]
    f_E   = compute_f_E(*valid) if len(valid) == 3 else (
        float(np.mean(valid)) if valid else None
    )

    n_degen    = sum(1 for r in results if r["shap_degenerate"])
    coher_vals = [r["coherence"] for r in results]

    summary = {
        "code": code,
        "sweep_arch":            model_cfg["arch"],
        "sweep_run":             (model_cfg["run_dir_name"] if sweep_cfg is not None
                                 else f"lr{model_cfg['lr']}_dr{model_cfg['dr']}"),
        "n_samples":             len(results),
        "n_shap_degenerate":     n_degen,
        "n_shap_degenerate_pct": round(100 * n_degen / len(results), 1) if results else None,
        # f_E and three components
        "f_E":                   f_E,
        "continuity_valid":      continuity_agg,
        "compactness_valid":     compactness_agg,
        "contrastivity_valid":   ct_mean,
        # Per-method continuity
        "continuity_sg":         float(np.mean(cont_sg_vals)) if cont_sg_vals else None,
        "continuity_gc":         float(np.mean(cont_gc_vals)) if cont_gc_vals else None,
        "continuity_sh_nondegen":float(np.mean(cont_sh_vals)) if cont_sh_vals else None,
        "continuity_lm":         float(np.mean(cont_lm_vals)) if cont_lm_vals else None,
        # Per-method compactness
        "compactness_sg":        float(np.mean(comp_sg_vals)) if comp_sg_vals else None,
        "compactness_gc":        float(np.mean(comp_gc_vals)) if comp_gc_vals else None,
        "compactness_sh_nondegen":float(np.mean(comp_sh_vals)) if comp_sh_vals else None,
        "compactness_lime_ref":  float(np.mean([r["compactness_lm"] for r in results])),
        # Per-method contrastivity
        "contrastivity_sg":      ct_sg,
        "contrastivity_gc":      ct_gc,
        "contrastivity_sh_nondegen": ct_sh,
        "contrastivity_lm":      ct_lm,
        # MPRT sanity check
        "sanity_sg_passed":      sanity_results.get("SG", {}).get("passed"),
        "sanity_sg_jaccard":     sanity_results.get("SG", {}).get("mean_jaccard"),
        "sanity_gc_passed":      sanity_results.get("GC", {}).get("passed"),
        "sanity_gc_jaccard":     sanity_results.get("GC", {}).get("mean_jaccard"),
        # Coherence (diagnostic)
        "coherence_mean":        float(np.mean(coher_vals)),
        "coherence_std":         float(np.std(coher_vals)),
    }

    summary_path = os.path.join(out_dir, f"xai_summary_{code}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    def _fmt4(x):
        return f"{x:.4f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "N/A"

    print(f"\n  [{code}] DONE")
    print(f"    f_E={_fmt4(f_E)}  cont={_fmt4(continuity_agg)}  "
          f"comp={_fmt4(compactness_agg)}  contr={_fmt4(ct_mean)}")
    cont_sg_str = f"{summary['continuity_sg']:.3f}" if summary.get('continuity_sg') is not None else 'N/A'
    cont_gc_str = f"{summary['continuity_gc']:.3f}" if summary.get('continuity_gc') is not None else 'N/A'
    cont_sh_str = f"{summary['continuity_sh_nondegen']:.3f}" if cont_sh_vals else 'N/A'
    cont_lm_str = f"{summary['continuity_lm']:.3f}" if summary.get('continuity_lm') is not None else 'N/A'
    print(f"    SG={cont_sg_str}  GC={cont_gc_str}  SH={cont_sh_str}  LM={cont_lm_str}")
    if sanity_results:
        sg_s = "PASS" if sanity_results.get("SG", {}).get("passed") else "FAIL"
        gc_s = "PASS" if sanity_results.get("GC", {}).get("passed") else "FAIL"
        print(f"    MPRT: SG={sg_s}  GC={gc_s}")

    return summary


# ===========================================================================
# CSV append
# ===========================================================================
def append_to_batch_csv(summary, csv_path):
    new_row = pd.DataFrame([summary])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[existing["code"] != summary["code"]]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(csv_path, index=False)
    print(f"  Batch CSV updated ({len(combined)} codes so far): {csv_path}")


def append_to_sweep_fe_csv(summary, csv_path):
    """One row per (code, sweep_arch, sweep_run); replaces matching row if present."""
    new_row = pd.DataFrame([summary])
    keys = ["code", "sweep_arch", "sweep_run"]
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        mask = pd.Series(True, index=existing.index)
        for k in keys:
            if k in existing.columns and k in summary:
                mask &= existing[k].astype(str) == str(summary[k])
        existing = existing[~mask]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(csv_path, index=False)
    print(f"  Sweep f_E CSV ({len(combined)} rows): {csv_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="XAI pipeline: SG + GC + SHAP + LIME"
    )
    add_config_arg(parser)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--code",        nargs="+", default=["all"])
    parser.add_argument("--shap_bg",     type=int,  default=None)
    parser.add_argument("--n_samples",   type=int,  default=None)
    parser.add_argument("--hyper_sweep_fe", action="store_true",
                        help="Run f_E for every hyper_sweep config (not only Pareto-best)")
    parser.add_argument("--sweep_fe_csv", type=str, default=None,
                        help="Output path for sweep summary CSV (default: "
                             "xai_results/hyper_sweep_fe/sweep_fe_summary.csv). "
                             "Set per ICD when using parallel Slurm tasks so jobs "
                             "do not clobber the same file.")
    parser.add_argument("--sweep_only_arch", type=str, default=None,
                        help="With --hyper_sweep_fe, run only this architecture.")
    parser.add_argument("--sweep_only_run", type=str, default=None,
                        help="With --hyper_sweep_fe, run only this run_dir "
                             "(e.g. lr2e-4_dr0.1619).")
    parser.add_argument("--sweep_sanity", action="store_true",
                        help="With --hyper_sweep_fe, run MPRT per config (slow)")
    parser.add_argument("--skip_sanity", action="store_true",
                        help="Skip MPRT sanity checks")
    args  = parser.parse_args()

    pipe_cfg = load_config_or_snapshot(args.config, data_root=args.data_root)
    if args.data_root:
        pipe_cfg.output_root = args.data_root

    valid_codes = pipe_cfg.diagnoses.codes
    codes = list(valid_codes) if "all" in args.code else args.code
    for c in codes:
        if c not in valid_codes:
            raise ValueError(f"Unknown code: {c}. Options: {valid_codes}")

    n_samples = args.n_samples if args.n_samples is not None else pipe_cfg.xai.n_samples
    shap_bg = args.shap_bg if args.shap_bg is not None else pipe_cfg.xai.shap_bg

    print(f"Codes:        {codes}")
    print(f"N_SAMPLES:    {n_samples}")
    print(f"Mode:         {'hyper_sweep_fe (all configs)' if args.hyper_sweep_fe else 'Pareto-best only'}")
    print(f"Methods:      SG (primary) | GC | SHAP | LIME")
    print(f"f_E:          continuity + compactness + contrastivity  (1/3 each)")
    print(f"  continuity  = SG + GC + non-degen SHAP + LIME")
    print(f"  compactness = SG + GC + non-degen SHAP  (LIME excluded)")
    print(f"  contrastivity = SG + GC + non-degen SHAP + LIME")
    if args.hyper_sweep_fe:
        print(f"MPRT sanity:  {'per config' if args.sweep_sanity else 'OFF (use --sweep_sanity to enable)'}")
    else:
        print(f"MPRT sanity:  {'SKIPPED' if args.skip_sanity else 'ENABLED'}")

    tf, shap_lib = _import_tf_and_shap()

    xai_out_root = os.path.join(pipe_cfg.output_root, "xai_results")
    os.makedirs(xai_out_root, exist_ok=True)
    csv_path = os.path.join(xai_out_root, "batch_xai_summary.csv")
    sweep_csv = args.sweep_fe_csv or os.path.join(
        xai_out_root, "hyper_sweep_fe", "sweep_fe_summary.csv"
    )

    if args.hyper_sweep_fe:
        os.makedirs(os.path.dirname(os.path.abspath(sweep_csv)), exist_ok=True)
        run_sanity = args.sweep_sanity
        for code in codes:
            runs = discover_sweep_runs(
                code,
                pipe_cfg,
                only_arch=args.sweep_only_arch,
                only_run_dir=args.sweep_only_run,
            )
            print(f"\n  [{code}] {len(runs)} sweep configs with test_predictions.npz")
            if args.sweep_only_arch is not None or args.sweep_only_run is not None:
                print(
                    f"    filters: arch={args.sweep_only_arch} "
                    f"run={args.sweep_only_run}"
                )
            for sweep_cfg in runs:
                summary = run_code(
                    code, tf, shap_lib, pipe_cfg,
                    n_samples=n_samples,
                    shap_bg=shap_bg,
                    run_sanity=run_sanity,
                    sweep_cfg=sweep_cfg,
                )
                append_to_sweep_fe_csv(summary, sweep_csv)
        df = pd.read_csv(sweep_csv)
        print(f"\n{'CODE':<8} {'ARCH':<10} {'RUN':<16} {'f_E':>6} "
              f"{'Cont':>6} {'Comp':>6} {'Contr':>6}")
        print("-" * 72)
        for _, row in df.sort_values(["code", "sweep_arch", "sweep_run"]).iterrows():
            def _f(c): return f"{row[c]:.3f}" if pd.notna(row.get(c)) else "N/A"
            print(f"{row['code']:<8} {row['sweep_arch']:<10} {row['sweep_run']:<16} "
                  f"{_f('f_E'):>6} {_f('continuity_valid'):>6} "
                  f"{_f('compactness_valid'):>6} {_f('contrastivity_valid'):>6}")
        return

    for code in codes:
        summary = run_code(
            code, tf, shap_lib, pipe_cfg,
            n_samples=n_samples,
            shap_bg=shap_bg,
            run_sanity=not args.skip_sanity,
        )
        append_to_batch_csv(summary, csv_path)

    df = pd.read_csv(csv_path)
    print(f"\n{'CODE':<8} {'f_E':>6} {'Cont':>6} {'Comp':>6} {'Contr':>6} "
          f"{'MPRT_SG':>8} {'MPRT_GC':>8}")
    print("-" * 56)
    for _, row in df.iterrows():
        def _f(c): return f"{row[c]:.3f}" if pd.notna(row.get(c)) else "N/A"
        def _b(c): return ("PASS" if row[c] else "FAIL") if pd.notna(row.get(c)) else "N/A"
        print(f"{row['code']:<8} {_f('f_E'):>6} {_f('continuity_valid'):>6} "
              f"{_f('compactness_valid'):>6} {_f('contrastivity_valid'):>6} "
              f"{_b('sanity_sg_passed'):>8} {_b('sanity_gc_passed'):>8}")


if __name__ == "__main__":
    main()