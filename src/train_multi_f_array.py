#!/usr/bin/env python3
"""
Hyperparameter sweep for multi-lead ECG psychiatric-vs-normal classification.

Grid dimensions (code × architecture × learning rate × dropout) are read from the
YAML configuration or from datasets/pipeline_config.json after build_datasets.py.

Usage:
  python src/train_multi_f_array.py --config configs/romania.yaml --task-id 0
  sbatch slurm/train_multi_f_array.slurm
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split

from ecg_utils import demographic_match
from mof.paths import dataset_dir, shared_dir
from pipeline_config import PipelineConfig, add_config_arg, load_config_or_snapshot

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
DEFAULT_TRIAL5_EXP = os.environ.get("MOF_MODELS_PARENT", str(_SRC_DIR))

CSV_COLUMNS = [
    "task_id", "psych_code", "architecture", "learning_rate", "dropout_rate",
    "epochs_ran",
    "val_auc",  "val_acc",  "val_precision", "val_recall", "val_specificity",
    "test_auc", "test_acc", "test_precision", "test_recall", "test_specificity",
    "n_train", "n_val", "n_test",
    "results_dir",
]


def load_data(
    psych_code: str,
    pipe_cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    pickle_dir = dataset_dir(pipe_cfg, psych_code)
    shared = shared_dir(pipe_cfg)

    with open(pickle_dir / "recs_psych.pkl", "rb") as f:
        recs_psych = pickle.load(f)
    with open(shared / "recs_normal.pkl", "rb") as f:
        recs_normal = pickle.load(f)

    matched_p, matched_n = demographic_match(
        recs_psych, recs_normal, pipe_cfg.demographic_matching
    )
    all_recs = matched_p + matched_n
    signals = np.array([r["signal"] for r in all_recs], dtype=np.float32)
    labels = np.array([r["label"] for r in all_recs], dtype=np.int64)
    return signals, labels


def specificity_score(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def eval_split(
    model,
    X: np.ndarray,
    y_int: np.ndarray,
    split_name: str,
    results_dir: str,
    psych_code: str,
    arch: str,
    batch_size: int,
    threshold: float,
) -> dict:
    target_names = ["Normal", psych_code]
    proba = model.predict(X, batch_size=batch_size, verbose=0).ravel()
    y_pred = (proba >= threshold).astype(np.int64)

    acc = float(accuracy_score(y_int, y_pred))
    auc = float(roc_auc_score(y_int, proba))
    prec = float(precision_score(y_int, y_pred, zero_division=0))
    rec = float(recall_score(y_int, y_pred, zero_division=0))
    spec = specificity_score(y_int, y_pred)
    ap = float(average_precision_score(y_int, proba))
    cm_arr = confusion_matrix(y_int, y_pred, labels=[0, 1])
    rep = classification_report(
        y_int, y_pred, target_names=target_names, digits=4, zero_division=0
    )

    with open(os.path.join(results_dir, f"report_{split_name}.txt"), "w") as f:
        f.write(rep)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_arr, display_labels=target_names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix — {split_name} ({psych_code} | {arch})")
    plt.tight_layout()
    fig.savefig(
        os.path.join(results_dir, f"confusion_matrix_{split_name}.png"), dpi=150
    )
    plt.close(fig)

    return {
        "auc": round(auc, 4),
        "acc": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "avg_precision": round(ap, 4),
        "n": int(len(y_int)),
    }


def plot_roc_pr(
    model,
    X: np.ndarray,
    y_int: np.ndarray,
    arch: str,
    psych_code: str,
    results_dir: str,
    batch_size: int,
) -> None:
    proba = model.predict(X, batch_size=batch_size, verbose=0).ravel()

    fpr, tpr, _ = roc_curve(y_int, proba)
    auc_val = roc_auc_score(y_int, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — test ({psych_code} | {arch})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "roc_curve_test.png"), dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_int, proba)
    ap_val = average_precision_score(y_int, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"AP = {ap_val:.4f}")
    ax.axhline(
        y_int.mean(), color="k", linestyle="--", alpha=0.4,
        label=f"Baseline = {y_int.mean():.2f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve — test ({psych_code} | {arch})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "pr_curve_test.png"), dpi=150)
    plt.close(fig)

    np.savez_compressed(
        os.path.join(results_dir, "test_predictions.npz"),
        y_true=y_int,
        y_score=proba.astype(np.float32),
    )


def plot_history(history, arch: str, psych_code: str, results_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(
        axes, ["loss", "accuracy", "auc"], ["Loss", "Accuracy", "AUC"]
    ):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="val")
        ax.set_title(f"{title} — {psych_code} | {arch}")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_history.png"), dpi=150)
    plt.close(fig)


def append_csv_row(csv_path: str, row: dict) -> None:
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_one_config(
    task_id: int,
    psych_code: str,
    architecture: str,
    learning_rate: float,
    dropout_rate: float,
    trial5_exp: str,
    job_root: str,
    pipe_cfg: PipelineConfig,
) -> None:
    train_cfg = pipe_cfg.training
    ecg_cfg = pipe_cfg.ecg

    lr_tag = f"lr{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    dr_tag = f"dr{dropout_rate}"
    results_dir = os.path.join(job_root, psych_code, architecture, f"{lr_tag}_{dr_tag}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    metrics_path = os.path.join(results_dir, "metrics.json")
    if os.path.isfile(metrics_path):
        print(f"[task {task_id}] Already done — skipping: {results_dir}")
        return

    print(
        f"[task {task_id}] {psych_code} | {architecture} | "
        f"lr={learning_rate} | dr={dropout_rate}"
    )

    signals_arr, labels_arr = load_data(psych_code, pipe_cfg)
    print(
        f"  Dataset: {signals_arr.shape}  pos={labels_arr.sum()}  "
        f"neg={(labels_arr == 0).sum()}"
    )

    X_all = np.ascontiguousarray(
        np.transpose(signals_arr, (0, 2, 1)), dtype=np.float32
    )
    y_int = labels_arr.astype(np.int64)
    y_all = y_int.astype(np.float32).reshape(-1, 1)

    X_trv, X_test, y_trv, y_test, yi_trv, yi_test = train_test_split(
        X_all, y_all, y_int,
        test_size=0.2, stratify=y_int, random_state=train_cfg.random_state,
    )
    X_train, X_val, y_train, y_val, yi_train, yi_val = train_test_split(
        X_trv, y_trv, yi_trv,
        test_size=0.25, stratify=yi_trv, random_state=train_cfg.random_state,
    )
    print(
        f"  Split — train:{len(y_train)}  val:{len(y_val)}  test:{len(y_test)}"
        f"  | train pos={yi_train.sum()}  neg={(yi_train == 0).sum()}"
    )

    sys.path.insert(0, trial5_exp)
    from models.model_factory import create_model  # noqa: E402

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(train_cfg.random_state)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    model = create_model(
        architecture,
        input_shape=(ecg_cfg.signal_len, ecg_cfg.n_leads),
        num_classes=1,
        dropout_rate=dropout_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=10, restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, "best_val_auc.weights.h5"),
            monitor="val_auc", mode="max",
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(results_dir, "training_log.csv")
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save_weights(os.path.join(results_dir, "final.weights.h5"))
    plot_history(history, architecture, psych_code, results_dir)

    val_metrics = eval_split(
        model, X_val, yi_val, "val", results_dir, psych_code, architecture,
        train_cfg.batch_size, train_cfg.threshold,
    )
    test_metrics = eval_split(
        model, X_test, yi_test, "test", results_dir, psych_code, architecture,
        train_cfg.batch_size, train_cfg.threshold,
    )
    plot_roc_pr(
        model, X_test, yi_test, architecture, psych_code, results_dir,
        train_cfg.batch_size,
    )

    metrics = {
        "task_id": task_id,
        "psych_code": psych_code,
        "architecture": architecture,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "batch_size": train_cfg.batch_size,
        "epochs_ran": len(history.history["loss"]),
        "threshold": train_cfg.threshold,
        "random_state": train_cfg.random_state,
        "n_leads": ecg_cfg.n_leads,
        "signal_len": ecg_cfg.signal_len,
        "splits": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(yi_test)),
        },
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    append_csv_row(
        os.path.join(job_root, "all_runs.csv"),
        {
            "task_id": task_id,
            "psych_code": psych_code,
            "architecture": architecture,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "epochs_ran": len(history.history["loss"]),
            "val_auc": val_metrics["auc"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "test_auc": test_metrics["auc"],
            "test_acc": test_metrics["acc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_specificity": test_metrics["specificity"],
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(yi_test)),
            "results_dir": results_dir,
        },
    )

    print(
        f"  [done] val_auc={val_metrics['auc']:.4f}  "
        f"test_auc={test_metrics['auc']:.4f}  "
        f"epochs={len(history.history['loss'])}"
    )
    print(f"  Artifacts: {results_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_arg(p)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument(
        "--data-root", type=str, default=None,
        help="Override output_root from config.",
    )
    p.add_argument(
        "--trial5-exp", type=str, default=DEFAULT_TRIAL5_EXP,
        help="Directory containing the models package.",
    )
    p.add_argument(
        "--job-root", type=str, default=None,
        help="Sweep output root (default: <data-root>/hyper_sweep).",
    )
    p.add_argument("--print-grid", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    pipe_cfg = load_config_or_snapshot(args.config, data_root=data_root)
    if data_root:
        pipe_cfg.output_root = data_root
    data_root = data_root or pipe_cfg.output_root
    job_root = args.job_root or os.path.join(data_root, "hyper_sweep")

    if pipe_cfg.data.mode == "prebuilt":
        print(f"Dataset source (prebuilt): {pipe_cfg.data.prebuilt_root}")
    else:
        print(f"Dataset source (build): {data_root}")
    print(f"Artifacts root: {data_root}")

    if args.print_grid:
        grid = pipe_cfg.build_sweep_grid()
        print(f"Total configs: {len(grid)}")
        print(f"{'ID':>4}  {'code':<10} {'arch':<10} {'lr':>8} {'dr':>5}")
        print("-" * 48)
        for i, row in enumerate(grid):
            print(
                f"{i:>4}  {row['psych_code']:<10} {row['architecture']:<10} "
                f"{row['learning_rate']:>8.0e} {row['dropout_rate']:>5}"
            )
        return

    task_id = args.task_id
    if task_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is None:
            raise SystemExit("Provide --task-id or set SLURM_ARRAY_TASK_ID.")
        task_id = int(env_id)

    task = pipe_cfg.resolve_task(task_id)
    print(f"Resolved task {task_id}: {task}")

    run_one_config(
        task_id=task["task_id"],
        psych_code=task["psych_code"],
        architecture=task["architecture"],
        learning_rate=task["learning_rate"],
        dropout_rate=task["dropout_rate"],
        trial5_exp=args.trial5_exp,
        job_root=job_root,
        pipe_cfg=pipe_cfg,
    )


if __name__ == "__main__":
    main()
