"""NumPy-only neural network utilities for Allocation Multiple Model.

This module intentionally avoids sklearn/tensorflow/torch so the trained artifacts
can be loaded in a lightweight Streamlit environment.  It supports dense MLPs for
binary classification, regression/sizing, rank scoring, and multi-output auxiliary
heads.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

EPS = 1e-8


def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.0)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def activation_forward(x, name: str):
    if name == "relu":
        return relu(x)
    if name == "gelu":
        return gelu(x)
    if name == "tanh":
        return np.tanh(x)
    return x


def activation_backward(pre, grad, name: str):
    if name == "relu":
        return grad * (pre > 0)
    if name == "gelu":
        # Smooth numerical derivative good enough for this compact trainer.
        x = pre
        tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        t = np.tanh(tanh_arg)
        sech2 = 1 - t * t
        deriv = 0.5 * (1 + t) + 0.5 * x * sech2 * np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x * x)
        return grad * deriv
    if name == "tanh":
        y = np.tanh(pre)
        return grad * (1 - y * y)
    return grad


@dataclass
class NNSpec:
    name: str
    task: str = "regression"  # binary, regression, rank, auxiliary
    hidden: Tuple[int, ...] = (512, 256, 128)
    lr: float = 2e-4
    epochs: int = 120
    batch_size: int = 512
    dropout: float = 0.05
    activation: str = "gelu"
    weight_decay: float = 1e-5
    patience: int = 30
    seed: int = 42
    output_dim: int = 1
    print_every: int = 5
    validation_frac: float = 0.16


class Standardizer:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray):
        x = _as_float32(x)
        self.mean_ = np.nanmean(x, axis=0).astype(np.float32)
        self.std_ = np.nanstd(x, axis=0).astype(np.float32)
        self.std_[~np.isfinite(self.std_) | (self.std_ < 1e-6)] = 1.0
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        return self

    def transform(self, x: np.ndarray):
        x = _as_float32(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def state(self):
        return {"mean": self.mean_, "std": self.std_}

    @classmethod
    def from_state(cls, state):
        s = cls()
        s.mean_ = state["mean"].astype(np.float32)
        s.std_ = state["std"].astype(np.float32)
        return s


class NumpyMLP:
    def __init__(self, spec: NNSpec, input_dim: int):
        self.spec = spec
        self.input_dim = int(input_dim)
        self.rng = np.random.default_rng(spec.seed)
        dims = [self.input_dim, *spec.hidden, spec.output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / max(fan_in, 1))
            self.weights.append((self.rng.normal(0, scale, size=(fan_in, fan_out))).astype(np.float32))
            self.biases.append(np.zeros((fan_out,), dtype=np.float32))
        self.scaler = Standardizer()
        self.history: List[Dict] = []

    def _forward(self, x, train=False):
        activations = [x]
        preacts = []
        masks = []
        h = x
        for i in range(len(self.weights) - 1):
            z = h @ self.weights[i] + self.biases[i]
            preacts.append(z)
            h = activation_forward(z, self.spec.activation)
            if train and self.spec.dropout > 0:
                mask = (self.rng.random(h.shape) >= self.spec.dropout).astype(np.float32) / (1.0 - self.spec.dropout)
                h = h * mask
            else:
                mask = np.ones_like(h, dtype=np.float32)
            masks.append(mask)
            activations.append(h)
        out = h @ self.weights[-1] + self.biases[-1]
        preacts.append(out)
        activations.append(out)
        return out, activations, preacts, masks

    def predict_raw(self, x):
        xs = self.scaler.transform(x)
        out, *_ = self._forward(xs, train=False)
        return out.astype(np.float32)

    def predict(self, x):
        raw = self.predict_raw(x)
        if self.spec.task == "binary":
            return sigmoid(raw).reshape(-1)
        return raw.reshape((raw.shape[0], -1))

    def _loss_grad(self, pred, y):
        n = max(len(y), 1)
        y = y.astype(np.float32)
        if self.spec.task == "binary":
            p = sigmoid(pred)
            loss = -np.mean(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS))
            grad = (p - y) / n
            return float(loss), grad.astype(np.float32)
        # Huber regression for sizing/ranking/auxiliary.
        diff = pred - y
        delta = 1.0
        absdiff = np.abs(diff)
        loss = np.mean(np.where(absdiff <= delta, 0.5 * diff ** 2, delta * (absdiff - 0.5 * delta)))
        grad = np.where(absdiff <= delta, diff, delta * np.sign(diff)) / n
        return float(loss), grad.astype(np.float32)

    def fit(self, x, y, sample_weight=None, verbose=True):
        x = _as_float32(x)
        y = _as_float32(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n = len(x)
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        rng = np.random.default_rng(self.spec.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_n = max(1, int(n * self.spec.validation_frac)) if n >= 20 else max(1, n // 5)
        val_idx = idx[:val_n]
        train_idx = idx[val_n:] if val_n < n else idx
        xtr, ytr = x[train_idx], y[train_idx]
        xva, yva = x[val_idx], y[val_idx]
        if verbose:
            print(f"[{self.spec.name}] train_rows={len(xtr):,} val_rows={len(xva):,} batch_size={self.spec.batch_size} validation_frac={self.spec.validation_frac}", flush=True)

        mw = [np.zeros_like(w) for w in self.weights]
        vw = [np.zeros_like(w) for w in self.weights]
        mb = [np.zeros_like(b) for b in self.biases]
        vb = [np.zeros_like(b) for b in self.biases]
        beta1, beta2 = 0.9, 0.999
        best_loss = float("inf")
        best_state = None
        bad = 0
        step = 0

        for epoch in range(1, self.spec.epochs + 1):
            order = rng.permutation(len(xtr))
            batch_losses = []
            for start in range(0, len(order), self.spec.batch_size):
                batch = order[start:start + self.spec.batch_size]
                xb, yb = xtr[batch], ytr[batch]
                pred, acts, pres, masks = self._forward(xb, train=True)
                loss, grad = self._loss_grad(pred, yb)
                batch_losses.append(loss)

                grad_w = [None] * len(self.weights)
                grad_b = [None] * len(self.biases)
                g = grad
                for layer in reversed(range(len(self.weights))):
                    a_prev = acts[layer]
                    grad_w[layer] = a_prev.T @ g + self.spec.weight_decay * self.weights[layer]
                    grad_b[layer] = np.sum(g, axis=0)
                    if layer > 0:
                        g = g @ self.weights[layer].T
                        g = g * masks[layer - 1]
                        g = activation_backward(pres[layer - 1], g, self.spec.activation)

                step += 1
                for i in range(len(self.weights)):
                    mw[i] = beta1 * mw[i] + (1 - beta1) * grad_w[i]
                    vw[i] = beta2 * vw[i] + (1 - beta2) * (grad_w[i] ** 2)
                    mb[i] = beta1 * mb[i] + (1 - beta1) * grad_b[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * (grad_b[i] ** 2)
                    mw_hat = mw[i] / (1 - beta1 ** step)
                    vw_hat = vw[i] / (1 - beta2 ** step)
                    mb_hat = mb[i] / (1 - beta1 ** step)
                    vb_hat = vb[i] / (1 - beta2 ** step)
                    self.weights[i] -= self.spec.lr * mw_hat / (np.sqrt(vw_hat) + EPS)
                    self.biases[i] -= self.spec.lr * mb_hat / (np.sqrt(vb_hat) + EPS)

            train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
            val_raw, *_ = self._forward(xva, train=False)
            val_loss, _ = self._loss_grad(val_raw, yva)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            if self.spec.task == "binary":
                p = sigmoid(val_raw).reshape(-1)
                ybin = yva.reshape(-1) > 0.5
                pred_bin = p >= 0.5
                tp = int(np.sum(pred_bin & ybin)); fp = int(np.sum(pred_bin & ~ybin)); fn = int(np.sum(~pred_bin & ybin))
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, EPS)
                row["val_acc"] = float(np.mean(pred_bin == ybin))
                row["val_precision"] = float(precision)
                row["val_recall"] = float(recall)
                row["val_f1"] = float(f1)
                row["val_positive_rate"] = float(np.mean(pred_bin))
            else:
                predv = val_raw.reshape(yva.shape)
                row["val_mae"] = float(np.mean(np.abs(predv - yva)))
            self.history.append(row)

            if verbose and (epoch == 1 or epoch % self.spec.print_every == 0 or epoch == self.spec.epochs):
                msg = f"[{self.spec.name}] epoch {epoch:04d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
                if "val_acc" in row:
                    msg += f" val_acc={row['val_acc']:.4f} val_f1={row.get('val_f1', 0):.4f} val_precision={row.get('val_precision', 0):.4f} val_recall={row.get('val_recall', 0):.4f}"
                if "val_mae" in row:
                    msg += f" val_mae={row['val_mae']:.4f}"
                msg += f" best_val={best_loss if np.isfinite(best_loss) else val_loss:.5f} bad_epochs={bad}/{self.spec.patience}"
                print(msg, flush=True)

            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
                bad = 0
            else:
                bad += 1
                if bad >= self.spec.patience:
                    if verbose:
                        print(f"[{self.spec.name}] early stop at epoch {epoch}; best_val_loss={best_loss:.5f}", flush=True)
                    break
        if best_state is not None:
            self.weights, self.biases = best_state
        return self

    def save(self, path: str | Path, extra: Optional[Dict] = None):
        path = Path(path)
        arrays = {}
        for i, w in enumerate(self.weights):
            arrays[f"W{i}"] = w.astype(np.float32)
        for i, b in enumerate(self.biases):
            arrays[f"b{i}"] = b.astype(np.float32)
        arrays["scaler_mean"] = self.scaler.mean_.astype(np.float32)
        arrays["scaler_std"] = self.scaler.std_.astype(np.float32)
        meta = {"spec": asdict(self.spec), "input_dim": self.input_dim, "history": self.history, "extra": extra or {}}
        arrays["metadata_json"] = np.array(json.dumps(meta), dtype=object)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["metadata_json"].item()))
        spec = NNSpec(**meta["spec"])
        model = cls(spec, int(meta["input_dim"]))
        weights, biases = [], []
        i = 0
        while f"W{i}" in z:
            weights.append(z[f"W{i}"].astype(np.float32))
            biases.append(z[f"b{i}"].astype(np.float32))
            i += 1
        model.weights = weights
        model.biases = biases
        model.scaler = Standardizer.from_state({"mean": z["scaler_mean"], "std": z["scaler_std"]})
        model.history = meta.get("history", [])
        return model


def make_meta_features(x: np.ndarray, classifier_probs=None, rank_scores=None, aux_outputs=None) -> np.ndarray:
    parts = [_as_float32(x)]
    for arr in [classifier_probs, rank_scores, aux_outputs]:
        if arr is None:
            continue
        a = _as_float32(arr)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        parts.append(a)
    return np.concatenate(parts, axis=1).astype(np.float32)


def binary_metrics(y_true, prob, threshold=0.5) -> Dict[str, float]:
    y = np.asarray(y_true).reshape(-1) > 0.5
    p = np.asarray(prob).reshape(-1)
    pred = p >= threshold
    tp = int(np.sum(pred & y)); tn = int(np.sum(~pred & ~y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    brier = float(np.mean((p - y.astype(float)) ** 2))
    return {"accuracy": float(np.mean(pred == y)), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "brier": brier}


def regression_metrics(y_true, pred) -> Dict[str, float]:
    y = np.asarray(y_true).reshape(-1).astype(float)
    p = np.asarray(pred).reshape(-1).astype(float)
    return {"mae": float(np.mean(np.abs(p - y))), "rmse": float(np.sqrt(np.mean((p - y) ** 2))), "bias": float(np.sum(p - y)), "rows": int(len(y))}
