"""CTR probability evaluation and correction."""
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def correct_negative_sampling(proba, keep):
    if not 0 < keep <= 1:
        raise ValueError("Expected negative retention in (0, 1]")
    return keep * proba / (1 - proba + keep * proba)


def pointwise_loss(labels, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1-1e-12)
    return -np.where(labels, np.log(p), np.log1p(-p))


def baseline_loss(labels, train_ctr):
    return float(pointwise_loss(labels, train_ctr).mean())


def summarize(labels, proba, train_ctr, bins=10):
    loss = float(log_loss(labels, np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1-1e-12), labels=[0, 1]))
    ids = np.minimum((proba * bins).astype(int), bins - 1)
    reliability = []
    for i in range(bins):
        mask = ids == i
        if mask.any():
            reliability.append({"bin": i, "n": int(mask.sum()),
                                "predicted": float(proba[mask].mean()),
                                "observed": float(labels[mask].mean())})
    return {"log_loss": loss, "ne": loss / baseline_loss(labels, train_ctr),
            "auc": float(roc_auc_score(labels, proba)) if np.unique(labels).size == 2 else None,
            "ece": sum(r["n"] * abs(r["predicted"]-r["observed"]) for r in reliability) / len(labels),
            "predicted_ctr": float(proba.mean()), "observed_ctr": float(labels.mean()),
            "reliability": reliability}


def paired_delta(y, base, treatment, train_ctr, seed=42, block=256, attempts=1000):
    """Resample contiguous ordered test blocks in matched model pairs."""
    delta = pointwise_loss(y, treatment) - pointwise_loss(y, base)
    starts = np.arange(0, len(y), block)
    if len(starts) < 2:
        return None
    sums = np.add.reduceat(delta, starts)
    counts = np.diff(np.r_[starts, len(y)])
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(starts), size=(attempts, len(starts)))
    draws = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    ci = np.quantile(draws, [.025, .975])
    denom = baseline_loss(y, train_ctr)
    return {"log_loss": float(delta.mean()), "log_loss_ci95": ci.tolist(),
            "ne": float(delta.mean()/denom), "ne_ci95": (ci/denom).tolist(), "block_rows": block}
