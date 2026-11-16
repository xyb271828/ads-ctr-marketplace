"""Train and compare click models on the same ordered split."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction import FeatureHasher
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import murmurhash3_32

from ads.calibration import correct_negative_sampling, paired_delta, summarize
from ads.data import NAMES, batches, encode, fit_rarity, prepare


BUCKETS = 2**20


def benchmark(source, output, rows=200_000, seed=42, keep=.25, full=False, epochs=3):
    if not 0 < keep <= 1 or epochs < 1:
        raise ValueError("Expected negative keep in (0, 1] and at least one epoch")
    output = Path(output)
    root = output / "splits"
    manifest = prepare(source, root, rows, seed)
    train_ctr = manifest["windows"]["train"]["ctr"]
    if not 0 < train_ctr < 1:
        raise ValueError("Training window requires both click classes")
    rarity = fit_rarity(root / "train.tsv")
    hasher = FeatureHasher(n_features=BUCKETS, input_type="dict", dtype=np.float32)
    rng = np.random.default_rng(seed)
    models = [SGDClassifier(loss="log_loss", random_state=seed, average=True)
              for _ in range(2)]
    collision_sample = set()
    probability_inputs = {}
    full_inputs = {}
    model_ms = 0.
    train_seconds = 0.
    for split in NAMES:
        n = manifest["windows"][split]["sample_rows"]
        y, raw, down = np.empty(n, np.int8), np.empty(n), np.empty(n)
        present = np.empty(n, bool)
        if full:
            folder = output / "encoded" / split
            folder.mkdir(parents=True, exist_ok=True)
            ids = np.lib.format.open_memmap(folder / "ids.npy", "w+", np.int32, (n, 39))
            values = np.lib.format.open_memmap(folder / "values.npy", "w+", np.float32, (n, 39))
            tree = np.lib.format.open_memmap(folder / "tree.npy", "w+", np.float32, (n, 39))
        start = 0
        for batch in batches(root / f"{split}.tsv"):
            end = start + len(batch)
            features = []
            for row in batch:
                nums, tokens = encode(row, rarity)
                features.append({**{f"n{i}": v for i, v in enumerate(nums)},
                                 **dict.fromkeys(tokens, 1.)})
            x = hasher.transform(features)
            labels = np.array([int(row[0]) for row in batch], dtype=np.int8)
            y[start:end] = labels
            present[start:end] = [bool(row[14]) for row in batch]
            if full:
                keys = np.asarray([[abs(murmurhash3_32(key)) % BUCKETS for key in record]
                                   for record in features], dtype=np.int32)
                vals = np.asarray([list(record.values()) for record in features], np.float32)
                ids[start:end], values[start:end] = keys, vals
                tree[start:end, :13], tree[start:end, 13:] = vals[:, :13], keys[:, 13:]
            tick = perf_counter()
            if split == "train":
                retained = (labels == 1) | (rng.random(len(labels)) < keep)
                models[0].partial_fit(x, labels, classes=[0, 1])
                if retained.any():
                    models[1].partial_fit(x[retained], labels[retained], classes=[0, 1])
                train_seconds += perf_counter()-tick
                for feature_dict in features:
                    if len(collision_sample) >= 100_000:
                        break
                    collision_sample.update(feature_dict)
            else:
                raw[start:end] = models[0].decision_function(x)
                if split == "test":
                    model_ms += (perf_counter()-tick)*1000
                down[start:end] = models[1].decision_function(x)
            start = end
        if full:
            full_inputs[split] = (ids, values, y, tree)
        if split != "train":
            probability_inputs[split] = (y, raw, down, present)
    yv, lv, _, _ = probability_inputs["validation"]
    y, logits, sampled, present = probability_inputs["test"]
    platt = LogisticRegression().fit(lv[:, None], yv)
    iso = IsotonicRegression(out_of_bounds="clip").fit(lv, yv)
    base = expit(logits)
    predictions = {"background": np.full(len(y), train_ctr), "hashed_lr": base,
                   "platt": platt.predict_proba(logits[:, None])[:, 1],
                   "isotonic": iso.predict(logits), "negative_sampled": expit(sampled),
                   "negative_corrected": correct_negative_sampling(expit(sampled), keep)}
    stats = {"hashed_lr": {"test_scoring_ms": model_ms,
                           "batch_size": 8192,
                           "train_rows_per_second_joint_lr": manifest["windows"]["train"]["sample_rows"]
                           / train_seconds, "model_bytes": models[0].coef_.nbytes
                           + models[0].intercept_.nbytes}}
    if full:
        from ads.models import train_deepfm, train_gbdt_lr

        train, valid, test = (full_inputs[name] for name in NAMES)
        predictions["gbdt_lr"], stats["gbdt_lr"] = train_gbdt_lr(train[3], train[2], test[3], seed)
        predictions["deepfm"], stats["deepfm"] = train_deepfm(train[:3], valid[:3], test[:3], seed, epochs)
    result = {}
    for name, p in predictions.items():
        result[name] = {**summarize(y, p, train_ctr), **stats.get(name, {}),
                        "paired_vs_lr": paired_delta(y, base, p, train_ctr, seed),
                        "segments": {key: {"n": int(mask.sum()), "observed": float(y[mask].mean()),
                                           "predicted": float(p[mask].mean())}
                                     for key, mask in (("c0_present", present), ("c0_missing", ~present))
                                     if mask.any()}}
    used = {abs(murmurhash3_32(token)) % BUCKETS for token in collision_sample}
    manifest.update({"negative_keep": keep, "hash_buckets": BUCKETS, "platform": platform.platform(),
                     "collision_sample_distinct_tokens": len(collision_sample),
                     "collision_fraction": 1 - len(used)/len(collision_sample)})
    report = {"data": manifest, "models": result}
    (output / "ctr.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--out", default="runs/ctr")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", type=float, default=.25)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    result = benchmark(args.source, args.out, args.rows, args.seed, args.keep, args.full, args.epochs)
    for name, item in result["models"].items():
        print(name, "log_loss", item["log_loss"], "NE", item["ne"], "AUC", item["auc"])
