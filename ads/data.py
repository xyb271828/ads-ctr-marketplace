"""Prepare chronological Criteo samples without loading the source into RAM."""
import argparse
import csv
import gzip
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from sklearn.utils import murmurhash3_32


NAMES = ("train", "validation", "test")


def read_rows(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        for number, row in enumerate(csv.reader(source, delimiter="\t"), 1):
            if len(row) != 40 or row[0] not in ("0", "1"):
                raise ValueError(f"Row {number}: expected one label and 39 features")
            yield row


def prepare(source, output, size=200_000, seed=42):
    """Split by source row order, then uniformly sample within each window."""
    source, output = Path(source), Path(output)
    total = sum(1 for _ in read_rows(source))
    if not 100 <= size <= total:
        raise ValueError(f"Use a sample size between 100 and {total}")
    boundaries = (0, total * 70 // 100, total * 85 // 100, total)
    sizes = (size * 70 // 100, size * 15 // 100)
    sizes += (size - sum(sizes),)
    rng = np.random.default_rng(seed)
    indices = np.concatenate([
        lo + np.sort(rng.choice(hi-lo, n, replace=False))
        for lo, hi, n in zip(boundaries[:-1], boundaries[1:], sizes, strict=True)
    ])
    ctr = [0, 0, 0]
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        writers = [csv.writer(stack.enter_context((output / f"{name}.tsv").open("w", newline="")),
                              delimiter="\t") for name in NAMES]
        chosen = 0
        for index, row in enumerate(read_rows(source)):
            if index != indices[chosen]:
                continue
            split = 0 if index < boundaries[1] else 1 if index < boundaries[2] else 2
            ctr[split] += int(row[0])
            writers[split].writerow(row)
            chosen += 1
            if chosen == len(indices):
                break
    if chosen != len(indices):
        raise ValueError("Source changed between passes")
    manifest = {"seed": seed, "source": source.name, "source_rows": total, "windows": {
        name: {"start_row_zero_based": boundaries[i], "end_row_exclusive": boundaries[i+1],
               "sample_rows": sizes[i], "ctr": ctr[i] / sizes[i]}
        for i, name in enumerate(NAMES)
    }}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def batches(path, size=8192):
    batch = []
    for row in read_rows(path):
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def fixture(path, size=20000, seed=42):
    """Write a Criteo-shaped synthetic benchmark with a mild temporal shift."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for i in range(size):
            x = rng.integers(0, 10, 13)
            cat = rng.integers(0, 25, 26)
            logit = -2.3 + .28*x[0] + .7*(cat[0] < 4) - .4*(i >= .85*size)
            click = int(rng.random() < 1/(1+np.exp(-logit)))
            writer.writerow([click, *x, *[str(v) if rng.random() > .025 else "" for v in cat]])


def categorical_tokens(row):
    return [f"c{i}={value or '<missing>'}" for i, value in enumerate(row[14:])]


class RarityMap:
    """Train-only approximate counts with fixed memory."""

    def __init__(self, buckets=2**20, threshold=3):
        self.buckets, self.threshold = buckets, threshold
        self.counts = np.zeros((3, buckets), dtype=np.uint8)

    def locations(self, token):
        for row, seed in enumerate((11, 29, 61)):
            yield row, murmurhash3_32(token, seed=seed, positive=True) % self.buckets

    def add(self, token):
        for row, index in self.locations(token):
            if self.counts[row, index] < self.threshold:
                self.counts[row, index] += 1

    def token(self, token, field):
        if token.endswith("=<missing>") or all(
            self.counts[r, i] >= self.threshold for r, i in self.locations(token)
        ):
            return token
        return f"c{field}=<oov>"


def fit_rarity(path):
    rarity = RarityMap()
    for row in read_rows(path):
        for token in categorical_tokens(row):
            rarity.add(token)
    return rarity


def encode(row, rarity):
    values = np.asarray([float(value) if value else 0. for value in row[1:14]])
    numerics = np.sign(values) * np.log1p(np.abs(values))
    tokens = [rarity.token(token, i) for i, token in enumerate(categorical_tokens(row))]
    return numerics, tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_path")
    parser.add_argument("--rows", type=int, default=20000)
    arguments = parser.parse_args()
    fixture(arguments.fixture_path, arguments.rows)
