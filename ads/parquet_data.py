"""Build day-disjoint splits from Criteo Click Logs Parquet shards."""
import csv
import hashlib
import json
import re
from datetime import date

import numpy as np

from ads.data import NAMES


COLUMNS = ["label", *[f"integer_feature_{i}" for i in range(1, 14)],
           *[f"categorical_feature_{i}" for i in range(1, 27)]]
DATE_PREFIX = re.compile(r"(\d{4}-\d{2}-\d{2})(?:-.+)?\.parquet\Z")


def prepare_parquet(source, output, size, seed):
    """Sample each split by row, while keeping dates strictly disjoint."""
    import pyarrow.parquet as pq

    shards = {}
    for path in sorted(source.glob("*.parquet")):
        match = DATE_PREFIX.fullmatch(path.name)
        if not match:
            raise ValueError(f"Name the shard YYYY-MM-DD[<dash>suffix].parquet: {path.name}")
        day = date.fromisoformat(match[1]).isoformat()
        parquet = pq.ParquetFile(path)
        if not set(COLUMNS) <= set(parquet.schema_arrow.names):
            raise ValueError(f"{path.name} does not match the Criteo Click Logs schema")
        if not parquet.metadata.num_rows:
            raise ValueError(f"{path.name} is empty")
        shards.setdefault(day, []).append((path, parquet.metadata.num_rows))
    days = sorted(shards)
    if len(days) < 3:
        raise ValueError("Expected at least three dated shards for train, validation and test")
    groups = (days[:-2], days[-2:-1], days[-1:])
    available = [sum(n for day in group for _, n in shards[day]) for group in groups]
    total = sum(available)
    if size is None or size == 0:
        size = total
    if not 100 <= size <= total:
        raise ValueError(f"Expected between 100 and {total} rows or 0 to use all")
    sizes = [size * n // total for n in available]
    for i in sorted(range(3), key=lambda i: -(size * available[i] % total))[:size-sum(sizes)]:
        sizes[i] += 1
    if min(sizes) < 2:
        raise ValueError("Increase --rows so that every date split has at least two rows")

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    windows = {}
    for split, group, count, capacity in zip(NAMES, groups, sizes, available, strict=True):
        chosen = None if count == capacity else np.sort(rng.choice(capacity, count, replace=False))
        offset = clicks = written = 0
        details = []
        day_counts = {}
        print(f"{split}: {', '.join(group)} · {count:,}/{capacity:,} rows", flush=True)
        with (output / f"{split}.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            for day in group:
                day_start = written
                for path, n in shards[day]:
                    shard_start = written
                    parquet = pq.ParquetFile(path)
                    for batch in parquet.iter_batches(batch_size=65536, columns=COLUMNS):
                        stop = offset + batch.num_rows
                        if chosen is not None:
                            lo, hi = chosen.searchsorted([offset, stop])
                            batch = batch.take(chosen[lo:hi] - offset)
                        offset = stop
                        if not batch.num_rows:
                            continue
                        columns = [batch.column(batch.schema.get_field_index(name)).to_pylist()
                                   for name in COLUMNS]
                        if any(label not in (0, 1) for label in columns[0]):
                            raise ValueError(f"Invalid click label in {path.name}")
                        clicks += sum(columns[0])
                        written += batch.num_rows
                        writer.writerows(zip(*columns, strict=True))
                    with path.open("rb") as raw:
                        digest = hashlib.file_digest(raw, "sha256").hexdigest()
                    details.append({"date": day, "file": path.name, "source_rows": n,
                                    "sample_rows": written-shard_start, "sha256": digest})
                day_counts[day] = written-day_start
        if offset != capacity or written != count:
            raise ValueError(f"{split}: source row count changed during reading")
        windows[split] = {"dates": group, "sample_rows": written, "source_rows": capacity,
                          "rows_by_day": day_counts, "ctr": clicks/written, "shards": details}
    manifest = {"source": "Criteo 1TB Click Logs", "split_policy": "disjoint calendar days",
                "seed": seed, "source_rows": total, "windows": windows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
