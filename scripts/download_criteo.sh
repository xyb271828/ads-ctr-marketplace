#!/usr/bin/env bash
set -euo pipefail

target="${1:-data/criteo}"
mkdir -p "$target"
base="https://huggingface.co/datasets/criteo/CriteoClickLogs/resolve/main/data"
suffix="99c339d5-fbac-4110-9dcf-75453a61a5c1.c000.snappy.parquet"
while read -r day part; do
  curl --fail --location --silent --show-error --retry 3 --max-time 120 \
    "$base/day%3D$day/part-$part-$suffix" -o "$target/$day.parquet.partial"
  mv "$target/$day.parquet.partial" "$target/$day.parquet"
done <<'DAYS'
2015-02-15 00079
2015-02-16 00053
2015-02-17 00029
2015-02-18 00017
2015-02-19 00029
DAYS
