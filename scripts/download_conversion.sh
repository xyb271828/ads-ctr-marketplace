#!/usr/bin/env bash
set -euo pipefail
target="${1:-data/conversion/criteo_attribution_dataset.tsv.gz}"
mkdir -p "$(dirname "$target")"
url="https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/904188a63cbad78bee43cd26ff5ee4ac77903986/criteo_attribution_dataset.tsv.gz"
curl -fL --retry 3 --max-time 600 "$url" -o "$target.partial"
gzip -t "$target.partial"
mv "$target.partial" "$target"
