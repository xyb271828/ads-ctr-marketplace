# Real conversion model HTTP serving

This service scores **seven-day post-impression conversion probability** using the three frozen models in the [conversion experiment](../conversion/README.md). This is a separate endpoint from the synthetic contextual CTR auction. Conversion scores must not be substituted for click probabilities in a CPC auction.

## Reproduce

```bash
pip install -r requirements-conversion.txt -r requirements-serving.txt
bash scripts/download_conversion.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ads.conversion data/conversion/criteo_attribution_dataset.tsv.gz --output runs/conversion-serving
docker build -f Dockerfile.conversion -t conversion-scoring .
docker run --rm --cpus=1 --memory=1g -p 127.0.0.1:8769:8000 -v "$PWD/runs/conversion-serving/artifacts:/app/models/conversion:ro" conversion-scoring
# From another terminal:
python -m ads.conversion_load_test --fixture runs/conversion-serving/requests.json --output runs/conversion-serving/http-benchmark.json --requests 1000 --qps 50 200
```

`GET /health` exposes the frozen model version and target. `POST /score` accepts `model` (naive_recent, mature_only, delay_likelihood) and 1–200 `records`, each containing nonnegative integer campaign and cat1–cat9 identifiers. Extra fields, missing features and invalid types are rejected. Weights load once on startup; checksum and schema validation fail closed. No training or dataset download occurs during server startup.

## Measurement protocol

- One Uvicorn worker; container CPU quota 1 CPU and memory limit 1 GiB; BLAS/OpenMP threads 1. Client runs outside the container on the same GitHub-hosted VM. CPU quota does not imply a dedicated physical core.
- Three models × 1/100 candidates × 50/200 offered requests per second, 1,000 timed requests each. Twenty warmup requests precede each model/batch pair. Warmup is excluded.
- Paced open-loop arrivals, at most 100 concurrent in-flight requests. Record queue/dispatch delay separately and include it in scheduled-to-response latency. Client scheduling can also limit offered load.
- HTTP latency includes request serialization, validation, feature hashing, inference, response transport and client overhead; server timing includes feature construction and inference only. There is no remote feature-store fetch or wide-area network.
- Each timed response is checked against held-out offline predictions and its artifact version. Export checks all three models on 2,000 held-out impressions. Timed tests replay the first 1 or 100 records; this is a hot-model fixed-payload workload, not a representative traffic mix or capacity ceiling.
- Report p50/p95/p99 and failures. A single short run per setting does not establish an SLA, long-run stability or maximum sustainable QPS. Workflows stop containers after testing; this is a reproducible deployed HTTP benchmark, not a permanently hosted public endpoint.

The GitHub Actions workflow uploads model weights, manifest, parity results, hardware and benchmark JSON as `real-conversion-serving`. Raw source data and request fixtures are not committed or uploaded. Measurements will be recorded here after the container run completes.
