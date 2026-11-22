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

The GitHub Actions workflow uploads model weights, manifest, parity results, hardware and benchmark JSON as `real-conversion-serving`. Raw source data and request fixtures are not committed or uploaded.

## Observed container results

[Successful workflow run](https://github.com/xyb271828/ads-ctr-marketplace/actions/runs/45084310945) on commit `a375008fe74c5130db3b9ca0f9447cdf03924421`. Runner: Ubuntu x86_64, AMD EPYC 9V74, four visible vCPUs; scoring container limited to one CPU quota and 1 GiB. Python 3.11, pinned dependencies. See [hardware](hardware.txt), [model manifest](manifest.json), [export parity](parity.json) and [complete benchmark JSON](http-benchmark.json).

All **12,000 timed requests succeeded**, with maximum absolute online/offline probability difference **0.0** for all three models. Model export independently matched 2,000 held-out predictions per model. The same weights checksum was reproduced in the local export and Actions export.

| Model | Candidates | Offered QPS | Achieved QPS | HTTP p50 / p95 / p99 ms | Scheduled p99 ms | Server p99 ms | Errors |
|---|---:|---:|---:|---|---:|---:|---:|
| naive_recent | 1 | 50 | 50.04 | 1.830 / 2.079 / 2.295 | 4.109 | 0.195 | 0 |
| naive_recent | 1 | 200 | 200.08 | 1.604 / 1.891 / 2.623 | 4.538 | 0.132 | 0 |
| naive_recent | 100 | 50 | 50.04 | 2.741 / 3.275 / 3.902 | 4.958 | 0.749 | 0 |
| naive_recent | 100 | 200 | 200.08 | 2.464 / 2.666 / 4.917 | 8.048 | 0.724 | 0 |
| mature_only | 1 | 50 | 50.04 | 1.835 / 2.292 / 2.529 | 4.323 | 0.213 | 0 |
| mature_only | 1 | 200 | 200.12 | 1.518 / 1.675 / 1.959 | 2.744 | 0.117 | 0 |
| mature_only | 100 | 50 | 50.04 | 2.803 / 3.539 / 3.762 | 4.941 | 0.657 | 0 |
| mature_only | 100 | 200 | 200.07 | 2.449 / 2.579 / 2.830 | 3.623 | 0.524 | 0 |
| delay_likelihood | 1 | 50 | 50.04 | 1.821 / 1.950 / 2.239 | 4.071 | 0.157 | 0 |
| delay_likelihood | 1 | 200 | 200.12 | 1.511 / 1.775 / 2.376 | 3.266 | 0.135 | 0 |
| delay_likelihood | 100 | 50 | 50.04 | 2.843 / 3.783 / 3.910 | 5.045 | 0.644 | 0 |
| delay_likelihood | 100 | 200 | 200.07 | 2.608 / 2.988 / 7.320 | 16.402 | 0.766 | 0 |

The selected delay-aware model at 100 candidates and 200 QPS had HTTP p99 **7.320 ms**, but scheduled-arrival-to-response p99 was **16.402 ms**. These figures support a short loopback HTTP benchmark under the stated load; they do **not** establish a 15 ms end-to-end SLA. The scorer-only p99 was 0.766 ms. The differing timing boundaries expose overhead and scheduling tails, but percentiles cannot be subtracted to assign a cause. Further profiling and longer mixed-payload runs are needed before attributing the tail to a specific component.

All three models use identical sparse linear inference; the delayed-label training objective adds no new inference layer. Differences between their single-run timing measurements should not be interpreted as robust speed rankings. The run demonstrates 200 offered QPS with zero errors in this workload, not maximum service capacity.
