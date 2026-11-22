# Ads CTR and Marketplace

Week 3 of the discovery systems project. This repository compares click prediction models and calibration on an ordered dataset, then measures the effects of pCTR bias in a separate auction replay. The contextual HTTP demo uses generated topic and grade features.

## Verified synthetic results

| Check | Result |
| --- | --- |
| Hashed LR versus fixed training CTR | LogLoss 0.5300 versus 0.5680 on 3,000 held-out synthetic impressions |
| Negative-sampling correction | LogLoss 0.8503 to 0.5365 at unchanged AUC 0.6812 |
| Paced versus unpaced auction replay | Mean clicks 562.0 to 753.3 and early budget depletion 98.7% to 19.3% across three seeds |
| Contextual Docker service | 0.477 ms scorer-plus-auction p99 and 3.703 ms HTTP p99, with 1,000 requests at 50.04 achieved QPS and zero errors |

These results use generated data and the fictional four-sponsor service catalog. See the [report](W03_Ads_Report.md) and [verified workflow](https://github.com/xyb271828/ads-ctr-marketplace/actions/runs/43970089759).

## Reproduce with generated data

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-models.txt
python -m unittest discover -s tests -v
python -m ads.data runs/input.tsv --rows 20000
python -m ads.ctr runs/input.tsv --out runs/synthetic --rows 20000 --full --epochs 2
python -m ads.marketplace > runs/marketplace.json
```

Pass `--full` to run GBDT+LR and DeepFM; the hashed LR and all calibration experiments also run with `requirements.txt` alone. To open the notebooks, install JupyterLab and Matplotlib in the same environment.

## Public CTR benchmark

On **3,615,995 real Criteo impressions**, DeepFM reduced held-out LogLoss from **0.14358 to 0.13596 (5.31%)** versus hashed LR. AUC increased from **0.73690 to 0.74432**. Platt-calibrated hashed LR reached **0.13707** LogLoss. The final test had **594,104 impressions** from a later calendar day. [Full analysis](W03_Ads_Report.md#real-data-ctr-benchmark) · [Verified run](https://github.com/xyb271828/ads-ctr-marketplace/actions/runs/43976259862)

Use dated shards from [Criteo 1TB Click Logs](https://huggingface.co/datasets/criteo/CriteoClickLogs). The downloader selects the five shards for the reproducible evaluation. If you already have the files, pass their directory directly.

```sh
pip install -r requirements-criteo.txt
bash scripts/download_criteo.sh
python -m ads.ctr data/criteo --out runs/criteo --full
```

The five dates are February 15–19, 2015. The first three are training, the fourth is validation, and the fifth is test. Each file is named `YYYY-MM-DD.parquet`, with an optional dash and shard suffix for multiple files per day. The benchmark uses every row in the selected files by default. Add `--rows 200000` for a development sample. The manifest records the dates, actual counts, click rates and SHA-256 of each input shard.

The original ordered TSV input remains supported. Paired intervals are relative to hashed LR on the same test impressions. GBDT+LR fits up to 200,000 rows per stage and uses the latest training day to fit the leaf LR for this five-day evaluation.

## Serve contextual ads

```sh
docker build -t contextual-ads .
docker run --rm -p 127.0.0.1:8000:8000 contextual-ads
```

From another terminal, activate the environment and run:

```sh
curl -fsS -H 'Content-Type: application/json' \
  -d '{"topic":"fractions","grade":5}' http://localhost:8000/auction
mkdir -p runs
python -m ads.load_test --qps 50 --requests 1000 > runs/latency.json
```

A matching fictional sponsor can win at most one slot. The API accepts only topic, grade and relevance threshold. It can return an empty sponsored slot. The pCTR is calibrated on generated examples, with a feature schema designed for the contextual demo. The load test reports HTTP and scorer-plus-auction percentiles independently.

[CTR notebook](W03_CTR.ipynb) · [Auction notebook](W03_Marketplace.ipynb) · [Technical report](W03_Ads_Report.md) · [Engineering log](weekly/Wk-03-EngLog.md)

Week 1 [Search](https://github.com/xyb271828/SciFact_Search) and Week 2 [Recommendations](https://github.com/xyb271828/fari-recsys) run independently.

## Conversion and delayed feedback

A second real-data experiment uses Criteo Attribution timestamps to replay seven-day conversion labels at a training cutoff. On 103,581 later impressions, a delay-aware logistic likelihood reduced LogLoss by **0.66%** versus an otherwise comparable recent-label model. ECE decreased from **0.732 to 0.417 percentage points**. At the training cutoff, **11.66%** of eventual positive rows were still unobserved. Waiting seven days retained about half the training impressions.

[Methods, source, charts and limitations](reports/conversion/README.md) · [Machine-readable results](reports/conversion/results.json)

## Real-model HTTP deployment

The three real conversion models have a versioned `/score` service with shared training/serving feature hashing, artifact integrity checks and offline prediction parity. [Deployment and load-test protocol](reports/serving/README.md). Reproduce the CPU container benchmark through the **Real conversion HTTP benchmark** GitHub Actions workflow.

Measured on an AMD EPYC GitHub runner with a **1 CPU / 1 GiB container**: the delay-aware model scored 100 candidates per request at 200 offered QPS with **7.32 ms HTTP p99** (16.40 ms scheduled-to-response p99). Across three models and 12 workload settings, **12,000/12,000 requests succeeded** with identical offline/online predictions. This is a short loopback benchmark; [full timing boundaries and results](reports/serving/README.md).
