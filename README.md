# Ads CTR and Marketplace

Week 3 of the discovery systems project. This repository compares click prediction models and calibration on an ordered dataset, then measures the effects of pCTR bias in a separate auction replay. The contextual HTTP demo uses generated topic and grade features.

## Verified synthetic results

| Check | Result |
| --- | --- |
| Hashed LR versus fixed training CTR | LogLoss 0.5300 versus 0.5680 on 3,000 held-out synthetic impressions |
| Negative-sampling correction | LogLoss 0.8503 to 0.5365 at unchanged AUC 0.6812 |
| Paced versus unpaced auction replay | Mean clicks 562.0 to 753.3 and early budget depletion 98.7% to 19.3% across three seeds |
| Contextual Docker service | 0.477 ms scorer-plus-auction p99 and 3.703 ms HTTP p99, with 1,000 requests at 50.04 achieved QPS and zero errors |

These results use generated data and the fictional four-sponsor service catalog. See the [report](W03_Ads_Report.md) and [verified workflow](https://github.com/xyb271828/ads-ctr-marketplace/actions/runs/43970089759). The public Criteo run is pending.

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

Obtain the original labeled [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data) training file and keep its original row order. The benchmark splits the labeled source by row ordinal and then samples each window. This is a row-order proxy for time. The Kaggle test file without labels is unused.

```sh
python -m ads.ctr data/criteo/train.txt --out runs/criteo --rows 5000000 --full
```

Run a 200,000-row development benchmark first. Tree fitting and leaf LR fitting each use up to 200,000 rows from disjoint training windows. Results include observed stage-specific fit counts, AUC, LogLoss, normalized entropy, ECE and paired intervals for LogLoss and NE.

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
