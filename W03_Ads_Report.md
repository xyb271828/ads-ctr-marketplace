# Week 3 · Ads CTR and Marketplace

## Experimental design

The legacy TSV path splits the labeled source into consecutive 70/15/15 row windows. The Parquet path assigns dates to distinct splits and samples within each split only when requested. It streams TSV in fixed batches and writes split files to disk. Continuous values use signed log1p; missing numeric values become zero. Field-qualified categories use a train-only count-min sketch and a minimum observed frequency of three. A sketch collision can retain an otherwise rare category. The hash collision diagnostic uses up to approximately 100,000 distinct feature tokens from the training prefix. The hash space uses 2^20 buckets.

The candidate models are constant training CTR, hashed logistic regression, LightGBM leaf features followed by logistic regression, and DeepFM. The tree and its leaf LR fit chronologically disjoint portions of the training window. DeepFM selects its checkpoint using validation LogLoss. The test window evaluates all predictions. Normalized entropy divides test LogLoss by the loss of constant training CTR on the same test labels. Contiguous ordered blocks of test rows are sampled jointly for paired LogLoss and NE intervals. AUC measures discrimination, and reliability bins and ECE measure calibration. The code reports observed and predicted CTR for two groups defined by missingness of an anonymous category field.

Platt and isotonic methods fit on validation scores. A separate LR is trained after randomly discarding some negative examples while preserving positives. Its probabilities are corrected by multiplying sampled odds by the negative retention rate. The correction targets the labeled source distribution before this additional sampling.

## Auction replay

The simulation samples 50 advertisers, two slots, ten candidates per request and a 24-hour traffic curve. The CPC reserve is fixed at 0.25. Effective bids are capped by unspent budgets. True synthetic CTR and matched uniform draws generate clicks, with a lower examination probability for the second slot. Three probability variants (oracle, global 10% inflation and segment 10% inflation) run with and without probabilistic pacing. Uniform scaling is expected to leave allocation and CPC unchanged in this design. Segment inflation can affect allocation and price. The experiment reports clicks, spend, CTR, CPC, coverage, early depletion, advertiser exposure, and hourly delivery over three seeds.

## Verified synthetic results

[GitHub Actions run and raw artifacts](https://github.com/xyb271828/ads-ctr-marketplace/actions/runs/43970089759)

On 20,000 generated impressions (3,000 held-out test rows), hashed LR led the model ladder on LogLoss. Platt and isotonic did not improve its LogLoss on this fixture.

| Model or calibration variant | LogLoss | NE | AUC | ECE |
| --- | ---: | ---: | ---: | ---: |
| Training CTR | 0.56796 | 1.00000 | 0.50000 | 0.07121 |
| Hashed LR | 0.52996 | 0.93309 | 0.67762 | 0.05838 |
| Platt | 0.53058 | 0.93420 | 0.67762 | 0.06543 |
| Isotonic | 0.58405 | 1.02833 | 0.67552 | 0.06921 |
| Negative sampling (25% retained) | 0.85033 | 1.49718 | 0.68119 | 0.37422 |
| Negative-prior correction | 0.53650 | 0.94461 | 0.68119 | 0.08128 |
| GBDT + leaf LR | 0.77889 | 1.37138 | 0.62337 | 0.19573 |
| DeepFM | 0.56393 | 0.99290 | 0.64416 | 0.09557 |

Correcting the negative-sampling prior restored much of the probability quality with the same AUC. The fixture cannot support a model choice for real traffic. The raw CTR artifact contains paired uncertainty, batch scoring time and model sizes.

| 18,000 requests/day × three seeds | Mean clicks | Mean platform revenue (synthetic units) | Depleted before hour 18 |
| --- | ---: | ---: | ---: |
| Oracle, no pacing | 562.00 | 332.11 | 98.7% |
| Oracle, paced | 753.33 | 331.03 | 19.3% |
| Uniform +10%, no pacing | 562.00 | 332.11 | 98.7% |
| Uniform +10%, paced | 753.33 | 331.03 | 19.3% |
| Segment +10%, no pacing | 565.67 | 332.56 | 98.7% |
| Segment +10%, paced | 756.33 | 330.78 | 18.0% |

Eight tests passed. The Docker contextual demo served 1,000 requests without errors at 50.04 achieved QPS. On this Ubuntu GitHub-hosted runner, pCTR plus auction took **0.477 ms p99** and the client observed **3.703 ms HTTP p99**. The demo had four fictional sponsors and a fixed request mix, so the result describes this benchmark only.

## Real-data CTR benchmark

[GitHub Actions run and full metrics artifact](https://github.com/xyb271828/ads-ctr-marketplace/actions/runs/43976259862) · [Committed metric summary and shard checksums](reports/criteo_temporal_results.json)

The run used all **3,615,995 impressions** in five public [Criteo 1TB Click Logs](https://huggingface.co/datasets/criteo/CriteoClickLogs) Parquet shards, one shard per day. These shards are a subset of the release. The three model families share the same held-out test rows, with training-only feature counts. Platt and isotonic mappings fit only on February 18. GBDT+LR fits up to 200,000 rows per stage and uses February 17 for leaf LR, after fitting trees on February 15–16. DeepFM ran for three epochs with checkpoint selection on validation LogLoss.

| Split | Dates in February 2015 | Impressions | Observed CTR |
| --- | --- | ---: | ---: |
| Train | 15–17 | 2,315,220 | 3.213% |
| Validation | 18 | 706,671 | 3.170% |
| Test | 19 | 594,104 | 3.461% |

| Model or calibration variant | LogLoss | AUC | NE | ECE |
| --- | ---: | ---: | ---: | ---: |
| Constant training CTR | 0.15050 | 0.50000 | 1.00000 | 0.247 pp |
| Hashed LR | 0.14358 | 0.73690 | 0.95402 | 1.614 pp |
| Hashed LR + Platt | 0.13707 | 0.73690 | 0.91077 | 0.242 pp |
| Hashed LR + isotonic | 0.13717 | 0.73672 | 0.91138 | 0.232 pp |
| Negative-sampled LR | 0.15211 | 0.73738 | 1.01064 | 3.632 pp |
| Negative-sampled LR + prior correction | 0.14244 | 0.73738 | 0.94643 | 1.397 pp |
| GBDT + LR | 0.14120 | 0.70618 | 0.93818 | 0.488 pp |
| DeepFM | 0.13596 | 0.74432 | 0.90339 | 0.090 pp |

- **DeepFM** had the lowest held-out LogLoss and highest AUC. LogLoss improved **5.31%** relative to hashed LR and **9.66%** relative to constant training CTR. The paired change versus hashed LR was −0.00762, with a 95% block-bootstrap interval of [−0.00804, −0.00723].
- **Platt calibration** improved hashed LR LogLoss by **4.53%**, with unchanged AUC and ECE decreasing from 1.614 to 0.242 percentage points. Its LogLoss was 0.00111 above DeepFM. The benchmark reports paired intervals versus hashed LR, so the two leading variants have no direct paired interval in this run.
- **Negative-sampling prior correction** lowered LogLoss from 0.15211 to 0.14244 and ECE from 3.632 to 1.397 percentage points, with unchanged AUC. Calibration conclusions refer to the published dataset’s sampled distribution.
- **CPU batch scoring** took 90.6 ms for raw hashed LR and 368.9 ms for DeepFM across 594,104 already-encoded test rows in this run. These timers measure model batch scoring. Platt calibration and service overhead are outside the hashed LR timer. The results cover one held-out day and one shard per date, so neither the intervals nor the batch timing establish temporal stability or a serving SLA.

## Seven-day conversion extension

The delayed-feedback benchmark adds a timestamp-aware label pipeline using Criteo Attribution. It compares recent incomplete labels, a mature-only cohort, and a likelihood using an empirical delay CDF estimated from mature training positives. Validation labels mature before testing. All methods share features, regularization and optimization. The delay-aware model improved held-out LogLoss by 0.66% versus the recent-label baseline. [Full experimental design, measurements and limits](reports/conversion/README.md).

## Real conversion HTTP deployment

A separate FastAPI scorer now serves all three frozen conversion models using the same feature encoder as training. A 1 CPU / 1 GiB Docker container on an AMD EPYC GitHub runner completed 12,000 paced HTTP requests across 1/100 candidates and 50/200 QPS with no errors or offline prediction mismatches. At 100 candidates and 200 QPS, the delay-aware model achieved 7.32 ms HTTP p99 and 16.40 ms scheduled-to-response p99; scorer-only p99 was 0.77 ms. This is a short loopback workload rather than a sustained production SLA. [Complete protocol, hardware, model version and measurements](reports/serving/README.md).
