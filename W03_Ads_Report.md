# Week 3 · Ads CTR and Marketplace

## Experimental design

The pipeline splits the labeled Criteo source into consecutive 70/15/15 row windows before configured uniform sampling. It streams TSV in fixed batches and writes split files to disk. Continuous values use signed log1p; missing numeric values become zero. Field-qualified categories use a train-only count-min sketch and a minimum observed frequency of three. A sketch collision can retain an otherwise rare category. The hash collision diagnostic uses up to approximately 100,000 distinct feature tokens from the training prefix. The hash space uses 2^20 buckets.

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

## Public-data results pending

The labeled original Criteo training file and its row order are required for the 5-million-row benchmark. Public-data scores are pending.
