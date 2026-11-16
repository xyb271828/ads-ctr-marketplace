# Week 3 · Ads CTR and Marketplace

## Experimental design

The pipeline splits the labeled Criteo source into consecutive 70/15/15 row windows before configured uniform sampling. It streams TSV in fixed batches and writes split files to disk. Continuous values use signed log1p; missing numeric values become zero. Field-qualified categories use a train-only count-min sketch and a minimum observed frequency of three. A sketch collision can retain an otherwise rare category. The hash collision diagnostic uses up to approximately 100,000 distinct feature tokens from the training prefix. The hash space uses 2^20 buckets.

The candidate models are constant training CTR, hashed logistic regression, LightGBM leaf features followed by logistic regression, and DeepFM. The tree and its leaf LR fit chronologically disjoint portions of the training window. DeepFM selects its checkpoint using validation LogLoss. The test window evaluates all predictions. Normalized entropy divides test LogLoss by the loss of constant training CTR on the same test labels. Contiguous ordered blocks of test rows are sampled jointly for paired LogLoss and NE intervals. AUC measures discrimination, and reliability bins and ECE measure calibration. The code reports observed and predicted CTR for two groups defined by missingness of an anonymous category field.

Platt and isotonic methods fit on validation scores. A separate LR is trained after randomly discarding some negative examples while preserving positives. Its probabilities are corrected by multiplying sampled odds by the negative retention rate. The correction targets the labeled source distribution before this additional sampling.

## Auction replay

The simulation samples 50 advertisers, two slots, ten candidates per request and a 24-hour traffic curve. The CPC reserve is fixed at 0.25. Effective bids are capped by unspent budgets. True synthetic CTR and matched uniform draws generate clicks, with a lower examination probability for the second slot. Three probability variants (oracle, global 10% inflation and segment 10% inflation) run with and without probabilistic pacing. Uniform scaling is expected to leave allocation and CPC unchanged in this design. Segment inflation can affect allocation and price. The experiment reports clicks, spend, CTR, CPC, coverage, early depletion, advertiser exposure, and hourly delivery over three seeds.

## Results status

The tests and workflow use synthetic fixtures. The held-out public Criteo benchmark requires the original labeled training file. Quantitative public-data conclusions and the deployment p99 budget result remain pending.
