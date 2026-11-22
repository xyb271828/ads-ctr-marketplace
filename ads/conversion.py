"""Seven-day post-impression conversion with as-of delayed-label likelihood."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

from ads.calibration import summarize, pointwise_loss
from ads.conversion_serving import export, transform

DAY = 86400
FEATURES = ['campaign', *[f'cat{i}' for i in range(1, 10)]]
HORIZON = 7 * DAY
SOURCE_URL = ('https://huggingface.co/datasets/criteo/criteo-attribution-dataset/'
              'resolve/904188a63cbad78bee43cd26ff5ee4ac77903986/'
              'criteo_attribution_dataset.tsv.gz')


def observed_labels(timestamp, conversion_time, asof, horizon=HORIZON):
    """A positive must both arrive by asof and fall within the target horizon."""
    elapsed = asof - timestamp
    if np.any(elapsed < 0):
        raise ValueError('Training contains impressions after its label cutoff')
    delay = conversion_time - timestamp
    return ((conversion_time >= 0) & (delay >= 0) & (delay <= horizon)
            & (conversion_time <= asof)).astype(np.int8)


def delay_cdf(ages, mature_delays):
    """P(delay <= age | conversion within H), estimated on mature training rows."""
    if not len(mature_delays):
        raise ValueError('Mature training cohort has no positives')
    ordered = np.sort(mature_delays)
    return np.clip(np.searchsorted(ordered, np.minimum(ages, HORIZON), side='right')
                   / len(ordered), .02, 1.)


def delayed_objective(parameters, x, labels, cdf, regularization=1e-5):
    p = expit(x @ parameters[:-1] + parameters[-1])
    q = np.clip(p * cdf, 1e-12, 1 - 1e-12)
    loss = -np.mean(labels * np.log(q) + (1-labels)*np.log1p(-q))
    residual = (q-labels) * (1-p) / (1-q)
    gradient = np.r_[np.asarray(x.T @ residual).ravel()/len(labels), residual.mean()]
    loss += .5 * regularization * np.dot(parameters[:-1], parameters[:-1])
    gradient[:-1] += regularization * parameters[:-1]
    return float(loss), gradient


def load(path, modulus=10, seed=42):
    """Sample users deterministically across the complete 30-day log."""
    if modulus < 1:
        raise ValueError('user-modulus must be positive')
    columns = ['timestamp', 'uid', 'conversion', 'conversion_timestamp', *FEATURES]
    pieces = []
    total = 0
    end = 0
    for chunk in pd.read_csv(path, sep='\t', compression='infer', usecols=columns,
                             chunksize=500_000):
        total += len(chunk)
        end = max(end, int(chunk.timestamp.max()))
        # Stable integer user sampling avoids consuming future labels for selection.
        hashed = pd.util.hash_pandas_object(chunk.uid + seed, index=False).to_numpy()
        pieces.append(chunk.loc[hashed % modulus == 0])
    frame = pd.concat(pieces, ignore_index=True).sort_values('timestamp', kind='stable')
    invalid = (frame.conversion_timestamp >= 0) & (frame.conversion_timestamp < frame.timestamp)
    if invalid.any():
        raise ValueError('Conversion precedes its impression')
    return frame, total, end


def metrics(y, p, prior):
    result = summarize(y, p, prior)
    result['predicted_conversion_rate'] = result.pop('predicted_ctr')
    result['observed_conversion_rate'] = result.pop('observed_ctr')
    return {**result, 'pr_auc_average_precision': float(average_precision_score(y, p))}


def interval(y, baseline, treatment, times, seed=42):
    delta = pointwise_loss(y, treatment)-pointwise_loss(y, baseline)
    _, group = np.unique(times // 3600, return_inverse=True)
    sums = np.bincount(group, weights=delta)
    counts = np.bincount(group)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(counts), size=(1000, len(counts)))
    estimates = sums[draws].sum(1)/counts[draws].sum(1)
    return {'delta_log_loss': float(delta.mean()), 'ci95': np.quantile(estimates,[.025,.975]).tolist(),
            'resampling_unit': 'hour of impression', 'blocks': len(counts)}


def run(source, output, modulus=10, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    frame, source_rows, end = load(source, modulus, seed)
    t = frame.timestamp.to_numpy()
    conversion_time = frame.conversion_timestamp.to_numpy()
    # Reserve seven complete days after the last test impression to mature negatives.
    test_end = end-HORIZON
    test_start = test_end-2*DAY
    # Validation labels must mature before the first test request.
    cutoff = test_start-HORIZON-DAY
    validation_end = cutoff+DAY
    train = t < cutoff
    valid = (t >= cutoff) & (t < validation_end)
    test = (t >= test_start) & (t < test_end)
    mature = train & (t <= cutoff-HORIZON)
    if min(train.sum(), valid.sum(), test.sum(), mature.sum()) < 100:
        raise ValueError('Insufficient time span or sampled rows')
    y_train = observed_labels(t[train], conversion_time[train], cutoff)
    y_mature = observed_labels(t[mature], conversion_time[mature], cutoff)
    y_valid = observed_labels(t[valid], conversion_time[valid], test_start)
    y_test = observed_labels(t[test], conversion_time[test], end)
    final_train = observed_labels(t[train], conversion_time[train], end)
    prior = float(y_mature.mean())
    if not 0 < prior < 1:
        raise ValueError('Need both classes in mature training')
    records = frame[FEATURES].to_dict('records')
    x = transform(records)
    xt, xv, xe = x[train], x[valid], x[test]
    optimizers = {}
    parameters = {}
    validation_predictions = {}
    test_predictions = {}
    for name, design, labels in [('naive_recent',xt,y_train), ('mature_only',x[mature],y_mature)]:
        model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=.02,
                              alpha=1e-5, random_state=seed, average=True)
        for _ in range(3):
            model.partial_fit(design, labels, classes=[0,1])
        initial = np.r_[model.coef_.ravel(), model.intercept_]
        fitted = minimize(delayed_objective, initial, args=(design, labels, np.ones(len(labels))),
                          jac=True, method='L-BFGS-B', options={'maxiter':150, 'ftol':1e-8})
        parameters[name] = fitted.x.copy()
        optimizers[name] = {'success':bool(fitted.success),'iterations':fitted.nit,
                            'message':str(fitted.message)}
        if name == 'naive_recent':
            naive_parameters = fitted.x.copy()
        validation_predictions[name] = expit(xv @ fitted.x[:-1]+fitted.x[-1])
        test_predictions[name] = expit(xe @ fitted.x[:-1]+fitted.x[-1])
    mature_delay = conversion_time[mature][y_mature == 1]-t[mature][y_mature == 1]
    cdf = delay_cdf(cutoff-t[train], mature_delay)
    initial = naive_parameters
    fit = minimize(delayed_objective, initial, args=(xt,y_train,cdf), jac=True,
                   method='L-BFGS-B', options={'maxiter':150, 'ftol':1e-8})
    if not np.isfinite(fit.fun):
        raise ValueError('Nonfinite delayed likelihood')
    parameters['delay_likelihood'] = fit.x.copy()
    for name, design, predictions in [('validation',xv,validation_predictions),
                                       ('test',xe,test_predictions)]:
        predictions['delay_likelihood'] = expit(design @ fit.x[:-1]+fit.x[-1])
    validation = {name:metrics(y_valid,p,prior) for name,p in validation_predictions.items()}
    selected = min(validation, key=lambda k:validation[k]['log_loss'])
    result = {name:{**metrics(y_test,p,prior), 'paired_vs_naive':interval(
        y_test,test_predictions['naive_recent'],p,t[test],seed)} for name,p in test_predictions.items()}
    missed = (final_train == 1) & (y_train == 0)
    ages = cutoff-t[train]
    age_stats=[]
    for low,high in [(0,1),(1,3),(3,7),(7,1000)]:
        mask=(ages >= low*DAY)&(ages < high*DAY)
        age_stats.append({'age_days':[low,high], 'rows':int(mask.sum()),
                          'eventual_positive_rows':int(final_train[mask].sum()),
                          'hidden_positive_rows':int(missed[mask].sum())})
    waits=[]
    for days in [0,1,3,7]:
        cohort = t < cutoff-days*DAY
        seen = observed_labels(t[cohort],conversion_time[cohort],cutoff)
        eventual = observed_labels(t[cohort],conversion_time[cohort],end)
        waits.append({'wait_days':days, 'training_rows':int(cohort.sum()),
                      'hidden_fraction_of_eventual_positives':float(((eventual==1)&(seen==0)).sum()
                                                                 /max(1,eventual.sum()))})
    with open(source, 'rb') as handle:
        source_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    summary={'dataset':'Criteo Attribution', 'source_url':SOURCE_URL,
             'source_sha256':source_digest,
             'source_rows':source_rows,'sampled_rows':len(frame),'user_modulus':modulus,'seed':seed,
             'target':'conversion within 7 days after impression; row-weighted exposure labels',
             'horizon_seconds':HORIZON, 'cutoff':int(cutoff),'test_start':int(test_start),
             'test_end_exclusive':int(test_end),'observation_end':end,
             'validation_end_exclusive':int(validation_end),
             'selection_label_cutoff':int(test_start),
             'splits':{name:{'rows':int(mask.sum()),'positives':int(labels.sum())}
                       for name,mask,labels in [('train_asof',train,y_train),('train_mature',mature,y_mature),
                                                ('validation',valid,y_valid),('test',test,y_test)]},
             'features':FEATURES,'validation':validation,'selected_on_validation':selected,'test':result,
             'waiting_policy_diagnostics':waits,'age_buckets':age_stats,
             'mature_positive_delay_quantiles_days':dict(zip(['p50','p90','p99'],
                 (np.quantile(mature_delay,[.5,.9,.99])/DAY).tolist())),
             'baseline_optimizers':optimizers,
             'optimizer':{'success':bool(fit.success),'message':str(fit.message),'iterations':fit.nit,
                          'objective':float(fit.fun)},'runtime_seconds':perf_counter()-start,
             'environment':{'python':platform.python_version(),'platform':platform.platform(),
                 'packages':{p:version(p) for p in ('numpy','scipy','scikit-learn','pandas')},
                 'blas_threads':1},
             'fit_settings':{'hash_buckets':2**16,'regularization':1e-5,'maxiter':150,
                             'cdf_floor':.02,'baseline_optimizer':'L-BFGS-B',
                             'delay_model':'empirical global CDF from mature training positives'}}
    test_records = [records[i] for i in np.flatnonzero(test)[:2000]]
    reference = {name: p[:len(test_records)] for name, p in test_predictions.items()}
    export(output/'artifacts', parameters,
           {'source_sha256':source_digest, 'target':summary['target'],
            'selected_model':selected, 'training_cutoff':int(cutoff)}, test_records, reference)
    (output/'requests.json').write_text(json.dumps(
        {'records':test_records, 'expected':{name:p.tolist() for name,p in reference.items()}})+'\n')
    (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(output/'test_predictions.npz',labels=y_test,timestamp=t[test],
                        **test_predictions)
    print(json.dumps({k:summary[k] for k in ['source_rows','sampled_rows','splits',
          'selected_on_validation','test','optimizer','waiting_policy_diagnostics']},indent=2))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,default=Path('runs/conversion'))
    parser.add_argument('--user-modulus',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    run(args.source,args.output,args.user_modulus,args.seed)
