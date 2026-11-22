"""Render the waiting-policy and held-out calibration diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report=json.loads((run_dir/'results.json').read_text())
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    rows=report['waiting_policy_diagnostics']
    ax.plot([r['wait_days'] for r in rows],
            [100*r['hidden_fraction_of_eventual_positives'] for r in rows],'o-',color='#315b91')
    ax.set(xlabel='Wait before including impressions (days)',
           ylabel='Future positives hidden at cutoff (%)',xticks=[0,1,3,7],ylim=(0,None))
    ax.set_title('Label maturity at a fixed training snapshot')
    fig.savefig(output/'waiting.svg')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    predictions=np.load(run_dir/'test_predictions.npz',allow_pickle=False)
    names={'naive_recent':'Recent labels as observed','mature_only':'Mature labels only',
           'delay_likelihood':'Delay-aware likelihood'}
    for key,label in names.items():
        p=predictions[key]
        y=predictions['labels']
        # Equal-size bins keep the low conversion rate range readable.
        groups=np.array_split(np.argsort(p,kind='stable'),10)
        ax.plot([p[g].mean() for g in groups],[y[g].mean() for g in groups],'o-',label=label)
    limit=ax.get_xlim()[1]
    ax.plot([0,limit],[0,limit],'--',color='gray')
    ax.set(xlabel='Mean predicted conversion probability',ylabel='Observed conversion fraction')
    ax.set_title('Seven-day conversion calibration on held-out impressions')
    ax.legend(frameon=False)
    fig.savefig(output/'calibration.svg')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--output',default='reports/conversion')
    args=parser.parse_args()
    render(args.run_dir,args.output)
