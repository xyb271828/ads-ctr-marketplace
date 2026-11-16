"""Replay marketplace policies under matched traffic and click draws."""
import argparse
import json
import math
import random
from statistics import mean, stdev

from ads.auction import Bid, rank


MODES = ("calibrated_oracle", "global_10", "segment_10")
CURVE = [1+.65*math.cos((hour-16)*math.pi/12) for hour in range(24)]


def replay(seed, mode, paced, volume):
    catalog, traffic = random.Random(seed), random.Random(seed+1)
    advertisers = [str(i) for i in range(50)]
    caps = {i: catalog.uniform(2, 12) for i in advertisers}
    bids = {i: catalog.uniform(.30, 2.) for i in advertisers}
    true_ctr = {i: catalog.uniform(.018, .075) for i in advertisers}
    groups = {i: catalog.randrange(3) for i in advertisers}
    balance = caps.copy()
    exposure = dict.fromkeys(advertisers, 0)
    depleted = {}
    hourly, clicks = [], 0
    prior_requests = 0
    for hour in range(24):
        target = sum(CURVE[:hour+1]) / sum(CURVE)
        reached = round(target * volume)
        start_spend, start_clicks = sum(balance.values()), clicks
        for _ in range(reached-prior_requests):
            topic = traffic.randrange(3)
            pool = traffic.sample(advertisers, 10)
            clicks_u = {i: traffic.random() for i in pool}
            pacing_u = {i: traffic.random() for i in pool}
            truth = {i: true_ctr[i]*(1.40 if topic == groups[i] else .75) for i in pool}
            eligible = []
            for i in pool:
                spent_fraction = 1-balance[i]/caps[i]
                admission = max(.03, min(1., .50 + 4*(target-spent_fraction)))
                if paced and pacing_u[i] > admission:
                    continue
                biased = mode == "global_10" or (mode == "segment_10" and topic == groups[i] == 0)
                eligible.append(Bid(i, bids[i], truth[i]*(1.10 if biased else 1), balance[i]))
            for slot, winner in enumerate(rank(eligible, reserve=.25, slots=2)):
                i = winner.advertiser
                exposure[i] += 1
                if clicks_u[i] < truth[i]*(1. if slot == 0 else .7):
                    clicks += 1
                    balance[i] -= winner.cpc
                    if balance[i] < .25:
                        depleted.setdefault(i, hour)
        prior_requests = reached
        hourly.append({"hour": hour, "clicks": clicks-start_clicks,
                       "spend": start_spend-sum(balance.values())})
    spend = {i: caps[i]-balance[i] for i in advertisers}
    impressions = sum(exposure.values())
    revenue = sum(spend.values())
    return {"clicks": clicks, "revenue": revenue, "ctr": clicks/max(1, impressions),
            "cpc": revenue/max(1, clicks), "early_depletion": sum(h < 18 for h in depleted.values())/50,
            "coverage": sum(bool(n) for n in exposure.values())/50, "impressions": impressions,
            "advertiser_spend": spend, "exposure": exposure, "hourly": hourly}


def experiment(volume=18000, seeds=(42, 43, 44)):
    if volume <= 0 or len(seeds) < 2:
        raise ValueError("Expected positive request volume and two or more seeds")
    replays = {mode: {str(paced): [replay(seed, mode, paced, volume) for seed in seeds]
                      for paced in (False, True)} for mode in MODES}
    keys = ("clicks", "revenue", "ctr", "cpc", "early_depletion", "coverage")
    comparisons = {}
    for mode, policies in replays.items():
        for paced, runs in policies.items():
            paired = replays["calibrated_oracle"][paced]
            comparisons[f"{mode}/paced={paced}"] = {
                key: {"mean": mean(row[key] for row in runs), "std": stdev(row[key] for row in runs),
                      "delta_vs_oracle": mean(row[key]-base[key] for row, base in zip(runs, paired))}
                for key in keys
            }
    return {"volume": volume, "seeds": seeds, "summary": comparisons, "replays": replays}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, default=18000)
    args = parser.parse_args()
    print(json.dumps(experiment(volume=args.volume), indent=2))
