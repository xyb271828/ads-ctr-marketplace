"""Quality-weighted generalized second-price auction with a CPC reserve."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Bid:
    advertiser: str
    bid: float
    pctr: float
    balance: float


@dataclass(frozen=True)
class Winner:
    advertiser: str
    cpc: float
    score: float


def rank(bids, reserve, slots=1):
    if not isfinite(reserve) or reserve < 0 or slots < 1:
        raise ValueError("Expected a nonnegative reserve and positive slot count")
    eligible, seen = [], set()
    for bid in bids:
        if bid.advertiser in seen:
            raise ValueError("Duplicate advertiser")
        seen.add(bid.advertiser)
        if not (isfinite(bid.bid) and bid.bid >= 0 and 0 <= bid.pctr <= 1
                and isfinite(bid.balance) and bid.balance >= 0):
            raise ValueError("Invalid bid, pCTR or budget")
        effective_bid = min(bid.bid, bid.balance)
        if effective_bid >= reserve and effective_bid > 0 and bid.pctr > 0:
            eligible.append((effective_bid * bid.pctr, bid, effective_bid))
    eligible.sort(key=lambda item: (-item[0], item[1].advertiser))
    return [
        Winner(bid.advertiser, min(limit, max(reserve, eligible[i+1][0]/bid.pctr))
               if i+1 < len(eligible) else reserve, score)
        for i, (score, bid, limit) in enumerate(eligible[:slots])
    ]
