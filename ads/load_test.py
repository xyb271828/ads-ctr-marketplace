"""Paced HTTP requests with per-request application timing."""
import argparse
import asyncio
import json
import platform
from time import perf_counter

import httpx
import numpy as np


async def run(root, count, qps):
    if qps <= 0 or count <= 0:
        raise ValueError("qps and requests must be positive")
    routes = [("fractions", 5), ("geometry", 5), ("fractions", 4), ("unmatched", 5)]
    limit = asyncio.Semaphore(25)
    async with httpx.AsyncClient(base_url=root, timeout=5) as client:
        (await client.get("/health")).raise_for_status()
        started = perf_counter()

        async def submit(index):
            due = started + index/qps
            await asyncio.sleep(max(0, due-perf_counter()))
            async with limit:
                sent = perf_counter()
                topic, grade = routes[index % len(routes)]
                try:
                    response = await client.post("/auction", json={"topic": topic, "grade": grade})
                    response.raise_for_status()
                    finished = perf_counter()
                    stage = response.json()["stage_ms"]["pctr_auction"]
                except (httpx.HTTPError, ValueError, KeyError):
                    return None
                return (finished-due)*1000, (finished-sent)*1000, stage

        results = await asyncio.gather(*(submit(i) for i in range(count)))
        elapsed = perf_counter()-started
    okay = [row for row in results if row is not None]
    labels = ("scheduled_to_response_ms", "http_ms", "pctr_auction_ms")
    return {"offered_qps": qps, "achieved_qps": len(okay)/elapsed, "requests": count,
            "errors": count-len(okay), "client_platform": platform.platform(),
            **{name: dict(zip(("p50", "p95", "p99"), np.percentile(x, [50,95,99]).tolist()))
               for name, x in zip(labels, zip(*okay))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--qps", type=float, default=50)
    args = parser.parse_args()
    report = asyncio.run(run(args.url, args.requests, args.qps))
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))
