"""Open-loop HTTP benchmark with held-out prediction parity and explicit queue delay."""
import argparse
import asyncio
import json
import platform
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np

from ads.conversion_serving import MODELS


def percentiles(values):
    return dict(zip(('p50', 'p95', 'p99'), np.percentile(values, [50, 95, 99]).tolist()))


async def benchmark(root, fixture, count, rates):
    results = []
    async with httpx.AsyncClient(base_url=root, timeout=10, trust_env=False,
                                limits=httpx.Limits(max_connections=100)) as client:
        health = (await client.get('/health')).json()
        for model in MODELS:
            for batch in (1, 100):
                records = fixture['records'][:batch]
                expected = np.array(fixture['expected'][model][:batch])
                payload = {'model': model, 'records': records}
                for _ in range(20):
                    (await client.post('/score', json=payload)).raise_for_status()
                for qps in rates:
                    semaphore = asyncio.Semaphore(100)
                    start = perf_counter()

                    async def submit(index):
                        due = start + index/qps
                        await asyncio.sleep(max(0, due-perf_counter()))
                        async with semaphore:
                            sent = perf_counter()
                            try:
                                response = await client.post('/score', json=payload)
                                response.raise_for_status()
                                finished = perf_counter()
                                body = response.json()
                                error = float(np.max(np.abs(np.array(body['probabilities'])-expected)))
                                if error > 1e-10 or body['version'] != health['version']:
                                    return {'error': 'prediction/version mismatch'}
                                return {'http_ms': (finished-sent)*1000,
                                        'scheduled_ms': (finished-due)*1000,
                                        'dispatch_lag_ms': (sent-due)*1000,
                                        'server_ms': body['stage_ms']['total'],
                                        'prediction_max_abs_error': error}
                            except (httpx.HTTPError, ValueError, KeyError) as exc:
                                return {'error': str(exc)}
                    rows = await asyncio.gather(*(submit(i) for i in range(count)))
                    elapsed = perf_counter()-start
                    okay = [row for row in rows if 'error' not in row]
                    result = {'model': model, 'candidates': batch, 'offered_qps': qps,
                              'requests': count, 'errors': count-len(okay),
                              'elapsed_seconds': elapsed, 'achieved_qps': len(okay)/elapsed,
                              'max_prediction_abs_error': max((r['prediction_max_abs_error'] for r in okay), default=None),
                              'latency_ms': {key: percentiles([r[key] for r in okay]) for key in
                                             ('http_ms', 'scheduled_ms', 'dispatch_lag_ms', 'server_ms')} if okay else {},
                              'error_examples': [r['error'] for r in rows if 'error' in r][:3]}
                    results.append(result)
                    print('SERVING_RESULT '+json.dumps(result), flush=True)
    return {'health': health, 'client': platform.platform(), 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8769')
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests', type=int, default=1000)
    parser.add_argument('--qps', type=float, nargs='+', default=[50, 200])
    args = parser.parse_args()
    if args.requests <= 0 or any(rate <= 0 for rate in args.qps):
        parser.error('Requests and rates must be positive')
    result = asyncio.run(benchmark(args.url, json.loads(Path(args.fixture).read_text()), args.requests, args.qps))
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    if any(row['errors'] for row in result['results']):
        raise SystemExit('Benchmark contained errors')


if __name__ == '__main__':
    main()
