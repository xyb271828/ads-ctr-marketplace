"""Print a compact Markdown summary from the synthetic CI artifacts."""
import json
from pathlib import Path


def read(name):
    return json.loads((Path("runs") / name).read_text())


def main():
    models = read("synthetic/ctr.json")["models"]
    replay = read("marketplace.json")
    latency = read("latency.json")
    print("## Synthetic CTR (20,000 generated rows)\n")
    print("| Model | LogLoss | NE | AUC | ECE |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, result in models.items():
        print(f"| {name} | {result['log_loss']:.5f} | {result['ne']:.5f} | "
              f"{result['auc']:.5f} | {result['ece']:.5f} |")
    print("\n## Synthetic replay (three seeds, 18,000 requests per day)\n")
    print("| Configuration | Mean clicks | Mean revenue | Early depletion |")
    print("| --- | ---: | ---: | ---: |")
    for name, result in replay["summary"].items():
        print(f"| {name} | {result['clicks']['mean']:.2f} | {result['revenue']['mean']:.2f} | "
              f"{result['early_depletion']['mean']:.1%} |")
    print("\n## Docker latency on this runner\n")
    print(f"{latency['requests']} requests, {latency['errors']} errors, "
          f"{latency['offered_qps']:.0f} offered and {latency['achieved_qps']:.2f} achieved QPS.")
    print(f"pCTR plus auction p99 {latency['pctr_auction_ms']['p99']:.3f} ms. "
          f"HTTP p99 {latency['http_ms']['p99']:.3f} ms.")


if __name__ == "__main__":
    main()
