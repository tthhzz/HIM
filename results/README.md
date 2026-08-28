# Archived results

`paper_metrics.csv` is the compact source of the values printed in the manuscript. The JSON file under `main/` retains per-question predictions and aggregate metrics for the archived HIM run.

## Table I: LoCoMo main results

Each entry is F1 / BLEU-1 (%).

| Method | Multi-hop | Temporal | Open-domain | Single-hop | Adversarial |
|---|---:|---:|---:|---:|---:|
| LoCoMo | 4.61 / 4.29 | 3.11 / 2.71 | 4.55 / 5.97 | 7.03 / 5.69 | 16.95 / 14.81 |
| ReadAgent | 2.47 / 1.78 | 3.01 / 3.01 | 5.57 / 5.22 | 3.25 / 2.51 | 15.78 / 14.01 |
| MemoryBank | 3.60 / 3.39 | 1.72 / 1.97 | 6.63 / 6.58 | 4.11 / 3.32 | 13.07 / 10.30 |
| MemGPT | 5.07 / 4.31 | 2.94 / 2.95 | 7.04 / 7.10 | 7.26 / 5.52 | 14.47 / 12.39 |
| **HIM** | **14.21 / 9.91** | **24.64 / 18.33** | **5.45 / 9.06** | **22.32 / 18.18** | **39.98 / 37.23** |

## Table II: ablation results

Each entry is F1 / BLEU-1 / ROUGE-L (%).

| Method | Multi-hop | Temporal | Adversarial |
|---|---:|---:|---:|
| HIM w/o Encoding | 10.64 / 8.75 / 7.96 | 22.61 / 15.74 / 22.15 | 33.14 / 31.16 / 33.05 |
| HIM w/o Consolidation | 12.40 / 10.77 / 10.28 | 22.37 / 16.69 / 22.89 | 30.88 / 27.64 / 31.33 |
| **HIM (Full)** | **14.21 / 9.91 / 15.22** | **24.64 / 18.33 / 25.05** | **39.98 / 37.23 / 40.17** |

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) before comparing rows or treating these files as a clean reproduction.
