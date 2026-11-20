# Results

This file has two parts:

1. The results reported in the [paper](../paper/Paper_ACCESO_DDYA.pdf).
2. The results of the latest local run of this repository.

All data is synthetic, calibrated for the Provincia de La Libertad (Boyacá, Colombia).

## Results reported in the paper

Conditions: the core engine (full mode) ran on a 2.1 GHz virtual core with 3 GB of RAM.

| ID | KPI | Target | Result | Status |
|---|---|---|---|---|
| N1 | Maximum-distance reduction | ≥ 20 % | 26.4 % (30 synthetic regions) | ✅ |
| N2 | Population in extreme situation (> 25 km) | reduce ≥ 30 % | 20.7 % → 13.3 % (−35.9 %) | ✅ |
| N4 | Centers on unsuitable terrain | 0 % proposed | 23.9 % avoided, cost +1.4 % | ✅ |
| T1 | Placement time, p95 | ≤ 2 s (V ≤ 50 000) | 78.8 ms with V = 50 000 | ✅ |
| T2 | Scalability and memory | slope ≈ 1, < 50 MB | slope 1.02, 0.4 MB | ✅ |
| T3 | Road-closure recompute, p95 | ≤ 1 s | 10.3 ms (V = 5000) | ✅ |
| T4 | Approximation factor | ≤ 2, mean ≤ 1.5 | mean 1.46, max 1.95 | ✅ |
| T5 | Lighthouse: accessibility / best practices / SEO | ≥ 75 each | 100 / 78 / 82 | ✅ |
| T6 | Lighthouse: performance | ≥ 75 | 3 (correction plan in Section V-B) | ❌ |

### Table III: simulated impact, mean of 30 regions

| Strategy | Max. distance (km) | Pop. > 25 km | Pop. < 8 km |
|---|---|---|---|
| Most populated communities | 46.5 | 20.7 % | 26.2 % |
| Random (no analysis) | 45.0 | 20.2 % | – |
| **ACCESO with terrain AI** | **34.3** | **13.3 %** | 17.1 % |

## Latest local run

Date: 2026-11-19.

Environment:

- Windows 11, Intel Core (Tiger Lake)
- Python 3.14.2
- Streamlit 1.67, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, NetworkX 3.7, pandas 3.1, Plotly 7.1, openai 3.51

### Automated suite

```text
$ python -m unittest discover -s tests -t .
Ran 35 tests in 10.0s
OK
```

### Core engine, quick mode (`python src/kpi_audit.py`)

| V (communities) | Incremental p50 (ms) | Incremental p95 (ms) | Memory per vector (MB) | Full matrix (ms) | Matrix memory (MB) |
|---|---|---|---|---|---|
| 500 | 1.36 | 2.04 | 0.00 | 32.2 | 2 |
| 1 000 | 4.73 | 7.44 | 0.01 | 129.9 | 8 |
| 2 000 | 5.72 | 9.39 | 0.02 | – | 32 |
| 5 000 | 16.21 | 20.73 | 0.04 | – | 200 |
| 10 000 | 23.24 | 29.44 | 0.08 | – | 800 |

The full matrix is only timed up to 1 000 communities in quick mode.

| ID | Measured | Target | Status |
|---|---|---|---|
| T1 | 29.4 ms p95 with V = 10 000 | ≤ 2000 ms | ✅ |
| T2 | log-log slope 0.91 (matrix 2.01), 0.08 MB | slope ≈ 1, < 50 MB | ✅ |
| T3 | 13.5 ms p95 (V = 5000); 354 ms on the full relief | ≤ 1000 ms | ✅ |
| T4 | mean 1.28, max 1.82, 100 % within bound; refined mean 1.08 | ≤ 2, mean ≤ 1.5 | ✅ |
| T8 | AUC 0.953, recall (unsuitable) 0.894, accuracy 93.0 % | AUC ≥ 0.85, recall ≥ 0.80 | ✅ |
| T9 | 100 % correct, 0 unverified figures (guided mode) | ≥ 90 % | ✅ |
| T10 | 4.09 s p95 (guided mode) | ≤ 5 s | ✅ |
| T11 | 100 % of core functions work offline | 100 % | ✅ |

### Table III, quick mode (10 regions)

| Strategy | Max. distance (km) | Pop. > 25 km (%) | Pop. < 8 km (%) |
|---|---|---|---|
| Most populated communities | 50.7 | 26.9 | 19.6 |
| Random (no analysis) | 52.2 | 32.1 | 14.6 |
| **ACCESO with terrain AI** | **34.4** | **17.8** | 14.5 |

- **N1:** 29.7 %. ACCESO was better in 100 % of the regions.
- **N2:** 26.4 % in quick mode, below the 30 % target. The paper's 35.9 % comes from the full mode with 30 regions.

### Case study: La Libertad health centers (k = 3)

| Strategy | Worst trip (min) | Worst distance (km) | Pop. < 1 h walking (%) | Centers on unsuitable terrain |
|---|---|---|---|---|
| **ACCESO with terrain AI (refined)** | **86.6** | **25.9** | 4.0 | 0 of 3 |
| ACCESO, paper rules only (not refined) | 145.7 | 37.0 | 23.0 | 0 of 3 |
| Most populated communities | 142.6 | 33.0 | 25.4 | 0 of 3 |
| Random (mean of 20 draws) | 168.5 | 41.8 | 10.1 | – |
| ACCESO without terrain AI | 134.4 | 35.1 | 19.2 | 1 of 3 |
| Maximum coverage at 60 min (RF12) | 120.6 | 27.5 | 1.5 | 0 of 3 |

ACCESO cuts the worst trip from 142.6 min to 86.6 min, a 39.3 % reduction. Without the terrain AI, 33 % of the centers would fall on unsuitable terrain.

As in the paper, minimizing the worst trip lowers the share of people within 1 h walking (4.0 % vs. 25.4 %). This is why N3 does not pass in this scenario, and why RF12 lets the planner compare both criteria.

### Pending measurements

- **N3, N5, N6:** need real data from La Libertad.
- **T5–T7:** Lighthouse and SUS scores are entered manually in the app.
- **T9–T10 with a language model:** need LM Studio or a cloud model connected.
