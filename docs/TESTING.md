# Testing guide

ACCESO is verified at three levels: an automated unit-test suite, module self-tests, and the in-app KPI audit that reproduces Table II of the paper.

## 1. Automated test suite

The suite lives in [`tests/`](../tests) and uses the standard-library `unittest` runner, so no extra dependency is needed.

```bash
python -m unittest discover -s tests -t . -v
```

Run it from the repository root. `tests/__init__.py` adds `src/` to the import path.

| File | What it checks | Paper reference |
|---|---|---|
| `test_data_generator.py` | Grid size, altitude range, 64 villages across the 4 municipalities, bridges in the road network, reproducibility with a fixed seed | Section IV |
| `test_terrain_ai.py` | Random Forest AUC and recall of the *unsuitable* class under spatial cross-validation, 3-class output, feature importances | T8 |
| `test_routes.py` | Tobler hiking speed, terrain route longer than the straight line, vehicle faster than walking, closing roads never shortens a trip | RF05–RF06, HU-04 |
| `test_k_centers.py` | k distinct centers, first center on the most populated suitable community, 2-approximation bound on 25 random networks against the brute-force optimum, forbidden nodes are never chosen | RF07–RF09, T4 |
| `test_scenarios.py` | Configured *k* per center type, centers on allowed terrain classes, ACCESO beats the most-populated baseline, no center on unsuitable terrain, road-closure scenarios | N1, N4 |
| `test_kpi_audit.py` | Placement time, near-linear scaling and memory, road-closure recompute time, approximation factor, Table III reduction, offline operation, guided-assistant accuracy and latency, SUS scoring | T1–T4, T9–T11, N1 |
| `test_ai_assistant.py` | Answers cite the scenario ID, no unverified figures, the right engine tool is used for each kind of question, every declared tool has an implementation | T9, RAG design |

## 2. Module self-tests

Every engine module can be run on its own and prints a short report:

```bash
cd src
python data_generator.py
python terrain_ai.py
python routes.py
python k_centers.py
python scenarios.py
python ai_assistant.py
python kpi_audit.py
```

## 3. In-app KPI audit

Open the **Auditoría KPI** page of the web app (`streamlit run src/main.py`):

- **Ejecutar pruebas del núcleo** runs T1–T4 and the synthetic Table III. The *quick* mode takes about 10 s; the *full* mode uses the paper's sizes (up to 50 000 communities, 300 instances, 30 regions) and takes 1–3 minutes.
- **Probar sin conexión (T11)** blocks network access and runs the core engine.
- **Probar el asistente (T9 y T10)** runs the verified question bank against LM Studio, a cloud model or the guided mode.
- **T5–T7** are entered manually: Lighthouse scores from Chrome DevTools and SUS questionnaires.

The resulting Table II can be downloaded as CSV.

Results of the latest run are recorded in [RESULTS.md](RESULTS.md).
