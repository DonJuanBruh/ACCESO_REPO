import contextlib
import socket
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay

import config
import k_centers


def synthetic_network(V, rng, side_km=None):
    side_km = side_km or 1.2 * np.sqrt(V)
    points = rng.uniform(0, side_km, (V, 2))
    tri = Delaunay(points).simplices
    edges = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    o, d = edges[:, 0], edges[:, 1]
    phase = rng.uniform(0, 6.28, 3)
    altitude = (600 * np.sin(points[:, 0] / 9 + phase[0]) * np.cos(points[:, 1] / 11 + phase[1])
                + 250 * np.sin((points[:, 0] + points[:, 1]) / 4 + phase[2]))
    length = np.linalg.norm(points[o] - points[d], axis=1)
    elevation_gain_km = np.abs(altitude[o] - altitude[d]) / 1000
    weight = length * rng.uniform(config.SINUOSITY_MIN, config.SINUOSITY_MAX, len(o)) \
        * (1 + 2 * elevation_gain_km / np.maximum(length, 1e-6))
    graph = csr_matrix((np.r_[weight, weight], (np.r_[o, d], np.r_[d, o])), shape=(V, V))
    population = np.round(rng.lognormal(5, 0.8, V))
    suitable = rng.uniform(0, 1, V) > 0.24
    return graph, population, suitable


def percentile95(vals):
    return float(np.percentile(vals, 95))


def gonzalez_with_matrix(D, population, suitable, k):
    suitable_cells = np.flatnonzero(suitable)
    centers = [int(suitable_cells[np.argmax(population[suitable_cells])])]
    cost = D[:, centers[0]].copy()
    while len(centers) < k:
        farthest = int(np.argmax(cost))
        target = farthest if suitable[farthest] else int(suitable_cells[np.argmin(D[farthest, suitable_cells])])
        centers.append(target)
        cost = np.minimum(cost, D[:, target])
    return centers


def test_scalability(mode="quick", k=5, seed=config.SEED, progress=None):
    rng = np.random.default_rng(seed)
    rows = []
    for V in config.TEST_SIZES[mode]:
        if progress:
            progress(f"Red sintética de {V:,} comunidades...")
        g, pop, suitable = synthetic_network(V, rng)
        nodes = np.arange(V)
        repetitions = 15 if V <= 10000 else 7
        timings = []
        for _ in range(repetitions):
            start = time.perf_counter()
            k_centers.place_k_centers(g, g, nodes, pop, suitable, k)
            timings.append((time.perf_counter() - start) * 1000)
        row = {"V": V, "Incremental p50 (ms)": float(np.median(timings)),
               "Incremental p95 (ms)": percentile95(timings), "Bytes por vector (MB)": V * 8 / 1e6,
               "Matriz (ms)": np.nan, "Matriz memoria (MB)": V * V * 8 / 1e6}
        if V <= config.MAX_V_MATRIX[mode]:
            start = time.perf_counter()
            D = dijkstra(g)
            gonzalez_with_matrix(D, pop, suitable, k)
            row["Matriz (ms)"] = (time.perf_counter() - start) * 1000
        rows.append(row)
    table = pd.DataFrame(rows)

    slope_inc = float(np.polyfit(np.log10(table["V"]), np.log10(table["Incremental p50 (ms)"]), 1)[0])
    with_matrix = table.dropna(subset=["Matriz (ms)"])
    slope_mat = float(np.polyfit(np.log10(with_matrix["V"]), np.log10(with_matrix["Matriz (ms)"]), 1)[0]) \
        if len(with_matrix) >= 2 else float("nan")

    if progress:
        progress("Cierres de vía en una red de 5 000 comunidades...")
    g, pop, suitable = synthetic_network(5000, rng)
    coo = g.tocoo()
    recomputes = []
    for _ in range(20):
        e = int(rng.integers(coo.nnz))
        a, b = coo.row[e], coo.col[e]
        start = time.perf_counter()
        remove = ~(((coo.row == a) & (coo.col == b)) | ((coo.row == b) & (coo.col == a)))
        g2 = csr_matrix((coo.data[remove], (coo.row[remove], coo.col[remove])), shape=g.shape)
        k_centers.place_k_centers(g2, g2, np.arange(5000), pop, suitable, k)
        recomputes.append((time.perf_counter() - start) * 1000)

    largest = table.iloc[-1]
    summary = {
        "V_max": int(largest["V"]), "T1_p95_ms": float(largest["Incremental p95 (ms)"]),
        "T2_slope": slope_inc, "T2_slope_matrix": slope_mat,
        "T2_mb": float(largest["Bytes por vector (MB)"]), "T3_p95_ms": percentile95(recomputes),
    }
    return table, summary


def test_approximation(mode="quick", n=12, k=3, seed=config.SEED):
    rng = np.random.default_rng(seed + 1)
    factors, refined = [], []
    for _ in range(config.APPROXIMATION_INSTANCES[mode]):
        g, pop, _ = synthetic_network(n, rng, side_km=40)
        suitable = np.ones(n, dtype=bool)
        D = dijkstra(g)
        optimum, _ = k_centers.brute_force_optimum(D, k)
        res = k_centers.place_k_centers(g, g, np.arange(n), pop, suitable, k)
        factors.append(res["cost"].max() / optimum)
        nodes = [c["node"] for c in res["centers"]]
        _, _, cost_ref, _ = k_centers.refine_centers(D, nodes, suitable)
        refined.append(cost_ref.max() / optimum)
    factors, refined = np.array(factors), np.array(refined)
    return {"instances": len(factors), "mean": float(factors.mean()), "max": float(factors.max()),
            "within_bound": float(100 * (factors <= 2 + 1e-9).mean()),
            "mean_refined": float(refined.mean()), "max_refined": float(refined.max())}


def test_table_iii(mode="quick", V=400, k=6, side_km=58, seed=config.SEED):
    rng = np.random.default_rng(seed + 2)
    nodes = np.arange(V)
    records, improvements, best, n2, n4, cost_n4 = [], [], [], [], [], []

    def measure_strategy(name, cost, pop):
        total = pop.sum()
        return {"Estrategia": name, "Dist. máx. (km)": float(cost.max()),
                "Pobl. >25 km (%)": float(100 * pop[cost > config.FAR_THRESHOLD_KM].sum() / total),
                "Pobl. <8 km (%)": float(100 * pop[cost < config.NEAR_THRESHOLD_KM].sum() / total)}

    for _ in range(config.TABLE_III_REGIONS[mode]):
        g, pop, suitable = synthetic_network(V, rng, side_km)
        suitable_cells = np.flatnonzero(suitable)
        populated = suitable_cells[np.argsort(-pop[suitable_cells])[:k]]
        cost_pop = dijkstra(g, indices=populated, min_only=True)
        random_nodes = rng.choice(suitable_cells, k, replace=False)
        cost_random = dijkstra(g, indices=random_nodes, min_only=True)
        access = k_centers.place_k_centers(g, g, nodes, pop, suitable, k)
        without_ai = k_centers.place_k_centers(g, g, nodes, pop, np.ones(V, bool), k)
        rows = [measure_strategy("Comunidades más pobladas", cost_pop, pop), measure_strategy("Al azar (sin análisis)", cost_random, pop),
                measure_strategy("ACCESO con IA de terreno", access["cost"], pop)]
        records += rows
        improvements.append(100 * (rows[0]["Dist. máx. (km)"] - rows[2]["Dist. máx. (km)"]) / rows[0]["Dist. máx. (km)"])
        best.append(rows[2]["Dist. máx. (km)"] < rows[0]["Dist. máx. (km)"])
        if rows[0]["Pobl. >25 km (%)"] > 0:
            n2.append(100 * (rows[0]["Pobl. >25 km (%)"] - rows[2]["Pobl. >25 km (%)"]) / rows[0]["Pobl. >25 km (%)"])
        nodes_without_ai = [c["node"] for c in without_ai["centers"]]
        n4.append(100 * np.mean([not suitable[n] for n in nodes_without_ai]))
        cost_n4.append(100 * (access["cost"].max() - without_ai["cost"].max()) / without_ai["cost"].max())
    table = pd.DataFrame(records).groupby("Estrategia", sort=False).mean().reset_index()
    return table, {"regions": len(improvements), "N1": float(np.mean(improvements)), "pct_better": float(100 * np.mean(best)),
                   "N2": float(np.mean(n2)) if n2 else float("nan"), "N4_avoided": float(np.mean(n4)),
                   "N4_cost": float(np.mean(cost_n4))}


def test_territory_closure(base, params, repetitions=5, seed=config.SEED):
    import scenarios
    rng = np.random.default_rng(seed)
    timings = []
    for t in rng.choice(len(base["territory"]["roads"]), repetitions, replace=False):
        p = scenarios.parameters(dict(params["ks"]), params["mode"], params["criterion"], params["refine"],
                                 tuple(params["closed"]) + (int(t),), params["fixed_centers"])
        start = time.perf_counter()
        scenarios.compute_scenario(base, p)
        timings.append((time.perf_counter() - start) * 1000)
    return percentile95(timings)


@contextlib.contextmanager
def network_disabled():
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)

    def block(*args, **kwargs):
        raise OSError("Red desactivada para la prueba T11")

    socket.socket.connect = block
    socket.socket.connect_ex = block
    socket.create_connection = block
    socket.getaddrinfo = block
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = originals


def test_offline(base, params):
    import ai_assistant
    import data_generator
    import scenarios
    import routes
    import terrain_ai

    tests = {
        "Generar territorio (DEM, ríos, vías)": lambda: data_generator.generate_territory(),
        "IA de terreno: clasificar celdas": lambda: terrain_ai.classify_cells(base["ai"]["model"], base["territory"]),
        "Motor de rutas: construir grafos": lambda: routes.build_graphs(base["network"], params["mode"], (0,)),
        "Motor de ubicación: escenario nuevo": lambda: scenarios.compute_scenario(
            base, scenarios.parameters(dict(params["ks"]), params["mode"], params["criterion"], params["refine"],
                                       (1, 2), params["fixed_centers"])),
        "Cierre de vía y recálculo (HU-04)": lambda: scenarios.compute_scenario(
            base, scenarios.parameters(dict(params["ks"]), params["mode"], params["criterion"], False, (3,))),
        "Comparador de estrategias (Tabla III)": lambda: scenarios.compare_strategies(
            base, scenarios.compute_scenario(base, params), next(iter(dict(params["ks"]))), draws=3),
        "Asistente: modo guiado sin red": lambda: ai_assistant.respond(
            "¿Cuál es el peor viaje?", [], ai_assistant.Context(base, scenarios.compute_scenario(base, params)),
            force_offline=False),
    }
    rows = []
    with network_disabled():
        for name, func in tests.items():
            try:
                func()
                rows.append({"Función del núcleo": name, "Resultado": "OK"})
            except Exception as error:
                rows.append({"Función del núcleo": name, "Resultado": f"Falla: {error}"})
    table = pd.DataFrame(rows)
    return table, float(100 * (table["Resultado"] == "OK").mean())


def test_assistant(base, scenario, n_questions=10, force_offline=False, progress=None):
    import ai_assistant
    context = ai_assistant.Context(base, scenario)
    bank = ai_assistant.question_bank(context)[:n_questions]
    rows = []
    for i, item in enumerate(bank):
        if progress:
            progress(f"Pregunta {i + 1} de {len(bank)}")
        r = ai_assistant.respond(item["question"], [], context, force_offline=force_offline)
        hit = ai_assistant.contains_value(r["text"], item["expected"])
        rows.append({"Pregunta": item["question"], "Esperado": str(item["expected"]), "Correcta": hit,
                     "Cifras verificadas": not r["verification"]["unverified"],
                     "Latencia (s)": r["latency_s"], "Fuente": r["source"]})
    table = pd.DataFrame(rows)
    summary = {"T9": float(100 * table["Correcta"].mean()) if len(table) else float("nan"),
               "T9_no_fabrication": bool(table["Cifras verificadas"].all()) if len(table) else False,
               "T10": percentile95(table["Latencia (s)"]) if len(table) else float("nan"),
               "source": table["Fuente"].mode().iloc[0] if len(table) else "–"}
    return table, summary


def sus_score(responses):
    total = sum((r - 1) if i % 2 == 0 else (5 - r) for i, r in enumerate(responses))
    return total * 2.5


def _status(value, passes):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return "✓" if passes else "×"


def kpi_table(r):
    M = config.TARGETS
    g = r.get

    def num(key):
        v = g(key)
        return v if v is not None else float("nan")

    def first(*key_list):
        for key in key_list:
            v = g(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return float("nan")

    def f(v, template):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return template.format(v)

    def join_parts(*parts):
        measures = [p for p in parts if p]
        return " · ".join(measures) if measures else "Por medir"

    rows = [
        ("N1", "Reducción de distancia máxima", f"≥{M['N1']:.0f} %",
         join_parts(f(g("N1_scn"), "{:.1f} % (territorio)"), f(g("N1_synth"), "{:.1f} % (regiones sintéticas)")),
         _status(first("N1_scn", "N1_synth"), first("N1_scn", "N1_synth") >= M["N1"]),
         "Escenario actual / prueba sintética"),
        ("N2", "Población en situación extrema (>25 km)", f"Reducir ≥{M['N2']:.0f} %",
         join_parts(f(g("N2_scn"), "{:.1f} % menos (territorio)"), f(g("N2_synth"), "{:.1f} % menos (sintético)")),
         _status(first("N2_scn", "N2_synth"), first("N2_scn", "N2_synth") >= M["N2"]),
         "Escenario actual / prueba sintética"),
        ("N3", "Cobertura a 1 h a pie", f"+{M['N3']:.0f} p. p.", join_parts(f(g("N3_scn"), "{:+.1f} p. p.")),
         _status(num("N3_scn"), num("N3_scn") >= M["N3"]), "Escenario actual vs. más pobladas"),
        ("N4", "Centros propuestos en terreno no apto", "0 %",
         join_parts(f(g("N4_with_ai"), "{:.0f} % con IA"), f(g("N4_without_ai"), "sin IA caerían {:.0f} %"),
                    f(g("N4_cost"), "costo {:+.1f} %")),
         _status(num("N4_with_ai"), num("N4_with_ai") <= M["N4"]), "Comparador de estrategias"),
        ("N5", "Adopción institucional", "4 municipios (año 2)", "Por medir", "–", "Seguimiento institucional"),
        ("N6", "Brecha rural-urbana (tiempo)", "Tendencia a 1,0", join_parts(f(g("N6_scn"), "{:.2f}"), f(g("N6_base"), "más pobladas {:.2f}")),
         _status(num("N6_scn"), abs(num("N6_scn") - 1) <= abs(num("N6_base") - 1)),
         "Escenario actual vs. más pobladas"),
        ("T1", "Tiempo de ubicación (p95)", f"≤{M['T1']:.0f} ms", join_parts(f(g("T1"), "{:.1f} ms"), f(g("T1_V"), "V = {:,}")),
         _status(num("T1"), num("T1") <= M["T1"]), "Prueba del núcleo"),
        ("T2", "Escalabilidad y memoria", "pendiente ≈1; <50 MB",
         join_parts(f(g("T2_slope"), "pendiente {:.2f}"), f(g("T2_slope_matrix"), "matriz {:.2f}"),
                    f(g("T2_mb"), "{:.2f} MB")),
         _status(num("T2_slope"), num("T2_slope") <= M["T2_slope"] and num("T2_mb") < M["T2_mb"]),
         "Prueba del núcleo"),
        ("T3", "Recálculo por vía cerrada (p95)", f"≤{M['T3']:.0f} ms",
         join_parts(f(g("T3"), "{:.1f} ms (V = 5000)"), f(g("T3_territory"), "{:.0f} ms (relieve completo)")),
         _status(num("T3"), num("T3") <= M["T3"]), "Prueba del núcleo / territorio"),
        ("T4", "Calidad de la ubicación (factor)", "≤2; media ≤1,5",
         join_parts(f(g("T4_mean"), "media {:.2f}"), f(g("T4_max"), "máx. {:.2f}"), f(g("T4_mean_ref"), "refinado {:.2f}")),
         _status(num("T4_mean"), num("T4_max") <= M["T4_max"] and num("T4_mean") <= M["T4_mean"]),
         "Prueba del núcleo"),
        ("T5", "Lighthouse: accesib., buenas prác., SEO", f"≥{M['T5']} c/u", join_parts(f(g("T5"), "{}")),
         _status(g("T5_min") if g("T5_min") is not None else float("nan"), (g("T5_min") or 0) >= M["T5"]),
         "Medición manual (Chrome DevTools)"),
        ("T6", "Lighthouse: rendimiento", f"≥{M['T6']}", join_parts(f(g("T6"), "{:.0f}")),
         _status(num("T6"), num("T6") >= M["T6"]), "Medición manual (Chrome DevTools)"),
        ("T7", "Usabilidad percibida (SUS)", f"≥{M['T7']}", join_parts(f(g("T7"), "{:.1f}"), f(g("T7_n"), "{} personas")),
         _status(num("T7"), num("T7") >= M["T7"] and (g("T7_n") or 0) >= 5), "Cuestionario SUS en la app"),
        ("T8", "Clasificador de terreno (CV espacial)", f"AUC ≥{M['T8_auc']}; recall ≥{M['T8_recall']}",
         join_parts(f(g("T8_auc"), "AUC {:.3f}"), f(g("T8_recall"), "recall No apto {:.3f}")),
         _status(num("T8_auc"), num("T8_auc") >= M["T8_auc"] and num("T8_recall") >= M["T8_recall"]),
         "IA de terreno"),
        ("T9", "Exactitud del asistente IA", f"≥{M['T9']:.0f} %; 0 cifras no verificadas",
         join_parts(f(g("T9"), "{:.0f} %"),
                    None if g("T9") is None else ("sin cifras inventadas" if g("T9_no_fabrication") else "hay cifras no verificadas"),
                    f(g("T9_source"), "{}")),
         _status(num("T9"), num("T9") >= M["T9"] and bool(g("T9_no_fabrication"))), "Banco de preguntas"),
        ("T10", "Latencia del asistente (p95)", f"≤{M['T10']:.0f} s", join_parts(f(g("T10"), "{:.2f} s")),
         _status(num("T10"), num("T10") <= M["T10"]), "Banco de preguntas"),
        ("T11", "Operación sin conexión", "100 %", join_parts(f(g("T11"), "{:.0f} %")),
         _status(num("T11"), num("T11") >= M["T11"]), "Prueba con la red bloqueada"),
    ]
    return pd.DataFrame(rows, columns=["ID", "KPI", "Meta", "Medido", "Estado", "Fuente"])


def scenario_kpi(scenario, kind, comparison=None):
    r, b = scenario["types"][kind]["eval"]["kpi"], scenario["bases"][kind]["eval"]["kpi"]
    output = {
        "N1_scn": 100 * (b["max_km"] - r["max_km"]) / b["max_km"] if b["max_km"] > 0 else float("nan"),
        "N2_scn": 100 * (b["pct_over_25km"] - r["pct_over_25km"]) / b["pct_over_25km"] if b["pct_over_25km"] > 0 else float("nan"),
        "N3_scn": r["pct_under_1h_walk"] - b["pct_under_1h_walk"],
        "N6_scn": r["rural_urban_gap"], "N6_base": b["rural_urban_gap"],
    }
    if comparison:
        output.update({"N4_with_ai": comparison["pct_with_ai_unsuitable"], "N4_without_ai": comparison["pct_without_ai_unsuitable"],
                       "N4_cost": comparison["cost_pct"]})
    return output


if __name__ == "__main__":
    start = time.perf_counter()
    table, res = test_scalability("quick", progress=print)
    print(table.round(2).to_string(index=False))
    print(res)
    ap = test_approximation("quick")
    print("T4:", ap)
    t3, r3 = test_table_iii("quick")
    print(t3.round(1).to_string(index=False))
    print("Table III:", r3)
    measured = {"T1": res["T1_p95_ms"], "T1_V": res["V_max"], "T2_slope": res["T2_slope"],
                "T2_mb": res["T2_mb"], "T3": res["T3_p95_ms"], "T4_mean": ap["mean"], "T4_max": ap["max"],
                "T4_mean_ref": ap["mean_refined"], "N1_synth": r3["N1"], "N2_synth": r3["N2"], "T2_slope_matrix": res["T2_slope_matrix"]}
    print(kpi_table(measured).to_string(index=False))
    print(f"\nTests finished in {time.perf_counter() - start:.0f} s")
