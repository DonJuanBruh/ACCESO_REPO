import hashlib
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import config
import data_generator
import k_centers
import routes
import terrain_ai

URBAN_INTERNAL_MINUTES = 5


def prepare_base(seed=config.SEED):
    territory = data_generator.generate_territory(seed)
    ai = terrain_ai.train_or_load(territory)
    suit_class, prob = terrain_ai.classify_cells(ai["model"], territory)
    network = routes.prepare_network(territory)
    settlements = territory["settlements"]
    demand = settlements[settlements["population"] > 0].reset_index(drop=True)
    relief = territory["relief"]
    return {
        "territory": territory, "ai": ai, "suitability": suit_class, "prob": prob, "network": network,
        "demand": demand,
        "demand_cells": demand["cell"].to_numpy(),
        "population": demand["population"].to_numpy(dtype=float),
        "is_urban": (demand["type"] == "cabecera").to_numpy(),
        "settlement_tree": cKDTree(settlements[["x_km", "y_km"]].to_numpy()),
        "not_buildable": (relief["main_river"] | relief["stream"]).ravel(),
        "_cache": OrderedDict(), "_lock": threading.Lock(),
    }


def _cached(base, key, func, max_items=16):
    with base["_lock"]:
        if key in base["_cache"]:
            base["_cache"].move_to_end(key)
            return base["_cache"][key]
    value = func()
    with base["_lock"]:
        base["_cache"][key] = value
        while len(base["_cache"]) > max_items:
            base["_cache"].popitem(last=False)
    return value


def get_graphs(base, mode, closed):
    return _cached(base, ("graphs", mode, tuple(closed)),
                   lambda: routes.build_graphs(base["network"], mode, closed))


def get_matrix(base, mode, closed, metric):
    graphs = get_graphs(base, mode, closed)
    return _cached(base, ("D", mode, tuple(closed), metric),
                   lambda: k_centers.demand_matrix(graphs[metric], base["demand_cells"]), max_items=24)


def parameters(ks=None, mode=config.TRAVEL_MODES[1], criterion=config.CRITERIA[0], refine=True,
               closed=(), fixed=()):
    ks = ks if ks is not None else {t: v["k"] for t, v in config.CENTER_TYPES.items()}
    p = {
        "ks": tuple((t, int(ks.get(t, 0))) for t in config.CENTER_TYPES),
        "mode": mode, "criterion": criterion, "refine": bool(refine),
        "closed": tuple(sorted(int(c) for c in closed)),
        "fixed_centers": tuple(sorted((t, int(i)) for t, i in fixed)),
    }
    p["id"] = "ESC-" + hashlib.sha1(repr(sorted(p.items())).encode()).hexdigest()[:6].upper()
    return p


def metric_of(criterion):
    return ("time", "min") if criterion == config.CRITERIA[0] else ("km", "km")


def describe_cell(base, cell):
    grid = base["territory"]["grid"]
    row, col = divmod(int(cell), grid["nx"])
    d, i = base["settlement_tree"].query([grid["x_km"][col], grid["y_km"][row]])
    p = base["territory"]["settlements"].iloc[int(i)]
    if d < 0.3:
        return f"en {p['name']}"
    return f"a {d:.1f} km de {p['name']} ({p['municipality']})"


def allowed_for(base, graphs, kind):
    rule = config.CENTER_TYPES[kind]
    mask = np.isin(base["suitability"].ravel(), rule["allowed_classes"]) & ~base["not_buildable"]
    if rule["requires_road"]:
        mask &= graphs["open_road_cells"]
    return mask


def move_to_allowed(graphs, metric, node, allowed, occupied):
    if allowed[node] and node not in occupied:
        return int(node)
    from_cost = np.where(allowed, dijkstra(graphs[metric], indices=int(node)), np.inf)
    from_cost[list(occupied)] = np.inf
    return int(np.argmin(from_cost))


def evaluate(base, graphs, nodes, metric):
    dem, pop = base["demand_cells"], base["population"]
    output = {}
    for m in ("time", "km", "walk"):
        cost, next_node, source = routes.cost_to(graphs, m, nodes)
        output[m] = cost[dem]
        if m == metric:
            output["next"], output["source"] = next_node, source
            position = {int(n): j for j, n in enumerate(nodes)}
            output["assignment"] = np.array([position.get(int(f), -1) if np.isfinite(cost[c]) else -1
                                             for f, c in zip(source[dem], dem)])
    total = pop.sum()
    t, km, walk = output["time"], output["km"], output["walk"]
    sel = output[metric]
    finite = np.isfinite(sel)
    farthest = int(np.argmax(np.where(finite, sel, -1))) if finite.any() else None
    rural, urban = ~base["is_urban"], base["is_urban"]

    def mean(vals, mask):
        ok = mask & np.isfinite(vals)
        return float(np.average(vals[ok], weights=pop[ok])) if ok.any() and pop[ok].sum() else float("nan")

    output["kpi"] = {
        "max_time_min": float(t[np.isfinite(t)].max()) if np.isfinite(t).any() else float("nan"),
        "max_km": float(km[np.isfinite(km)].max()) if np.isfinite(km).any() else float("nan"),
        "max_criterion": float(sel[finite].max()) if finite.any() else float("nan"),
        "mean_time_min": mean(t, np.ones_like(rural)),
        "pct_over_25km": float(100 * pop[~(km <= config.FAR_THRESHOLD_KM)].sum() / total),
        "pct_over_2h": float(100 * pop[~(t <= config.FAR_THRESHOLD_MIN)].sum() / total),
        "pct_under_1h_walk": float(100 * pop[walk <= config.THRESHOLD_1H_MIN].sum() / total),
        "pct_under_8km": float(100 * pop[km <= config.NEAR_THRESHOLD_KM].sum() / total),
        "rural_urban_gap": (mean(t, rural) + URBAN_INTERNAL_MINUTES)
                               / (mean(t, urban) + URBAN_INTERNAL_MINUTES),
        "no_route": int((~finite).sum()),
    }
    output["farthest"] = farthest
    return output


def place_kind(base, graphs, params, kind, k, fixed_cells):
    metric, unit = metric_of(params["criterion"])
    allowed = allowed_for(base, graphs, kind)
    names = base["demand"]["name"].tolist()

    def describe(node):
        return describe_cell(base, node)

    start = time.perf_counter()
    res = k_centers.place_k_centers(graphs[metric], graphs[metric + "_T"], base["demand_cells"],
                                    base["population"], allowed, k, fixed=fixed_cells,
                                    names=names, describe=describe, unit=unit)
    ms_rules = (time.perf_counter() - start) * 1000
    nodes = [c["node"] for c in res["centers"]]
    steps = list(res["steps"])
    moves = []
    if params["refine"] and nodes:
        D = get_matrix(base, params["mode"], params["closed"], metric)
        refined_nodes, _, _, moves = k_centers.refine_centers(
            D, nodes, allowed, [c["fixed"] for c in res["centers"]], unit=unit, describe=describe)
        nodes = refined_nodes
        steps += [f"Refinamiento hacia el punto céntrico apto — {m}" for m in moves] or \
                 ["Refinamiento: ningún centro podía acercarse más a su grupo sin salir del terreno apto."]
    ms_total = (time.perf_counter() - start) * 1000

    ev = evaluate(base, graphs, nodes, metric)
    grid, relief = base["territory"]["grid"], base["territory"]["relief"]
    centers = []
    for j, node in enumerate(nodes):
        row, col = divmod(node, grid["nx"])
        members = ev["assignment"] == j
        original = res["centers"][j]
        moved = original["moved_from"]
        centers.append({
            "number": j + 1, "node": int(node), "type": kind,
            "lat": float(grid["lat"][row]), "lon": float(grid["lon"][col]),
            "x_km": float(grid["x_km"][col]), "y_km": float(grid["y_km"][row]),
            "altitude": float(relief["altitude"][row, col]),
            "slope": float(relief["slope"][row, col]),
            "ai_suitability": config.CLASSES[int(base["suitability"][row, col])],
            "location": describe(node),
            "reason": original["reason"] + (" Luego se refinó hacia el punto céntrico apto de su grupo."
                                            if node != original["node"] else ""),
            "moved_from": names[moved] if moved is not None else None,
            "existing": original["fixed"],
            "communities": int(members.sum()),
            "inhabitants": int(base["population"][members].sum()),
            "worst_criterion": float(ev[metric][members].max()) if members.any() else 0.0,
            "worst_time_min": float(ev["time"][members].max()) if members.any() else 0.0,
            "worst_km": float(ev["km"][members].max()) if members.any() else 0.0,
        })
    return {"type": kind, "k": k, "centers": centers, "eval": ev, "steps": steps,
            "moves": moves, "ms_rules": ms_rules, "ms_total": ms_total,
            "dijkstras": res["dijkstras"], "metric": metric, "unit": unit,
            "allowed_pct": float(100 * allowed.mean())}


def baseline(base, graphs, params, kind, k):
    metric, _ = metric_of(params["criterion"])
    allowed = allowed_for(base, graphs, kind)
    order = np.argsort(-base["population"])[:k]
    nodes = []
    for i in order:
        nodes.append(move_to_allowed(graphs, metric, base["demand_cells"][i], allowed, set(nodes)))
    return {"nodes": nodes, "eval": evaluate(base, graphs, nodes, metric)}


def compute_scenario(base, params):
    start = time.perf_counter()
    graphs = get_graphs(base, params["mode"], params["closed"])
    cell_of = dict(zip(base["territory"]["settlements"]["id"], base["territory"]["settlements"]["cell"]))
    kinds, bases = {}, {}
    for kind, k in params["ks"]:
        fixed = [int(cell_of[i]) for t, i in params["fixed_centers"] if t == kind]
        if k == 0 and not fixed:
            continue
        kinds[kind] = place_kind(base, graphs, params, kind, k, fixed)
        bases[kind] = baseline(base, graphs, params, kind, max(1, k + len(fixed)))
    return {"id": params["id"], "params": params, "types": kinds, "bases": bases,
            "ms": (time.perf_counter() - start) * 1000}


def straight_line(base, cell_a, cell_b, points=120):
    grid, alt = base["territory"]["grid"], base["territory"]["relief"]["altitude"]
    fa, ca = divmod(int(cell_a), grid["nx"])
    fb, cb = divmod(int(cell_b), grid["nx"])
    t = np.linspace(0, 1, points)
    rows = np.round(fa + (fb - fa) * t).astype(int)
    cols = np.round(ca + (cb - ca) * t).astype(int)
    x, y, z = grid["x_km"][cols], grid["y_km"][rows], alt[rows, cols]
    plan_view = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    dz = np.diff(z)
    return {"x_km": x, "y_km": y, "altitude": z, "lat": grid["lat"][rows], "lon": grid["lon"][cols],
            "horizontal_km": plan_view,
            "length_3d_km": float(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + (dz / 1000) ** 2).sum()),
            "ascent_m": int(dz[dz > 0].sum()), "max_altitude_m": int(z.max()),
            "crosses_main_river": bool(base["territory"]["relief"]["main_river"][rows, cols].any())}


def route_of(base, scenario, kind, i):
    r = scenario["types"].get(kind)
    if r is None or i is None:
        return None
    ev = r["eval"]
    j = int(ev["assignment"][i])
    if j < 0:
        return None
    graphs = get_graphs(base, scenario["params"]["mode"], scenario["params"]["closed"])
    cell = int(base["demand_cells"][i])
    cost = np.full(len(ev["next"]), np.inf)
    cost[cell] = 0.0
    path = routes.path_to(ev["next"], cost, cell)
    measure = routes.measure_path(base["network"], graphs, path, base["territory"]["relief"]["altitude"])
    return {"community": base["demand"].iloc[i]["name"], "index": int(i), "path": path, **measure,
            "center": j + 1, "center_location": r["centers"][j]["location"],
            "straight": straight_line(base, cell, r["centers"][j]["node"])}


def farthest_route(base, scenario, kind):
    r = scenario["types"].get(kind)
    return None if r is None else route_of(base, scenario, kind, r["eval"]["farthest"])


def all_routes(base, scenario, kind):
    r = scenario["types"].get(kind)
    if r is None:
        return []
    ev = r["eval"]
    paths = []
    for i, cell in enumerate(base["demand_cells"]):
        if np.isfinite(ev[r["metric"]][i]):
            cost = np.zeros(len(ev["next"]))
            paths.append(routes.path_to(ev["next"], cost, int(cell)))
    return paths


def compare_strategies(base, scenario, kind, draws=20):
    params = scenario["params"]
    r = scenario["types"][kind]
    k_total = len(r["centers"])
    graphs = get_graphs(base, params["mode"], params["closed"])
    metric, unit = metric_of(params["criterion"])
    allowed = allowed_for(base, graphs, kind)
    truth = base["territory"]["labels"].ravel()
    fixed = [c["node"] for c in r["centers"] if c["existing"]]
    k_new = k_total - len(fixed)

    def row(name, nodes, ev=None):
        ev = ev or evaluate(base, graphs, nodes, metric)
        kp = ev["kpi"]
        unsuitable = sum(truth[n] == 2 for n in nodes)
        return {"Estrategia": name, f"Peor viaje ({unit})": kp["max_criterion"],
                "Peor tiempo (min)": kp["max_time_min"], "Peor distancia (km)": kp["max_km"],
                "Pobl. >25 km (%)": kp["pct_over_25km"], "Pobl. >2 h (%)": kp["pct_over_2h"],
                "Pobl. <1 h a pie (%)": kp["pct_under_1h_walk"], "Pobl. <8 km (%)": kp["pct_under_8km"],
                "Centros en terreno No apto": f"{unsuitable} de {len(nodes)}"}

    rows = [row("ACCESO con IA de terreno" + (" (refinado)" if params["refine"] else ""),
                [c["node"] for c in r["centers"]], r["eval"])]
    rules = k_centers.place_k_centers(graphs[metric], graphs[metric + "_T"], base["demand_cells"],
                                      base["population"], allowed, k_new, fixed=fixed)
    rule_nodes = [c["node"] for c in rules["centers"]]
    if params["refine"]:
        rows.append(row("ACCESO solo reglas del artículo (sin refinar)", rule_nodes))
    rows.append(row("Comunidades más pobladas", scenario["bases"][kind]["nodes"],
                    scenario["bases"][kind]["eval"]))

    rng = np.random.default_rng(config.SEED)
    random_nodes = []
    for _ in range(draws):
        chosen_list = rng.choice(len(base["demand_cells"]), size=k_total, replace=False)
        nodes = []
        for i in chosen_list:
            nodes.append(move_to_allowed(graphs, metric, base["demand_cells"][i], allowed, set(nodes)))
        random_nodes.append(row("", nodes))
    average = pd.DataFrame(random_nodes).drop(columns=["Estrategia", "Centros en terreno No apto"]).mean().to_dict()
    rows.append({"Estrategia": f"Al azar (promedio de {draws} sorteos)", **average,
                 "Centros en terreno No apto": "–"})

    without_ai = k_centers.place_k_centers(graphs[metric], graphs[metric + "_T"], base["demand_cells"],
                                           base["population"], ~base["not_buildable"], k_new, fixed=fixed)
    nodes_without_ai = [c["node"] for c in without_ai["centers"]]
    row_without_ai = row("ACCESO sin IA de terreno (ignora el relieve)", nodes_without_ai)
    rows.append(row_without_ai)

    D = get_matrix(base, params["mode"], params["closed"], metric)
    threshold = config.THRESHOLD_1H_MIN if metric == "time" else config.NEAR_THRESHOLD_KM
    coverage = k_centers.max_coverage(D, base["population"], allowed, k_total, threshold)
    rows.append(row(f"Cobertura máxima a {threshold} {unit} (criterio alternativo RF12)", coverage))

    table = pd.DataFrame(rows)
    with_rules = evaluate(base, graphs, rule_nodes, metric)["kpi"]["max_criterion"]
    without = row_without_ai[f"Peor viaje ({unit})"]
    fallen = sum(truth[n] == 2 for n in nodes_without_ai)
    summary = {
        "pct_without_ai_unsuitable": 100 * fallen / max(1, len(nodes_without_ai)),
        "pct_with_ai_unsuitable": 100 * sum(truth[c["node"]] == 2 for c in r["centers"]) / max(1, k_total),
        "cost_pct": 100 * (with_rules - without) / without if without > 0 else 0.0,
        "unit": unit,
    }
    return table, summary


if __name__ == "__main__":
    start = time.perf_counter()
    base = prepare_base()
    print(f"Base prepared in {time.perf_counter() - start:.1f} s")
    p = parameters()
    scn = compute_scenario(base, p)
    print(f"\n{scn['id']} computed in {scn['ms']:.0f} ms")
    for kind, r in scn["types"].items():
        kp, kb = r["eval"]["kpi"], scn["bases"][kind]["eval"]["kpi"]
        print(f"\n{kind}: {len(r['centers'])} centers · worst trip {kp['max_time_min']:.0f} min "
              f"({kp['max_km']:.1f} km) vs most populated {kb['max_time_min']:.0f} min · "
              f"<1 h walking {kp['pct_under_1h_walk']:.0f} % · rules {r['ms_rules']:.0f} ms")
        for c in r["centers"]:
            print(f"   {c['number']}. {c['location']} · {c['ai_suitability']} · slope {c['slope']:.0f} % · "
                  f"{c['altitude']:.0f} m · serves {c['communities']} communities")
    far = farthest_route(base, scn, "Centro de salud")
    print(f"\nLongest trip (health): {far['community']} -> center {far['center']}: "
          f"{far['minutes']:.0f} min, {far['km']:.1f} km, climbs {far['ascent_m']} m")
    table, n4 = compare_strategies(base, scn, "Centro de salud")
    print("\n" + table.round(1).to_string(index=False))
    print(f"\nN4: without AI, {n4['pct_without_ai_unsuitable']:.0f} % of the centers would fall on unsuitable terrain; "
          f"avoiding it costs {n4['cost_pct']:+.1f} % of worst trip.")
