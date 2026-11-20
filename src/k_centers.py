import itertools

import numpy as np
from scipy.sparse.csgraph import dijkstra


def _default_describe(node):
    return f"nodo {node}"


def place_k_centers(graph, graph_T, demand, population, allowed, k, fixed=(),
                    names=None, describe=None, unit="min"):
    demand = np.asarray(demand)
    population = np.asarray(population, dtype=float)
    names = names if names is not None else [f"Comunidad {i}" for i in range(len(demand))]
    describe = describe or _default_describe
    cost = np.full(len(demand), np.inf)
    assignment = np.full(len(demand), -1)
    centers, steps = [], []
    count = {"dijkstras": 0}

    def text(c):
        return "sin ruta a ningún centro" if np.isinf(c) else f"a {c:.1f} {unit}"

    def add(node, reason, moved_from=None, displacement=0.0):
        new_costs = dijkstra(graph_T, indices=int(node))[demand]
        count["dijkstras"] += 1
        improvement = new_costs < cost
        cost[improvement] = new_costs[improvement]
        assignment[improvement] = len(centers)
        centers.append({"node": int(node), "reason": reason, "moved_from": moved_from,
                        "displacement": float(displacement), "fixed": reason.startswith("Centro existente")})
        steps.append(f"Centro {len(centers)} ({describe(node)}): {reason}")

    def nearest_suitable(i):
        from_cost = dijkstra(graph, indices=int(demand[i]))
        count["dijkstras"] += 1
        from_cost = np.where(allowed, from_cost, np.inf)
        for c in centers:
            from_cost[c["node"]] = np.inf
        j = int(np.argmin(from_cost))
        return j, float(from_cost[j])

    for node in fixed:
        add(node, "Centro existente (se conserva y se tiene en cuenta).")

    target_value = len(fixed) + int(k)
    if k > 0 and not np.any(allowed):
        steps.append("No hay ningún terreno permitido: no se puede ubicar ningún centro.")
        target_value = len(fixed)

    if len(centers) < target_value and not fixed:
        suitable_cells = np.flatnonzero(allowed[demand])
        if len(suitable_cells):
            i = int(suitable_cells[np.argmax(population[suitable_cells])])
            add(demand[i], f"Regla de inicio: {names[i]} es la comunidad apta más poblada "
                                f"({int(population[i])} hab).")
        else:
            i = int(np.argmax(population))
            j, d = nearest_suitable(i)
            add(j, f"Regla de inicio: {names[i]} es la más poblada, pero ninguna comunidad está en "
                       f"terreno apto; el centro va al terreno apto más cercano ({d:.1f} {unit}).", i, d)

    no_improvement = set()
    while len(centers) < target_value:
        candidates = [i for i in range(len(demand)) if cost[i] > 0 and i not in no_improvement]
        if not candidates:
            steps.append("Todas las comunidades ya tienen un centro tan cerca como el terreno permite.")
            break
        farthest = max(candidates, key=lambda i: (cost[i], population[i]))
        name = names[farthest]
        if allowed[demand[farthest]] and all(c["node"] != demand[farthest] for c in centers):
            add(demand[farthest], f"{name} era la comunidad más alejada ({text(cost[farthest])}) "
                                     "y su terreno es apto.")
            continue

        j, d = nearest_suitable(farthest)
        if not np.isfinite(d) or d >= cost[farthest]:
            no_improvement.add(farthest)
            steps.append(f"{name} ({text(cost[farthest])}) no es apta y ningún terreno apto la acerca más: "
                         "se pasa a la siguiente más alejada.")
            continue
        add(j, f"La más alejada era {name} ({text(cost[farthest])}), pero su terreno no es apto; "
                   f"el centro se corrió {d:.1f} {unit} por el camino hasta el terreno apto más cercano.",
            farthest, d)

    return {"centers": centers, "cost": cost, "assignment": assignment, "steps": steps,
            "dijkstras": count["dijkstras"]}


def demand_matrix(graph, demand):
    return dijkstra(graph, indices=np.asarray(demand))


def refine_centers(D, center_nodes, allowed, fixed_mask=None, center_names=None, unit="min",
                   describe=None, max_iter=8):
    describe = describe or _default_describe
    centers = [int(c) for c in center_nodes]
    fixed_mask = fixed_mask if fixed_mask is not None else [False] * len(centers)
    moves = []
    if not centers:
        return centers, np.full(D.shape[0], -1), np.full(D.shape[0], np.inf), moves
    for _ in range(max_iter):
        sub = D[:, centers]
        assignment = np.argmin(sub, axis=1)
        moved_flag = False
        for j in range(len(centers)):
            members = assignment == j
            if fixed_mask[j] or not members.any() or not np.isfinite(sub[members, j]).all():
                continue
            current = sub[members, j].max()
            worst = D[members].max(axis=0)
            worst = np.where(allowed, worst, np.inf)
            worst[[c for i, c in enumerate(centers) if i != j]] = np.inf
            best = int(np.argmin(worst))
            if worst[best] < current - 1e-6:
                moves.append(f"Centro {j + 1}: de {describe(centers[j])} a {describe(best)}; "
                             f"el peor viaje de su grupo bajó de {current:.1f} a {worst[best]:.1f} {unit}.")
                centers[j] = best
                sub = D[:, centers]
                moved_flag = True
        if not moved_flag:
            break
    sub = D[:, centers]
    return centers, np.argmin(sub, axis=1), sub.min(axis=1), moves


def max_coverage(D, population, allowed, k, threshold):
    covers = (D <= threshold) & allowed[None, :]
    covered = np.zeros(D.shape[0], dtype=bool)
    selected = []
    for _ in range(int(k)):
        gain = (covers & ~covered[:, None]).T.astype(float) @ population
        gain[selected] = -1
        if gain.max() <= 0:
            pending = ~covered if (~covered).any() else np.ones_like(covered)
            worst = np.where(allowed, D[pending].max(axis=0), np.inf)
            worst[selected] = np.inf
            j = int(np.argmin(worst))
        else:
            j = int(np.argmax(gain))
        selected.append(j)
        covered |= covers[:, j]
    return selected


def brute_force_optimum(D_nodes, k):
    best, combination = np.inf, None
    for combo in itertools.combinations(range(D_nodes.shape[0]), k):
        value = D_nodes[:, combo].min(axis=1).max()
        if value < best:
            best, combination = value, combo
    return best, combination


if __name__ == "__main__":
    from scipy.sparse import csr_matrix
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(7)
    points = rng.uniform(0, 30, (14, 2))
    edges = {tuple(sorted((a, b))) for t in Delaunay(points).simplices for a, b in itertools.combinations(t, 2)}
    o, d = np.array(list(edges)).T
    w = np.linalg.norm(points[o] - points[d], axis=1) * rng.uniform(1.15, 1.6, len(o))
    g = csr_matrix((np.r_[w, w], (np.r_[o, d], np.r_[d, o])), shape=(14, 14))
    nodes = np.arange(14)
    res = place_k_centers(g, g, nodes, rng.integers(50, 500, 14), np.ones(14, bool), 3, unit="km")
    print("\n".join(res["steps"]))
    optimum, _ = brute_force_optimum(dijkstra(g), 3)
    print(f"\nACCESO worst distance: {res['cost'].max():.1f} km · optimum: {optimum:.1f} km · "
          f"factor {res['cost'].max() / optimum:.2f} (theoretical bound: 2)")
