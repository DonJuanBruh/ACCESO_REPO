import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import config
from data_generator import OFFSETS

DIRECTION_INDEX = {d: k for k, d in enumerate(OFFSETS)}


def tobler_speed(slope):
    return config.TOBLER_VMAX_KMH * np.exp(-3.5 * np.abs(slope + 0.05))


def prepare_network(territory):
    grid, relief, roads = territory["grid"], territory["relief"], territory["roads"]
    ny_, nx_ = grid["ny"], grid["nx"]
    n = ny_ * nx_
    idx = np.arange(n).reshape(ny_, nx_)
    alt = relief["altitude"].ravel()
    river = relief["main_river"]
    stream = relief["stream"].ravel()

    position = np.full((n, 8), -1, dtype=np.int64)
    origin, target, horizontal, blocked = [], [], [], []
    total = 0
    for k, (df, dc) in enumerate(OFFSETS):
        f0, f1, c0, c1 = max(0, -df), ny_ - max(0, df), max(0, -dc), nx_ - max(0, dc)
        a = idx[f0:f1, c0:c1].ravel()
        b = idx[f0 + df:f1 + df, c0 + dc:c1 + dc].ravel()
        blk = river[f0:f1, c0:c1].ravel() | river[f0 + df:f1 + df, c0 + dc:c1 + dc].ravel()
        if df != 0 and dc != 0:
            blk |= river[f0 + df:f1 + df, c0:c1].ravel() & river[f0:f1, c0 + dc:c1 + dc].ravel()
        position[a, k] = np.arange(total, total + len(a))
        total += len(a)
        origin.append(a)
        target.append(b)
        horizontal.append(np.full(len(a), math.hypot(df, dc) * grid["cell_km"]))
        blocked.append(blk)
    origin, target = np.concatenate(origin), np.concatenate(target)
    horizontal, blocked = np.concatenate(horizontal), np.concatenate(blocked)

    ascent_m = alt[target] - alt[origin]
    slope = ascent_m / (horizontal * 1000)
    length_3d = np.sqrt(horizontal ** 2 + (ascent_m / 1000) ** 2)
    v_pie = tobler_speed(slope) * config.OFF_ROAD_FACTOR
    v_pie = np.where(stream[origin] | stream[target], v_pie * config.STREAM_FACTOR, v_pie)

    segment_edges, road_cells = [], np.zeros(n, dtype=bool)
    for path in roads["cells"]:
        path = np.asarray(path)
        road_cells[path] = True
        df = path[1:] // nx_ - path[:-1] // nx_
        dc = path[1:] % nx_ - path[:-1] % nx_
        k = np.array([DIRECTION_INDEX[(int(a), int(b))] for a, b in zip(df, dc)], dtype=np.int64)
        outbound = position[path[:-1], k]
        turn = position[path[1:], 7 - k]
        segment_edges.append(np.concatenate([outbound, turn]))

    return {
        "n": n, "origin": origin, "destination": target, "position": position,
        "horizontal": horizontal, "slope": slope, "length_3d": length_3d,
        "blocked": blocked, "off_road_walk_minutes": horizontal / v_pie * 60,
        "segment_edges": segment_edges, "segment_type": roads["type"].tolist(),
        "road_cells": road_cells,
    }


def build_graphs(network, mode, closed=()):
    closed = set(int(c) for c in closed)
    is_road = np.zeros(len(network["origin"]), dtype=bool)
    vehicle_speed = np.zeros(len(network["origin"]))
    for t, edges in enumerate(network["segment_edges"]):
        if t in closed:
            continue
        is_road[edges] = True
        v = config.VEHICLE_SPEED_SECONDARY_KMH if network["segment_type"][t] == "secundaria" \
            else config.VEHICLE_SPEED_TERTIARY_KMH
        vehicle_speed[edges] = np.maximum(vehicle_speed[edges], v)

    passable = ~network["blocked"] | is_road
    walk = network["off_road_walk_minutes"].copy()
    walk_on_road = network["horizontal"] / tobler_speed(network["slope"]) * 60
    walk[is_road] = walk_on_road[is_road]

    if mode == config.TRAVEL_MODES[1]:
        travel_time = walk.copy()
        v = vehicle_speed[is_road] / (1 + config.VEHICLE_SLOPE_PENALTY * np.abs(network["slope"][is_road]))
        travel_time[is_road] = network["length_3d"][is_road] / v * 60
    else:
        travel_time = walk

    def matrix(weights):
        o, d, w = network["origin"][passable], network["destination"][passable], weights[passable]
        return csr_matrix((w, (o, d)), shape=(network["n"], network["n"]))

    graphs = {"time": matrix(travel_time), "walk": matrix(walk), "km": matrix(network["length_3d"])}
    for name in list(graphs):
        graphs[name + "_T"] = graphs[name].T.tocsr()
    graphs["mode"], graphs["closed"] = mode, tuple(sorted(closed))
    graphs["open_road_cells"] = np.zeros(network["n"], dtype=bool)
    for t, edges in enumerate(network["segment_edges"]):
        if t not in closed:
            graphs["open_road_cells"][network["origin"][edges]] = True
    return graphs


def cost_to(graphs, metric, centers):
    if len(centers) == 0:
        n = graphs[metric].shape[0]
        return np.full(n, np.inf), np.full(n, -9999), np.full(n, -9999)
    return dijkstra(graphs[metric + "_T"], indices=list(centers), min_only=True, return_predecessors=True)


def cost_from(graphs, metric, origin):
    return dijkstra(graphs[metric], indices=int(origin))


def path_to(next_node, cost, origin):
    if not np.isfinite(cost[origin]):
        return []
    path = [int(origin)]
    for _ in range(len(next_node)):
        nxt = int(next_node[path[-1]])
        if nxt < 0:
            break
        path.append(nxt)
    return path


def measure_path(network, graphs, path, altitude):
    if len(path) < 2:
        return {"minutes": 0.0, "km": 0.0, "ascent_m": 0, "descent_m": 0, "profile_km": [0.0], "profile_altitude": []}
    path = np.asarray(path)
    alt = altitude.ravel()[path]
    nx_ = altitude.shape[1]
    df = path[1:] // nx_ - path[:-1] // nx_
    dc = path[1:] % nx_ - path[:-1] % nx_
    k = np.array([DIRECTION_INDEX[(int(a), int(b))] for a, b in zip(df, dc)])
    edges = network["position"][path[:-1], k]
    minutes = np.asarray(graphs["time"][path[:-1], path[1:]]).ravel()
    dz = np.diff(alt)
    return {
        "minutes": float(minutes.sum()),
        "km": float(network["length_3d"][edges].sum()),
        "ascent_m": int(dz[dz > 0].sum()), "descent_m": int(-dz[dz < 0].sum()),
        "profile_km": np.concatenate([[0], np.cumsum(network["length_3d"][edges])]).round(3).tolist(),
        "profile_altitude": alt.round(0).tolist(),
    }


if __name__ == "__main__":
    import time

    import data_generator

    t = data_generator.generate_territory()
    start = time.perf_counter()
    network = prepare_network(t)
    g = build_graphs(network, config.TRAVEL_MODES[0])
    print(f"Network prepared in {(time.perf_counter() - start) * 1000:.0f} ms · "
          f"{g['time'].nnz} passable edges")

    villages = t["settlements"][t["settlements"]["type"] == "vereda"]
    a, b = villages.iloc[0], villages.iloc[-1]
    straight = math.hypot(a["x_km"] - b["x_km"], a["y_km"] - b["y_km"])
    start = time.perf_counter()
    cost, next_node, _ = cost_to(g, "time", [b["cell"]])
    ms = (time.perf_counter() - start) * 1000
    path = path_to(next_node, cost, a["cell"])
    m = measure_path(network, g, path, t["relief"]["altitude"])
    print(f"{a['name']} -> {b['name']}")
    print(f"  Straight line: {straight:.1f} km")
    print(f"  Over the terrain: {m['km']:.1f} km, {m['minutes'] / 60:.1f} h walking, "
          f"climbs {m['ascent_m']} m and descends {m['descent_m']} m (Dijkstra in {ms:.1f} ms)")

    g2 = build_graphs(network, config.TRAVEL_MODES[1])
    c2, s2, _ = cost_to(g2, "time", [b["cell"]])
    print(f"  By vehicle on road + walking: {c2[a['cell']] / 60:.1f} h")

    with_bridge = t["roads"].index[t["roads"]["bridges"] > 0].tolist()
    g3 = build_graphs(network, config.TRAVEL_MODES[1], with_bridge)
    c3, _, _ = cost_to(g3, "time", [b["cell"]])
    print(f"  If all {len(with_bridge)} bridges fail: "
          + ("no route" if np.isinf(c3[a['cell']]) else f"{c3[a['cell']] / 60:.1f} h"))
