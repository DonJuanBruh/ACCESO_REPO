import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree

import config

VILLAGE_NAMES = [
    "El Cedro", "La Palma", "San Isidro", "El Carmen", "Buenavista", "La Esperanza",
    "El Roble", "Santa Bárbara", "Monserrate", "El Tablón", "Las Mercedes", "Pueblo Viejo",
    "La Laguna", "El Hato", "San Antonio", "Agua Blanca", "El Mortiño", "La Capilla",
    "Peña Negra", "El Salitre", "Tierra Negra", "La Chorrera", "El Arrayán", "San Rafael",
    "Los Pinos", "La Cumbre", "El Encenillo", "Quebrada Honda", "La Florida", "El Volcán",
    "San José", "Alto Grande", "El Pedregal", "La Vega", "Hato Viejo", "El Curubal",
    "Santa Rosa", "Los Alisos", "La Mesa", "El Chital", "Guayabal", "Piedra Gorda",
    "El Tunal", "La Playa", "San Pedro", "El Uvo", "Las Juntas", "Loma Alta",
    "El Páramo", "La Carbonera", "Siete Cueros", "El Frailejón", "Monte Redondo",
    "La Estancia", "El Retiro", "Corinto", "El Morro", "La Cabaña", "Caño Grande",
    "El Guamo", "Mundo Nuevo", "La Ceiba", "El Banco", "Sabanalarga", "El Higuerón",
    "Toquilla", "La Unión", "El Porvenir", "Canelas", "Miraflores", "El Venado",
]

MAIN_RIVERS = [
    {"name": "Río Cravo Norte (sintético)",
     "points": [(5.80, -72.62), (5.73, -72.55), (5.66, -72.51), (5.58, -72.49),
                (5.50, -72.45), (5.41, -72.40), (5.33, -72.36)],
     "valley_start": 2500, "valley_end": 430},
    {"name": "Río Cusiana (sintético)",
     "points": [(5.47, -72.80), (5.40, -72.76), (5.33, -72.72), (5.27, -72.68),
                (5.22, -72.61)],
     "valley_start": 2300, "valley_end": 700},
]


def create_grid():
    lat0 = math.radians((config.LAT_MIN + config.LAT_MAX) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat0)
    cell_km = config.CELL_RESOLUTION_M / 1000
    width_km = (config.LON_MAX - config.LON_MIN) * km_per_deg_lon
    height_km = (config.LAT_MAX - config.LAT_MIN) * km_per_deg_lat
    nx_ = int(round(width_km / cell_km)) + 1
    ny_ = int(round(height_km / cell_km)) + 1
    x_km = np.arange(nx_) * cell_km
    y_km = np.arange(ny_) * cell_km
    return {
        "nx": nx_, "ny": ny_, "cell_km": cell_km,
        "x_km": x_km, "y_km": y_km,
        "lon": config.LON_MIN + x_km / km_per_deg_lon,
        "lat": config.LAT_MIN + y_km / km_per_deg_lat,
        "km_per_degree_lat": km_per_deg_lat, "km_per_degree_lon": km_per_deg_lon,
    }


def to_km(grid, lat, lon):
    x = (np.asarray(lon) - config.LON_MIN) * grid["km_per_degree_lon"]
    y = (np.asarray(lat) - config.LAT_MIN) * grid["km_per_degree_lat"]
    return x, y


def cell_of(grid, x_km, y_km):
    col = int(np.clip(round(x_km / grid["cell_km"]), 0, grid["nx"] - 1))
    row = int(np.clip(round(y_km / grid["cell_km"]), 0, grid["ny"] - 1))
    return row, col, row * grid["nx"] + col


def smooth_field(rng, shape, sigma):
    noise = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return (noise - noise.mean()) / (noise.std() + 1e-9)


def densify(points_km, step_km=0.08):
    output = []
    for (x1, y1), (x2, y2) in zip(points_km[:-1], points_km[1:]):
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / step_km))
        t = np.linspace(0, 1, n, endpoint=False)
        output.append(np.column_stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]))
    output.append(np.array([points_km[-1]]))
    return np.vstack(output)


def meander(points_km, rng, amplitude_km=0.9):
    line = densify(points_km, 0.5)
    direction = np.gradient(line, axis=0)
    normal = np.column_stack([-direction[:, 1], direction[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
    wave = ndimage.gaussian_filter1d(rng.standard_normal(len(line)), 3) * amplitude_km * 2.5
    wave[0] = wave[-1] = 0
    return densify(list(map(tuple, line + normal * wave[:, None])), 0.08)


def generate_relief(grid, rng):
    ny_, nx_ = grid["ny"], grid["nx"]
    X, Y = np.meshgrid(grid["x_km"], grid["y_km"])
    xn, yn = X / X.max(), Y / Y.max()

    tilt = np.clip(0.72 * (1 - xn) + 0.28 * yn, 0, 1) ** 1.15
    altitude = config.ALTITUDE_PLAINS + (config.ALTITUDE_PARAMO - config.ALTITUDE_PLAINS) * tilt

    for amplitude, scale in zip(config.MOUNTAIN_AMPLITUDE_M, config.MOUNTAIN_SCALE_CELLS):
        altitude += amplitude * (0.7 + 0.3 * tilt) * smooth_field(rng, (ny_, nx_), scale)

    river_points, riverbed, lines = [], [], []
    for river in MAIN_RIVERS:
        lat, lon = zip(*river["points"])
        xs, ys = to_km(grid, lat, lon)
        line = meander(list(zip(xs, ys)), rng)
        river_points.append(line)
        riverbed.append(np.linspace(river["valley_start"], river["valley_end"], len(line)))
        lines.append({"name": river["name"], "type": "principal", "x_km": line[:, 0], "y_km": line[:, 1]})
    river_points = np.vstack(river_points)
    riverbed = np.concatenate(riverbed)
    main_tree = cKDTree(river_points)
    cells_km = np.column_stack([X.ravel(), Y.ravel()])
    d_main, nearest = main_tree.query(cells_km)
    d_main = d_main.reshape(ny_, nx_)
    weight = np.exp(-(d_main / config.MAIN_VALLEY_WIDTH_KM) ** 2)
    background = riverbed[nearest].reshape(ny_, nx_)
    altitude = altitude * (1 - weight) + np.minimum(altitude, background) * weight

    high = np.argwhere((tilt > 0.45) & (d_main > 6))
    stream_points = []
    for i in range(config.NUM_STREAMS):
        row, col = high[rng.integers(len(high))]
        start = (grid["x_km"][col], grid["y_km"][row])
        _, j = main_tree.query(start)
        end = tuple(river_points[j])
        middle = ((start[0] + end[0]) / 2 + rng.normal(0, 2), (start[1] + end[1]) / 2 + rng.normal(0, 2))
        line = meander([start, middle, end], rng, 0.5)
        stream_points.append(line)
        lines.append({"name": f"Quebrada {i + 1}", "type": "stream", "x_km": line[:, 0], "y_km": line[:, 1]})
    d_stream = cKDTree(np.vstack(stream_points)).query(cells_km)[0].reshape(ny_, nx_)
    altitude -= config.STREAM_VALLEY_DEPTH_M * np.exp(-(d_stream / config.STREAM_VALLEY_WIDTH_KM) ** 2) \
        * (0.4 + 0.6 * tilt)

    altitude = ndimage.gaussian_filter(altitude, 0.8)
    altitude = np.clip(altitude, 250, 4100)

    cell = grid["cell_km"]
    return {
        "altitude": altitude,
        "main_river": d_main < 0.75 * cell,
        "stream": (d_stream < 0.55 * cell) & ~(d_main < 0.75 * cell),
        "river_dist_m": np.minimum(d_main, d_stream) * 1000,
        "river_lines": lines,
    }


def derive_layers(relief, grid):
    alt = relief["altitude"]
    step_m = grid["cell_km"] * 1000
    dz_dy, dz_dx = np.gradient(alt, step_m)
    relief["slope"] = 100 * np.hypot(dz_dx, dz_dy)
    smooth = ndimage.gaussian_filter(alt, 1.5)
    relief["curvature"] = -ndimage.laplace(smooth) / (grid["cell_km"] ** 2)
    relief["tpi"] = alt - ndimage.uniform_filter(alt, size=9, mode="nearest")
    relief["orientation"] = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360) % 360
    return relief


def generate_simma(relief, grid, rng):
    slope_val = relief["slope"]
    prob = config.LANDSLIDE_BASE_RATE * np.exp((slope_val - 22) / 7)
    prob *= np.where(relief["river_dist_m"] < 600, 1.8, 1.0)
    prob *= np.where(relief["curvature"] > 0, 1.3, 1.0)
    prob = np.clip(prob, 0, 0.55)
    event = rng.uniform(0, 1, prob.shape) < prob
    rows, cols = np.nonzero(event)
    kinds = ["Deslizamiento traslacional", "Deslizamiento rotacional", "Flujo de detritos", "Caída de rocas"]
    middle = grid["cell_km"] / 2
    events = pd.DataFrame({
        "lat": grid["lat"][rows] + rng.uniform(-middle, middle, len(rows)) / grid["km_per_degree_lat"],
        "lon": grid["lon"][cols] + rng.uniform(-middle, middle, len(rows)) / grid["km_per_degree_lon"],
        "type": rng.choice(kinds, len(rows)),
        "year": rng.integers(1990, 2025, len(rows)),
        "cell": rows * grid["nx"] + cols,
    })
    return event, events


def label_terrain(relief, simma_event, rng):
    p, alt, d, tpi = relief["slope"], relief["altitude"], relief["river_dist_m"], relief["tpi"]
    is_river = relief["main_river"] | relief["stream"]
    unsuitable = ((p > config.SLOPE_UNSUITABLE) | (alt > config.ALTITUDE_UNSUITABLE) | (d < config.RIVER_UNSUITABLE_M)
                  | (tpi > config.PEAK_UNSUITABLE_M) | simma_event | is_river)
    conditional = ~unsuitable & ((p > config.SLOPE_CONDITIONAL) | (alt > config.ALTITUDE_CONDITIONAL)
                                 | (d < config.RIVER_CONDITIONAL_M) | (tpi > config.PEAK_CONDITIONAL_M))
    label = np.where(unsuitable, 2, np.where(conditional, 1, 0))
    wrong = rng.uniform(0, 1, label.shape) < config.LABEL_NOISE
    noise = (label + rng.integers(1, 3, label.shape)) % 3
    return np.where(wrong & ~is_river, noise, label)


def place_settlements(relief, grid, rng):
    names = list(rng.permutation(VILLAGE_NAMES))
    X, Y = np.meshgrid(grid["x_km"], grid["y_km"])
    rows = []

    town_xy = {}
    for municipality, (lat, lon, urban, _) in config.TOWN_CENTERS.items():
        x, y = to_km(grid, lat, lon)
        town_xy[municipality] = (float(x), float(y))
        f, c, idx = cell_of(grid, x, y)
        rows.append({"name": municipality + " (cabecera)", "type": "cabecera", "municipality": municipality,
                     "population": urban, "cell": idx})

    centers = np.array(list(town_xy.values()))
    municipality_of = np.argmin(((X.ravel()[:, None] - centers[:, 0]) ** 2
                                 + (Y.ravel()[:, None] - centers[:, 1]) ** 2), axis=1).reshape(X.shape)

    habitable = np.exp(-relief["slope"] / 11) * (relief["altitude"] < 3250) \
        * ~relief["main_river"] * ~relief["stream"] * (relief["river_dist_m"] > 250)
    occupied = [town_xy[m] for m in town_xy]
    for m, (municipality, (_, _, _, rural)) in enumerate(config.TOWN_CENTERS.items()):
        weight = (habitable * (municipality_of == m)).ravel()
        populations = np.maximum(25, rng.dirichlet(np.full(config.VILLAGES_PER_MUNICIPALITY, 2.5)) * rural).round()
        placed = 0
        for _ in range(4000):
            if placed == config.VILLAGES_PER_MUNICIPALITY:
                break
            idx = rng.choice(len(weight), p=weight / weight.sum())
            x, y = X.ravel()[idx], Y.ravel()[idx]
            if min(math.hypot(x - ox, y - oy) for ox, oy in occupied) < config.MIN_VILLAGE_SEPARATION_KM:
                continue
            occupied.append((x, y))
            rows.append({"name": "Vereda " + names.pop(), "type": "vereda", "municipality": municipality,
                         "population": int(populations[placed]), "cell": int(idx)})
            placed += 1

    f, c, idx = cell_of(grid, 0.0, float(to_km(grid, 5.70, config.LON_MIN)[1]))
    rows.append({"name": "Salida a Sogamoso", "type": "salida", "municipality": "Sogamoso",
                 "population": 0, "cell": idx})

    settlements = pd.DataFrame(rows)
    settlements.insert(0, "id", range(len(settlements)))
    settlements["row"] = settlements["cell"] // grid["nx"]
    settlements["col"] = settlements["cell"] % grid["nx"]
    settlements["x_km"] = grid["x_km"][settlements["col"]]
    settlements["y_km"] = grid["y_km"][settlements["row"]]
    settlements["lat"] = grid["lat"][settlements["row"]].round(5)
    settlements["lon"] = grid["lon"][settlements["col"]].round(5)
    settlements["altitude"] = relief["altitude"].ravel()[settlements["cell"]].round(0).astype(int)
    return settlements


OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def grid_graph(grid, cell_cost):
    ny_, nx_ = grid["ny"], grid["nx"]
    idx = np.arange(ny_ * nx_).reshape(ny_, nx_)
    origin, target, weight = [], [], []
    for df, dc in OFFSETS:
        f0, f1 = max(0, -df), ny_ - max(0, df)
        c0, c1 = max(0, -dc), nx_ - max(0, dc)
        a = idx[f0:f1, c0:c1].ravel()
        b = idx[f0 + df:f1 + df, c0 + dc:c1 + dc].ravel()
        length = math.hypot(df, dc) * grid["cell_km"]
        origin.append(a)
        target.append(b)
        weight.append(length * (cell_cost.ravel()[a] + cell_cost.ravel()[b]) / 2)
    n = ny_ * nx_
    return csr_matrix((np.concatenate(weight), (np.concatenate(origin), np.concatenate(target))), shape=(n, n))


def rebuild_path(predecessors, target):
    path = [target]
    while predecessors[path[-1]] >= 0:
        path.append(predecessors[path[-1]])
    return path[::-1]


def generate_road_network(relief, grid, settlements, rng):
    cost = 1 + (relief["slope"] / 9) ** 2 + 6 * (relief["altitude"] > config.ALTITUDE_UNSUITABLE)
    cost = cost + 3 * relief["stream"]
    cost = np.where(relief["main_river"], config.BRIDGE_COST, cost)
    graph = grid_graph(grid, cost)

    cells = settlements["cell"].to_numpy()
    cost_between, predecessors = dijkstra(graph, directed=False, indices=cells, return_predecessors=True)
    cost_pp = cost_between[:, cells]

    network = nx.Graph()
    network.add_nodes_from(range(len(settlements)))
    for tri in Delaunay(settlements[["x_km", "y_km"]].to_numpy()).simplices.tolist():
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            network.add_edge(a, b, cost=cost_pp[a, b])
    cheapest_face = sorted(network.edges, key=lambda e: network.edges[e]["cost"], reverse=True)
    for a, b in cheapest_face[: int(len(cheapest_face) * config.REMOVED_ROADS_FRACTION)]:
        data = network.edges[a, b]
        network.remove_edge(a, b)
        if not nx.is_connected(network):
            network.add_edge(a, b, **data)

    main_items = settlements.index[settlements["type"].isin(["cabecera", "salida"])].tolist()
    full = nx.Graph()
    for i, a in enumerate(main_items):
        for b in main_items[i + 1:]:
            full.add_edge(a, b, cost=cost_pp[a, b])
    secondary = {tuple(sorted(e)) for e in nx.minimum_spanning_edges(full, weight="cost", data=False)}
    for a, b in secondary:
        network.add_edge(a, b, cost=cost_pp[a, b])

    alt = relief["altitude"].ravel()
    river = relief["main_river"].ravel()
    names = settlements["name"].str.replace("Vereda ", "", regex=False).str.replace(" (cabecera)", "", regex=False)
    segments = []
    for a, b in network.edges:
        path = np.array(rebuild_path(predecessors[a], cells[b]))
        r_idx, col = path // grid["nx"], path % grid["nx"]
        horizontal = np.hypot(np.diff(r_idx), np.diff(col)) * grid["cell_km"]
        ascent = np.diff(alt[path])
        length_3d = np.sqrt(horizontal ** 2 + (ascent / 1000) ** 2).sum()
        in_river = river[path].astype(int)
        bridges = int(((np.diff(np.concatenate([[0], in_river])) == 1)).sum())
        kind = "secundaria" if tuple(sorted((a, b))) in secondary else "terciaria"
        segments.append({
            "origin": int(a), "destination": int(b),
            "name": f"{names[a]} ↔ {names[b]}" + (" 🌉" if bridges else ""),
            "type": kind,
            "length_km": round(float(length_3d), 2),
            "elevation_diff_m": int(np.abs(ascent).sum()),
            "max_altitude_m": int(alt[path].max()),
            "bridges": bridges,
            "cells": path,
            "status": "abierta",
        })
    roads = pd.DataFrame(segments)
    roads.insert(0, "id", range(len(roads)))
    return roads


def generate_territory(seed=config.SEED):
    rng = np.random.default_rng(seed)
    grid = create_grid()
    relief = derive_layers(generate_relief(grid, rng), grid)
    event, events = generate_simma(relief, grid, rng)
    relief["simma"] = event
    settlements = place_settlements(relief, grid, rng)
    roads = generate_road_network(relief, grid, settlements, rng)
    labels = label_terrain(relief, event, rng)
    return {"grid": grid, "relief": relief, "simma": events, "settlements": settlements,
            "roads": roads, "labels": labels}


if __name__ == "__main__":
    import time
    start = time.perf_counter()
    t = generate_territory()
    m, r = t["grid"], t["relief"]
    print(f"Territory generated in {time.perf_counter() - start:.1f} s")
    print(f"Grid: {m['ny']} x {m['nx']} = {m['ny'] * m['nx']} cells of {config.CELL_RESOLUTION_M} m")
    print(f"Altitude: {r['altitude'].min():.0f} to {r['altitude'].max():.0f} m · "
          f"median slope {np.median(r['slope']):.1f} %")
    print(f"Synthetic SIMMA events: {len(t['simma'])}")
    class_list, counts = np.unique(t["labels"], return_counts=True)
    print("Ground truth:", {config.CLASSES[c]: f"{n / t['labels'].size:.0%}" for c, n in zip(class_list, counts)})
    p = t["settlements"]
    print(f"\nSettlements: {len(p)} ({(p['type'] == 'vereda').sum()} villages) · "
          f"population {p['population'].sum()} inhabitants")
    print(p.drop(columns=["cell", "row", "col"]).head(6).to_string(index=False))
    v = t["roads"]
    print(f"\nRoad segments: {len(v)} · {v['length_km'].sum():.0f} km · with bridge: {(v['bridges'] > 0).sum()}")
    print(v.drop(columns=["cells"]).head(6).to_string(index=False))
