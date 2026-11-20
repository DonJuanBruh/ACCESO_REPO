import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import scenarios

CLASS_COLORS = {0: "#2e9e5b", 1: "#e0b100", 2: "#d64545"}
GROUP_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
RELIEF_SCALE = [[0.0, "#5a8f4e"], [0.25, "#a7c47a"], [0.5, "#e6d79a"], [0.7, "#c49a6c"],
                [0.88, "#9c8b7a"], [1.0, "#f4f4f4"]]
CLASS_SCALE = [[0, CLASS_COLORS[0]], [1 / 3, CLASS_COLORS[0]], [1 / 3, CLASS_COLORS[1]], [2 / 3, CLASS_COLORS[1]],
               [2 / 3, CLASS_COLORS[2]], [1, CLASS_COLORS[2]]]
MAP_HEIGHT = 640
HEIGHT_3D = 700
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True,
                 "toImageButtonOptions": {"format": "png", "filename": "acceso", "scale": 2}}
DATA_LABELS = {"name": "nombre", "type": "tipo", "municipality": "municipio", "population": "poblacion",
               "altitude": "altura", "origin": "origen", "destination": "destino", "length_km": "longitud_km",
               "elevation_diff_m": "desnivel_m", "max_altitude_m": "altura_max_m", "bridges": "puentes",
               "status": "estado", "year": "anio"}
fragment = getattr(st, "fragment", lambda func: func)


def state():
    return st.session_state["acceso"]


def short(name):
    return str(name).replace("Vereda ", "").replace(" (cabecera)", "")


def fmt(x, decimals=1, unit=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "sin ruta"
    return f"{x:,.{decimals}f}{(' ' + unit) if unit else ''}"


def sidebar(base):
    roads = base["territory"]["roads"]
    settlements = base["territory"]["settlements"]
    with st.sidebar:
        st.markdown("### 🧭 ACCESO")
        st.caption("Provincia de La Libertad, Boyacá · ODS 10 · datos sintéticos de prueba")

        st.markdown("**Centros nuevos a ubicar**")
        ks = {}
        for kind, rule in config.CENTER_TYPES.items():
            ks[kind] = st.number_input(f"{rule['emoji']} {kind}", 0, config.K_MAX_PER_KIND, rule["k"],
                                       key=f"k_{kind}",
                                       help=f"Terreno permitido: {', '.join(config.CLASSES[c] for c in rule['allowed_classes'])}"
                                            + (" y sobre la red vial." if rule["requires_road"] else "."))
        mode = st.radio("Modo de viaje", config.TRAVEL_MODES, index=1, key="mode",
                        help="A pie usa la función de Tobler (la pendiente frena). Vehículo: carro o moto por "
                             "la vía y a pie desde la vía hasta la casa.")
        criterion = st.radio("Criterio que se minimiza", config.CRITERIA, index=0, key="criterion",
                             help="Se minimiza el PEOR viaje (equidad). El tiempo considera subidas y bajadas.")
        refine = st.toggle("Refinar hacia el punto céntrico apto", value=True, key="refine",
                           help="Después de las reglas del artículo, mueve cada centro a la celda APTA que "
                                 "minimiza el peor viaje de su grupo. Nunca a una cima ni a una ladera no apta.")

        st.markdown("**Cierre de vías** (derrumbe o puente caído)")
        order = roads.sort_values(["bridges", "type", "name"], ascending=[False, True, True]).index.tolist()
        closed = st.multiselect(
            "Vías cerradas", order, key="closed", label_visibility="collapsed",
            format_func=lambda i: f"{roads.loc[i, 'name']} · {roads.loc[i, 'length_km']:.1f} km",
            placeholder="Elige tramos (🌉 = tiene puente)")

        fixed = []
        with st.expander("Centros existentes (opcional)"):
            st.caption("Se conservan y el algoritmo ubica los nuevos teniéndolos en cuenta.")
            options = settlements[settlements["population"] > 0]["id"].tolist()
            for kind in config.CENTER_TYPES:
                selected = st.multiselect(kind, options, key=f"fixed_{kind}",
                                          format_func=lambda i: settlements.loc[i, "name"])
                fixed += [(kind, i) for i in selected]
        st.caption("🔌 Todo el cálculo es local: funciona sin internet.")
    return scenarios.parameters(ks, mode, criterion, refine, closed, fixed)


def kind_selector(key, label="Ver resultados de"):
    kinds = list(state()["scenario"]["types"])
    if not kinds:
        st.warning("Pon al menos un centro en la barra lateral.")
        return None
    return st.radio(label, kinds, horizontal=True, key=key,
                    format_func=lambda t: f"{config.CENTER_TYPES[t]['emoji']} {t}")


def kpi_cards(scenario, kind, base):
    r = scenario["types"][kind]
    kp, kb = r["eval"]["kpi"], scenario["bases"][kind]["eval"]["kpi"]
    unit = r["unit"]
    c = st.columns(5)
    with c[0].container(border=True):
        st.metric("Peor viaje", fmt(kp["max_time_min"], 0, "min"),
                  delta=f"{kp['max_time_min'] - kb['max_time_min']:+.0f} min vs. más pobladas",
                  delta_color="inverse", help="Tiempo de la comunidad más alejada hasta su centro, por el terreno real.")
    with c[1].container(border=True):
        st.metric("Peor distancia por camino", fmt(kp["max_km"], 1, "km"),
                  delta=f"{kp['max_km'] - kb['max_km']:+.1f} km", delta_color="inverse",
                  help="Kilómetros 3D por el camino más corto (no en línea recta).")
    with c[2].container(border=True):
        st.metric("Población a >2 h", fmt(kp["pct_over_2h"], 1, "%"),
                  delta=f"{kp['pct_over_2h'] - kb['pct_over_2h']:+.1f} p. p.", delta_color="inverse",
                  help="Situación extrema (KPI N2) en el modo de viaje elegido; incluye a quien queda sin ruta.")
    with c[3].container(border=True):
        st.metric("Población a <1 h a pie", fmt(kp["pct_under_1h_walk"], 1, "%"),
                  delta=f"{kp['pct_under_1h_walk'] - kb['pct_under_1h_walk']:+.1f} p. p.",
                  help="Cobertura a pie (KPI N3). Minimizar el peor viaje no equivale a maximizar cobertura.")
    with c[4].container(border=True):
        st.metric("Terreno apto para este tipo", fmt(r["allowed_pct"], 0, "%"),
                  help="Porcentaje del territorio donde la IA de terreno permite construir este tipo de centro.")
    st.caption(f"Escenario **{scenario['id']}** · criterio en {unit} · ubicación calculada en "
               f"{r['ms_total']:.0f} ms ({r['dijkstras']} Dijkstra) · escenario completo en {scenario['ms']:.0f} ms")


def closure_notice(base, scenario, without_closures, kind):
    roads = base["territory"]["roads"]
    closed = scenario["params"]["closed"]
    a = without_closures["types"][kind]["eval"]["kpi"]
    d = scenario["types"][kind]["eval"]["kpi"]
    before = {c["location"] for c in without_closures["types"][kind]["centers"]}
    now = [c["location"] for c in scenario["types"][kind]["centers"]]
    text = (f"**Efecto de cerrar {len(closed)} vía(s)** ({', '.join(roads.loc[c, 'name'] for c in closed)}): "
            f"el peor viaje pasó de {fmt(a['max_time_min'], 0, 'min')} a {fmt(d['max_time_min'], 0, 'min')}. ")
    new_items = [u for u in now if u not in before]
    text += ("La ubicación se recalculó; centros nuevos: " + "; ".join(new_items) + ".") if new_items \
        else "Los centros propuestos no cambiaron."
    no_route = base["demand"]["name"][~np.isfinite(scenario["types"][kind]["eval"]["time"])].tolist()
    if no_route:
        text += " **Sin ruta a ningún centro:** " + ", ".join(short(n) for n in no_route) + "."
    st.info(text)


def cell_coordinates(grid, cells, smooth_array=True):
    cells = np.asarray(cells)
    f, c = cells // grid["nx"], cells % grid["nx"]
    lon, lat, x, y = grid["lon"][c], grid["lat"][f], grid["x_km"][c], grid["y_km"][f]
    if smooth_array and len(cells) > 4:
        k = np.ones(3) / 3
        lon[1:-1], lat[1:-1] = np.convolve(lon, k, "valid"), np.convolve(lat, k, "valid")
        x[1:-1], y[1:-1] = np.convolve(x, k, "valid"), np.convolve(y, k, "valid")
    return lon, lat, x, y


def join_lines(item_list):
    output = []
    for arr in item_list:
        output += list(arr) + [None]
    return output


def road_layers(base, closed):
    grid, roads = base["territory"]["grid"], base["territory"]["roads"]
    groups = {"secundaria": [], "terciaria": [], "cerrada": []}
    for i, v in roads.iterrows():
        group = "cerrada" if i in closed else v["type"]
        groups[group].append(v["cells"])
    output = {}
    for group, item_list in groups.items():
        coords = [cell_coordinates(grid, c) for c in item_list]
        output[group] = tuple(join_lines([cc[j] for cc in coords]) for j in range(4)) + (item_list,)
    return output


def community_text(base, scenario, kind, i):
    row = base["demand"].iloc[i]
    ev = scenario["types"][kind]["eval"]
    j = int(ev["assignment"][i])
    grid = base["territory"]["grid"]
    f, c = divmod(int(row["cell"]), grid["nx"])
    suit_class = config.CLASSES[int(base["suitability"][f, c])]
    return (f"<b>{row['name']}</b> ({row['municipality']})<br>Población: {row['population']} hab · "
            f"altura {row['altitude']} m<br>Terreno: {suit_class}<br>"
            f"{kind} {j + 1 if j >= 0 else '–'}: {fmt(ev['time'][i], 0, 'min')} · {fmt(ev['km'][i], 1, 'km')} · "
            f"a pie {fmt(ev['walk'][i], 0, 'min')}")


def map_figure(base, scenario, kind, background="Relieve", show_simma=False, show_all=False, route=None):
    grid, relief = base["territory"]["grid"], base["territory"]["relief"]
    osm = background.startswith("OpenStreetMap")
    fig = go.Figure()

    if not osm:
        if background == "Aptitud IA":
            fig.add_trace(go.Heatmap(x=grid["lon"], y=grid["lat"], z=base["suitability"], colorscale=CLASS_SCALE,
                                     zmin=0, zmax=2, showscale=False, opacity=0.85, hoverinfo="skip"))
        else:
            fig.add_trace(go.Heatmap(x=grid["lon"], y=grid["lat"], z=relief["altitude"].round(0),
                                     colorscale=RELIEF_SCALE, zsmooth="best", hoverinfo="skip",
                                     colorbar=dict(title=dict(text="m"), len=0.6, thickness=12)))
            shade = np.clip(0.5 + np.gradient(relief["altitude"], axis=1) / 180
                            + np.gradient(relief["altitude"], axis=0) / 180, 0, 1)
            fig.add_trace(go.Heatmap(x=grid["lon"], y=grid["lat"], z=shade, showscale=False, opacity=0.28,
                                     colorscale=[[0, "black"], [1, "white"]], zsmooth="best", hoverinfo="skip"))

    for river in relief["river_lines"]:
        lon = config.LON_MIN + river["x_km"] / grid["km_per_degree_lon"]
        lat = config.LAT_MIN + river["y_km"] / grid["km_per_degree_lat"]
        line = dict(color="#2b7bba", width=4 if river["type"] == "principal" else 1.5)
        if osm:
            fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", name=river["name"], hoverinfo="name",
                                        line=line, showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=lon, y=lat, mode="lines", name=river["name"], hoverinfo="name",
                                     line=line, showlegend=False))

    styles = {"secundaria": ("#3b3b3b", 3, "Vía secundaria"), "terciaria": ("#6e6e6e", 1.6, "Vía terciaria"),
              "cerrada": ("#d62728", 4, "Vía cerrada")}
    for group, (lon, lat, _, _, item_list) in road_layers(base, scenario["params"]["closed"]).items():
        if not item_list:
            continue
        color, width, name = styles[group]
        if osm:
            fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", name=name, hoverinfo="name",
                                        line=dict(color=color, width=width)))
        else:
            fig.add_trace(go.Scatter(x=lon, y=lat, mode="lines", name=name, hoverinfo="name",
                                     line=dict(color=color, width=width, dash="dot" if group == "cerrada" else None)))

    if show_simma and not osm:
        s = base["territory"]["simma"]
        fig.add_trace(go.Scatter(x=s["lon"], y=s["lat"], mode="markers", name="Movimientos en masa (SIMMA)",
                                 marker=dict(symbol="x", size=4, color="#7a4b1f"),
                                 hovertext=s["type"] + " · " + s["year"].astype(str), hoverinfo="text"))

    r = scenario["types"].get(kind)
    if r is not None and show_all:
        paths = scenarios.all_routes(base, scenario, kind)
        coords = [cell_coordinates(grid, c) for c in paths if len(c) > 1]
        lon, lat = join_lines([c[0] for c in coords]), join_lines([c[1] for c in coords])
        kw = dict(mode="lines", name="Rutas a su centro", hoverinfo="skip", line=dict(color="#ff9f1c", width=1.2))
        fig.add_trace(go.Scattermap(lon=lon, lat=lat, **kw) if osm else go.Scatter(x=lon, y=lat, opacity=0.7, **kw))

    if route is not None and len(route["path"]) > 1:
        lon, lat, _, _ = cell_coordinates(grid, route["path"])
        name = f"Ruta real: {short(route['community'])} ({route['minutes']:.0f} min, {route['km']:.1f} km)"
        straight = route["straight"]
        if osm:
            fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", name=name, line=dict(color="#ff6a00", width=5)))
        else:
            fig.add_trace(go.Scatter(x=straight["lon"], y=straight["lat"], mode="lines", name="Línea recta (no transitable)",
                                     line=dict(color="#222", width=2, dash="dash"), hoverinfo="name"))
            fig.add_trace(go.Scatter(x=lon, y=lat, mode="lines", name=name, line=dict(color="#ff6a00", width=5),
                                     hoverinfo="name"))

    if r is not None:
        dem = base["demand"]
        colors = [GROUP_PALETTE[j % len(GROUP_PALETTE)] if j >= 0 else "#000000" for j in r["eval"]["assignment"]]
        texts = [community_text(base, scenario, kind, i) for i in range(len(dem))]
        sizes = 7 + np.sqrt(dem["population"].to_numpy()) / 2.2
        if osm:
            fig.add_trace(go.Scattermap(lon=dem["lon"], lat=dem["lat"], mode="markers", name="Comunidades",
                                        marker=dict(size=sizes, color=colors), hovertext=texts, hoverinfo="text"))
        else:
            symbols = ["square" if t == "cabecera" else "circle" for t in dem["type"]]
            fig.add_trace(go.Scatter(x=dem["lon"], y=dem["lat"], mode="markers", name="Comunidades (color = su centro)",
                                     marker=dict(size=sizes, color=colors, symbol=symbols,
                                                 line=dict(color="white", width=1)),
                                     hovertext=texts, hoverinfo="text"))

    for t, res in scenario["types"].items():
        rule = config.CENTER_TYPES[t]
        cs = res["centers"]
        texts = [f"<b>{rule['emoji']} {t} {c['number']}</b>{' (existente)' if c['existing'] else ''}<br>"
                 f"{c['location']}<br>Terreno {c['ai_suitability']} · pendiente {c['slope']:.0f} % · "
                 f"{c['altitude']:.0f} m<br>Atiende {c['communities']} comunidades ({c['inhabitants']} hab)<br>"
                 f"Peor viaje: {fmt(c['worst_time_min'], 0, 'min')}" for c in cs]
        labels = [f"{rule['emoji']}{c['number']}" for c in cs]
        if osm:
            fig.add_trace(go.Scattermap(lon=[c["lon"] for c in cs], lat=[c["lat"] for c in cs], mode="markers+text",
                                        name=f"{rule['emoji']} {t}", text=labels, textposition="top center",
                                        marker=dict(size=18, color=rule["color"]), hovertext=texts, hoverinfo="text"))
        else:
            fig.add_trace(go.Scatter(x=[c["lon"] for c in cs], y=[c["lat"] for c in cs], mode="markers+text",
                                     name=f"{rule['emoji']} {t}", text=labels, textposition="top center",
                                     textfont=dict(size=14), hovertext=texts, hoverinfo="text",
                                     marker=dict(size=17 if t == kind else 13, color=rule["color"],
                                                 symbol=rule["symbol"], line=dict(color="white", width=2))))

    output = base["territory"]["settlements"].query("type == 'salida'")
    if len(output) and not osm:
        fig.add_annotation(x=float(output["lon"].iloc[0]), y=float(output["lat"].iloc[0]), text="← Sogamoso",
                           showarrow=False, xanchor="left", font=dict(size=12, color="#333"))

    fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=30, b=0), uirevision="mapa",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)),
                      plot_bgcolor="white")
    if osm:
        fig.update_layout(map=dict(style="open-street-map", zoom=9.4,
                                   center=dict(lat=float(np.mean(grid["lat"])), lon=float(np.mean(grid["lon"])))))
    else:
        fig.update_xaxes(range=[config.LON_MIN, config.LON_MAX], showgrid=False, title=None)
        fig.update_yaxes(range=[config.LAT_MIN, config.LAT_MAX], showgrid=False, title=None, scaleanchor="x",
                         scaleratio=grid["km_per_degree_lat"] / grid["km_per_degree_lon"])
    return fig


def profile_figure(route):
    fig = go.Figure()
    straight = route["straight"]
    straight_distance = np.linspace(0, straight["length_3d_km"], len(straight["altitude"]))
    fig.add_trace(go.Scatter(x=route["profile_km"], y=route["profile_altitude"], mode="lines", fill="tozeroy",
                             name=f"Ruta real ({route['km']:.1f} km, sube {route['ascent_m']} m)",
                             line=dict(color="#ff6a00", width=2.5), fillcolor="rgba(255,106,0,0.12)"))
    fig.add_trace(go.Scatter(x=straight_distance, y=straight["altitude"], mode="lines",
                             name=f"Línea recta ({straight['horizontal_km']:.1f} km, sube {straight['ascent_m']} m)",
                             line=dict(color="#222", width=1.5, dash="dash")))
    floor = min(min(route["profile_altitude"]), float(np.min(straight["altitude"]))) - 150
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15, x=0),
                      xaxis_title="km recorridos", yaxis_title="Altura (m)", yaxis_range=[floor, None])
    return fig


def show_route(route):
    straight = route["straight"]
    river = " y **cruza el río principal sin puente**" if straight["crosses_main_river"] else ""
    st.markdown(
        f"**{route['community']} → centro {route['center']}** ({route['center_location']}): "
        f"por el terreno real son **{route['minutes']:.0f} min** y **{route['km']:.1f} km** (sube {route['ascent_m']} m, "
        f"baja {route['descent_m']} m). La línea recta mide {straight['horizontal_km']:.1f} km, pero sube "
        f"{straight['ascent_m']} m y pasa a {straight['max_altitude_m']} m de altura{river}: no es un camino real.")
    st.plotly_chart(profile_figure(route), key=f"perfil_{route['index']}", config=PLOTLY_CONFIG)


def map_page():
    e = state()
    base, scn = e["base"], e["scenario"]
    st.title("ACCESO · ubicación equitativa de centros rurales")
    st.caption("Dónde construir centros de salud, colegios u hospitales para que la comunidad más alejada "
               "quede lo más cerca posible, por el terreno real y solo en terreno apto.")
    kind = kind_selector("type_map")
    if kind is None:
        return
    kpi_cards(scn, kind, base)
    if scn["params"]["closed"]:
        without = e["compute"](scenarios.parameters(dict(scn["params"]["ks"]), scn["params"]["mode"],
                                                     scn["params"]["criterion"], scn["params"]["refine"], (),
                                                     scn["params"]["fixed_centers"]))
        closure_notice(base, scn, without, kind)

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.6])
    background = c1.radio("Fondo", ["Relieve", "Aptitud IA", "OpenStreetMap (requiere internet)"], horizontal=False,
                          key="basemap")
    show_all = c2.toggle("Todas las rutas", key="all_routes")
    show_simma = c3.toggle("Deslizamientos (SIMMA)", key="show_simma")
    names = base["demand"]["name"].tolist()
    farthest = scn["types"][kind]["eval"]["farthest"]
    chosen = c4.selectbox("Resaltar la ruta de", range(len(names)), key="route_map",
                          index=farthest if farthest is not None else 0,
                          format_func=lambda i: names[i] + (" (la más alejada)" if i == farthest else ""))
    route = scenarios.route_of(base, scn, kind, chosen)
    st.plotly_chart(map_figure(base, scn, kind, background, show_simma, show_all, route), key="map_2d",
                    config=PLOTLY_CONFIG)
    st.caption("Color de cada comunidad = el centro que le corresponde · cuadrado = cabecera municipal · "
               "línea naranja = ruta real · línea negra discontinua = línea recta (no transitable) · "
               "rojo punteado = vía cerrada. Pasa el cursor para ver el detalle; rueda del ratón para acercar.")
    if route:
        show_route(route)
    else:
        st.warning("Esta comunidad no tiene ruta a ningún centro con las vías cerradas actuales.")

    tab_centers, tab_steps = st.tabs(["Centros elegidos", "Paso a paso del algoritmo"])
    with tab_centers:
        centers_table(scn)
    with tab_steps:
        for step in scn["types"][kind]["steps"]:
            st.markdown(f"- {step}")


def centers_table(scn):
    rows = []
    for kind, r in scn["types"].items():
        for c in r["centers"]:
            rows.append({"Centro": f"{config.CENTER_TYPES[kind]['emoji']} {kind} {c['number']}",
                         "Ubicación": c["location"], "Terreno (IA)": c["ai_suitability"],
                         "Pendiente (%)": round(c["slope"], 1), "Altura (m)": round(c["altitude"]),
                         "Comunidades": c["communities"], "Habitantes": c["inhabitants"],
                         "Peor viaje (min)": round(c["worst_time_min"], 1), "Peor distancia (km)": round(c["worst_km"], 1),
                         "Desplazado por terreno desde": c["moved_from"] or "–",
                         "Existente": "Sí" if c["existing"] else "No",
                         "Lat": round(c["lat"], 5), "Lon": round(c["lon"], 5)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def figure_3d(base, scn, kind, color_by="Aptitud IA", exaggeration=2.0, step=2, show_roads=True,
              show_all=False, route=None):
    grid, relief = base["territory"]["grid"], base["territory"]["relief"]
    alt = relief["altitude"]
    xs, ys = grid["x_km"][::step], grid["y_km"][::step]
    z = alt[::step, ::step].round(0)

    def altitude_at(x_km, y_km):
        col = np.clip(np.round(np.asarray(x_km, dtype=float) / grid["cell_km"]).astype(int), 0, grid["nx"] - 1)
        r_idx = np.clip(np.round(np.asarray(y_km, dtype=float) / grid["cell_km"]).astype(int), 0, grid["ny"] - 1)
        return alt[r_idx, col]

    suit_class = base["suitability"][::step, ::step]
    class_names = np.array(config.CLASSES, dtype=object)[suit_class]
    if color_by == "Aptitud IA":
        surface = dict(surfacecolor=suit_class, colorscale=CLASS_SCALE, cmin=0, cmax=2,
                       colorbar=dict(title=dict(text="Terreno"), tickvals=[1 / 3, 1, 5 / 3],
                                        ticktext=config.CLASSES, len=0.5, thickness=14))
    elif color_by == "Pendiente":
        surface = dict(surfacecolor=np.clip(relief["slope"][::step, ::step], 0, 60), colorscale="YlOrRd",
                       cmin=0, cmax=60, colorbar=dict(title=dict(text="Pendiente %"), len=0.5, thickness=14))
    else:
        surface = dict(surfacecolor=z, colorscale=RELIEF_SCALE,
                       colorbar=dict(title=dict(text="Altura m"), len=0.5, thickness=14))
    fig = go.Figure(go.Surface(
        x=xs, y=ys, z=z, text=class_names, name="Relieve",
        hovertemplate="Este %{x:.1f} km · Norte %{y:.1f} km<br>Altura %{z:.0f} m<br>Terreno: %{text}<extra></extra>",
        lighting=dict(ambient=0.55, diffuse=0.85, roughness=0.9, specular=0.08, fresnel=0.1),
        lightposition=dict(x=-1e5, y=1e5, z=6e4), **surface))

    for river in relief["river_lines"]:
        fig.add_trace(go.Scatter3d(x=river["x_km"], y=river["y_km"], z=altitude_at(river["x_km"], river["y_km"]) + 12,
                                   mode="lines", name=river["name"], showlegend=False, hoverinfo="name",
                                   line=dict(color="#1f78c8", width=6 if river["type"] == "principal" else 3)))
    if show_roads:
        for group, (_, _, x, y, item_list) in road_layers(base, scn["params"]["closed"]).items():
            if not item_list:
                continue
            zz = [None if v is None else float(altitude_at(v, w)) + 18 for v, w in zip(x, y)]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=zz, mode="lines", hoverinfo="name",
                                       name={"cerrada": "Vía cerrada", "secundaria": "Vía secundaria"}.get(group, "Vía terciaria"),
                                       line=dict(color={"cerrada": "#d62728", "secundaria": "#222"}.get(group, "#555"),
                                                 width=5 if group != "terciaria" else 3)))

    r = scn["types"].get(kind)
    if r is not None and show_all:
        paths = [cell_coordinates(grid, c) for c in scenarios.all_routes(base, scn, kind) if len(c) > 1]
        x, y = join_lines([c[2] for c in paths]), join_lines([c[3] for c in paths])
        zz = [None if v is None else float(altitude_at(v, w)) + 30 for v, w in zip(x, y)]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=zz, mode="lines", name="Rutas a su centro", hoverinfo="skip",
                                   line=dict(color="#ff9f1c", width=2)))
    if route is not None and len(route["path"]) > 1:
        _, _, x, y = cell_coordinates(grid, route["path"])
        fig.add_trace(go.Scatter3d(x=x, y=y, z=altitude_at(x, y) + 40, mode="lines", hoverinfo="name",
                                   name=f"Ruta real: {short(route['community'])} ({route['minutes']:.0f} min)",
                                   line=dict(color="#ff6a00", width=9)))
        straight = route["straight"]
        fig.add_trace(go.Scatter3d(x=straight["x_km"], y=straight["y_km"], z=straight["altitude"] + 40, mode="lines",
                                   name="Línea recta sobre el relieve", hoverinfo="name",
                                   line=dict(color="#111", width=5, dash="dash")))

    if r is not None:
        dem = base["demand"]
        colors = [GROUP_PALETTE[j % len(GROUP_PALETTE)] if j >= 0 else "#000" for j in r["eval"]["assignment"]]
        fig.add_trace(go.Scatter3d(
            x=dem["x_km"], y=dem["y_km"], z=dem["altitude"] + 70, mode="markers", name="Comunidades",
            marker=dict(size=3 + np.sqrt(dem["population"].to_numpy()) / 4, color=colors,
                        line=dict(color="white", width=1)),
            hovertext=[community_text(base, scn, kind, i) for i in range(len(dem))], hoverinfo="text"))

    for t, res in scn["types"].items():
        rule = config.CENTER_TYPES[t]
        cs = res["centers"]
        if not cs:
            continue
        x = [c["x_km"] for c in cs]
        y = [c["y_km"] for c in cs]
        z0 = [c["altitude"] for c in cs]
        height = 450
        fig.add_trace(go.Scatter3d(x=join_lines([[v, v] for v in x]), y=join_lines([[v, v] for v in y]),
                                   z=join_lines([[v, v + height] for v in z0]), mode="lines", showlegend=False,
                                   hoverinfo="skip", line=dict(color=rule["color"], width=6)))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=[v + height for v in z0], mode="markers+text", name=f"{rule['emoji']} {t}",
            text=[f"{rule['emoji']}{c['number']}" for c in cs], textposition="top center",
            marker=dict(size=9, color=rule["color"], symbol=rule["symbol"], line=dict(color="white", width=2)),
            hovertext=[f"<b>{t} {c['number']}</b><br>{c['location']}<br>Terreno {c['ai_suitability']} · "
                       f"pendiente {c['slope']:.0f} % · {c['altitude']:.0f} m" for c in cs], hoverinfo="text"))

    width, length = grid["x_km"][-1], grid["y_km"][-1]
    largest = max(width, length)
    z_range = (alt.max() - alt.min()) / 1000
    fig.update_layout(
        height=HEIGHT_3D, margin=dict(l=0, r=0, t=10, b=0), uirevision="relief_3d",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, x=0, font=dict(size=11)),
        scene=dict(aspectmode="manual",
                   aspectratio=dict(x=width / largest, y=length / largest, z=exaggeration * z_range / largest),
                   xaxis=dict(title=dict(text="Este (km)")), yaxis=dict(title=dict(text="Norte (km)")),
                   zaxis=dict(title=dict(text="Altura (m)")),
                   camera=dict(eye=dict(x=-0.9, y=-1.35, z=0.85))))
    return fig


def relief_3d_page():
    e = state()
    base, scn = e["base"], e["scenario"]
    st.title("⛰️ Relieve 3D")
    st.caption("Rota con clic y arrastre, acerca con la rueda, pasa el cursor para ver altura y aptitud. "
               "Los cambios de la barra lateral se recalculan al instante.")
    kind = kind_selector("type_3d")
    if kind is None:
        return
    view_3d(base, scn, kind)


@fragment
def view_3d(base, scn, kind):
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.4])
    color_by = c1.radio("Colorear el terreno por", ["Aptitud IA", "Altura", "Pendiente"], key="color_3d",
                        horizontal=True)
    exaggeration = c2.slider("Exageración vertical", 1.0, 4.0, 2.0, 0.5, key="exaggeration_3d",
                             help="Multiplica las alturas para que el relieve se aprecie mejor.")
    detail = c3.select_slider("Detalle", ["Bajo", "Medio", "Alto"], value="Medio", key="detail_3d",
                              help="Alto = cada celda de 450 m. Bajo = más rápido en equipos modestos.")
    names = base["demand"]["name"].tolist()
    farthest = scn["types"][kind]["eval"]["farthest"]
    chosen = c4.selectbox("Ruta de", range(len(names)), index=farthest if farthest is not None else 0, key="route_3d",
                          format_func=lambda i: names[i] + (" (la más alejada)" if i == farthest else ""))
    c5, c6 = st.columns(2)
    show_roads = c5.toggle("Mostrar vías", value=True, key="roads_3d")
    show_all = c6.toggle("Mostrar todas las rutas", key="all_3d")
    route = scenarios.route_of(base, scn, kind, chosen)
    step = {"Bajo": 3, "Medio": 2, "Alto": 1}[detail]
    st.plotly_chart(figure_3d(base, scn, kind, color_by, exaggeration, step, show_roads, show_all, route),
                    key="figure_3d", config=PLOTLY_CONFIG)
    if route:
        show_route(route)


def terrain_ai_page():
    import terrain_ai
    e = state()
    base = e["base"]
    ai = base["ai"]
    st.title("🧠 IA de terreno (Random Forest)")
    st.caption("Clasifica cada celda de 450 m como Apto, Condicionado o No apto con pendiente, altura, curvatura, "
               "posición topográfica (¿cima o valle?) y distancia al río. Validación cruzada ESPACIAL por bloques de "
               f"{config.CV_BLOCK_SIZE_KM} km.")
    c = st.columns(4)
    c[0].metric("AUC (macro)", f"{ai['auc']:.3f}", help=f"Meta T8: ≥ {config.TARGETS['T8_auc']}")
    c[1].metric("Recall No apto", f"{ai['recall_unsuitable']:.3f}", help=f"Meta T8: ≥ {config.TARGETS['T8_recall']}")
    c[2].metric("Exactitud", f"{ai['accuracy']:.1%}")
    c[3].metric("Celdas de entrenamiento", f"{ai['samples']:,}", help=f"{ai['blocks']} bloques espaciales")
    left, right = st.columns(2)
    with left:
        imp = ai["importances"].sort_values("importance")
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["variable"], orientation="h", marker_color="#1f6feb"))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), title="Importancia de variables")
        st.plotly_chart(fig, key="importances", config=PLOTLY_CONFIG)
    with right:
        mc = np.array(ai["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=mc, x=config.CLASSES, y=config.CLASSES, colorscale="Blues", showscale=False,
                                   text=mc, texttemplate="%{text}"))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), title="Matriz de confusión (CV espacial)",
                          xaxis_title="Predicción", yaxis_title="Verdad de campo", yaxis_autorange="reversed")
        st.plotly_chart(fig, key="confusion", config=PLOTLY_CONFIG)
    suit_class = base["suitability"]
    st.markdown("**Territorio clasificado:** " + " · ".join(
        f"{config.CLASSES[k]} {100 * (suit_class == k).mean():.0f} %" for k in range(3)))

    st.subheader("Consultar un punto")
    grid = base["territory"]["grid"]
    a, b = st.columns(2)
    lat = a.number_input("Latitud", float(config.LAT_MIN), float(config.LAT_MAX), float(np.mean(grid["lat"])),
                         step=0.005, format="%.4f", key="ai_lat")
    lon = b.number_input("Longitud", float(config.LON_MIN), float(config.LON_MAX), float(np.mean(grid["lon"])),
                         step=0.005, format="%.4f", key="ai_lon")
    row = int(np.argmin(np.abs(grid["lat"] - lat)))
    col = int(np.argmin(np.abs(grid["lon"] - lon)))
    info = terrain_ai.explain_cell(base["territory"], base["suitability"], base["prob"], row * grid["nx"] + col)
    st.json(info, expanded=True)

    st.subheader("Comunidades donde NO se puede construir")
    rows = []
    for _, p in base["demand"].iterrows():
        f, cc = divmod(int(p["cell"]), grid["nx"])
        if base["suitability"][f, cc] != 0:
            exp = terrain_ai.explain_cell(base["territory"], base["suitability"], base["prob"], p["cell"])
            rows.append({"Comunidad": p["name"], "Municipio": p["municipality"], "Clase": exp["suitability"],
                         "Prob. No apto": exp["probabilities"]["No apto"], "Motivos": "; ".join(exp["reasons"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def compare_page():
    e = state()
    base, scn = e["base"], e["scenario"]
    st.title("⚖️ Comparar estrategias (Tabla III)")
    st.caption("Mismas métricas para cada estrategia, sobre el relieve real y con las vías cerradas actuales.")
    kind = kind_selector("type_compare")
    if kind is None:
        return
    key = f"comparison|{scn['id']}|{kind}"
    if key not in st.session_state:
        with st.spinner("Comparando estrategias (20 sorteos al azar)..."):
            st.session_state[key] = scenarios.compare_strategies(base, scn, kind)
    table, n4 = st.session_state[key]
    st.session_state["last_n4"] = (scn["id"], kind, n4)
    columns = {c: st.column_config.NumberColumn(format="%.1f") for c in table.columns if table[c].dtype.kind == "f"}
    st.dataframe(table, hide_index=True, column_config=columns)
    worst_col = next(c for c in table.columns if c.startswith("Peor viaje"))
    fig = go.Figure(go.Bar(x=table[worst_col], y=table["Estrategia"], orientation="h",
                           marker_color=["#1f6feb"] + ["#9aa4b2"] * (len(table) - 1),
                           text=table[worst_col].round(1), textposition="outside"))
    fig.update_layout(height=320, margin=dict(l=0, r=40, t=30, b=0), title=f"{worst_col} (menos es mejor)",
                      yaxis_autorange="reversed")
    st.plotly_chart(fig, key="comparison", config=PLOTLY_CONFIG)
    st.info(f"**N4 · Sitios inviables evitados:** sin la IA de terreno, el **{n4['pct_without_ai_unsuitable']:.0f} %** de los "
            f"centros caería en terreno No apto (según la verdad de campo). Con IA: {n4['pct_with_ai_unsuitable']:.0f} %. "
            f"Evitarlo cambia el peor viaje en {n4['cost_pct']:+.1f} %.")
    st.caption("Criterio de equidad vs. cobertura (RF12): minimizar el peor viaje no equivale a maximizar la población "
               "cercana. La fila de cobertura máxima muestra la alternativa para decidir con ambos criterios a la vista.")


def assistant_context():
    import ai_assistant
    e = state()
    key = f"ctx_asistente|{e['scenario']['id']}"
    if key not in st.session_state:
        st.session_state[key] = ai_assistant.Context(e["base"], e["scenario"], recompute=e["compute"])
    return st.session_state[key]


def connection_panel():
    import ai_assistant
    provider = st.session_state.get("provider")
    with st.expander("🔌 Conexión con el modelo de lenguaje", expanded=provider is None):
        st.markdown(
            "1. **LM Studio (recomendado, sin internet):** abre LM Studio → descarga un modelo con soporte de "
            "herramientas (p. ej. *Qwen2.5 7B Instruct* o *Qwen3 4B*) → pestaña **Developer** → **Start Server** "
            "(puerto 1234).  \n2. **Nube (opcional, con internet):** cualquier proveedor compatible con la API de "
            "OpenAI. La clave solo vive en esta sesión.  \n3. Si no hay modelo, el chat responde en **modo guiado**.")
        a, b = st.columns(2)
        lm_url = a.text_input("URL de LM Studio", config.LM_STUDIO_URL, key="set_lm_url",
                              help="Si LM Studio corre en otro PC de la red local: http://IP-de-ese-PC:1234/v1")
        lm_model = b.text_input("Modelo (vacío = el que esté cargado)", config.LM_STUDIO_MODEL, key="set_lm_model")
        use_cloud = st.toggle("Usar un modelo en la nube si LM Studio no responde", value=bool(config.CLOUD_API_KEY),
                              key="set_use_cloud")
        cloud = {}
        if use_cloud:
            c, d, f = st.columns(3)
            cloud["cloud_url"] = c.text_input("URL de la API", config.CLOUD_URL, key="set_cloud_url")
            cloud["cloud_model"] = d.text_input("Modelo", config.CLOUD_MODEL, key="set_cloud_model")
            cloud["cloud_key"] = f.text_input("Clave API", config.CLOUD_API_KEY, type="password", key="set_cloud_key")
        timeout = st.slider("Espera máxima por respuesta (s)", 5, 120, config.RESPONSE_TIMEOUT_S, key="set_timeout",
                            help="Súbelo si tu computador es lento. La meta T10 del artículo es ≤ 5 s.")
        settings = {"lm_url": lm_url, "lm_model": lm_model, "use_cloud": use_cloud, "timeout": timeout, **cloud}
        st.session_state["ai_settings"] = settings
        if st.button("Probar conexión", type="primary") or provider is None:
            with st.spinner("Buscando un modelo..."):
                st.session_state["provider"] = ai_assistant.detect_provider(settings)
    provider = st.session_state["provider"]
    if provider["name"]:
        st.success(provider["message"])
    else:
        st.info(provider["message"])
    return provider


def assistant_page():
    import ai_assistant
    st.title("💬 Asistente de IA")
    st.caption("Pregunta en lenguaje natural. El asistente explica con cifras del motor y cita el escenario; "
               "nunca calcula por su cuenta. Cada cifra se verifica contra los datos del motor.")
    provider = connection_panel()
    ctx = assistant_context()
    history = st.session_state.setdefault("chat", [])

    st.markdown("**Preguntas sugeridas**")
    slope = None
    columns = st.columns(3)
    for i, question in enumerate(ai_assistant.guided_questions(ctx)):
        if columns[i % 3].button(question, key=f"guiada_{i}"):
            slope = question

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                response_footer(m)

    written = st.chat_input("Ej.: ¿Por qué el centro 2 no quedó en la vereda El Cedro? · ¿Qué pasa si se cae el puente?")
    question = written or slope
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Consultando el motor..."):
                r = ai_assistant.respond(question, history[:-1], ctx, st.session_state.get("ai_settings"),
                                         provider=provider)
            message = {"role": "assistant", "content": r["text"], **{k: r[k] for k in
                       ("source", "model", "latency_s", "verification", "tools", "note", "scenario")}}
            st.markdown(r["text"])
            response_footer(message)
        history.append(message)

    a, b = st.columns(2)
    if a.button("📝 Generar informe del escenario"):
        with st.spinner("Redactando el informe..."):
            st.session_state["report"] = ai_assistant.generate_report(ctx, st.session_state.get("ai_settings"))
    if b.button("🗑️ Limpiar conversación"):
        st.session_state["chat"] = []
        st.rerun()
    report = st.session_state.get("report")
    if report:
        with st.container(border=True):
            st.caption(f"Redactado por: {report['source']} · cifras no verificadas: "
                       f"{report['verification']['unverified'] or 'ninguna'}")
            st.markdown(report["text"])
            st.download_button("Descargar informe (.md)", report["text"], file_name=f"informe_{ctx.id}.md",
                               mime="text/markdown")

    with st.expander("🔍 Anatomía: qué se le envía al modelo"):
        messages, _ = ai_assistant.build_messages("(tu pregunta)", [], ctx)
        st.markdown("**1. Mensaje de sistema** = reglas + resumen del escenario (RAG):")
        st.code(messages[0]["content"][:4000], language="markdown")
        st.markdown("**2. Herramientas** que el modelo puede pedir (tool calling):")
        st.json([h["function"] for h in ai_assistant.TOOLS], expanded=False)
        st.markdown("**3. Bucle:** el modelo responde o pide una herramienta → Python la ejecuta con el motor → "
                    "el resultado vuelve al modelo → respuesta final → se verifican las cifras.")


def response_footer(m):
    if "source" not in m:
        return
    v = m["verification"]
    badge = "✅ todas las cifras verificadas" if not v["unverified"] else \
        f"⚠️ cifras no verificadas: {', '.join(str(x) for x in v['unverified'])}"
    st.caption(f"{m['source']} · {m['model']} · {m['latency_s']:.2f} s · {badge}"
               + (f" · {m['note']}" if m.get("note") else ""))
    if m.get("tools"):
        with st.expander(f"Herramientas usadas ({len(m['tools'])})"):
            for h in m["tools"]:
                st.markdown(f"**{h['tool']}** `{json.dumps(h['args'], ensure_ascii=False)}`")
                st.json(h["result"], expanded=False)


SUS_QUESTIONS = [
    "Creo que me gustaría usar este sistema con frecuencia.",
    "Encontré el sistema innecesariamente complejo.",
    "Pensé que el sistema era fácil de usar.",
    "Creo que necesitaría el apoyo de un técnico para poder usar este sistema.",
    "Encontré que las funciones del sistema estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en el sistema.",
    "Imagino que la mayoría de las personas aprendería a usar este sistema muy rápido.",
    "Encontré el sistema muy engorroso de usar.",
    "Me sentí muy seguro usando el sistema.",
    "Necesité aprender muchas cosas antes de poder usar este sistema.",
]


def kpi_page():
    import kpi_audit
    e = state()
    base, scn = e["base"], e["scenario"]
    st.title("📊 Auditoría de KPI (Tabla II)")
    st.caption("Cada prueba se ejecuta aquí mismo, con o sin internet. Lo que mide una persona (Lighthouse y SUS) "
               "se ingresa abajo. Los resultados se guardan mientras la pestaña siga abierta.")
    measured = st.session_state.setdefault("kpi_measured", {})

    kind = kind_selector("type_kpi", "KPI de negocio calculados para")
    if kind is None:
        return
    n4 = st.session_state.get("last_n4")
    n4 = n4[2] if n4 and n4[0] == scn["id"] and n4[1] == kind else None
    if n4 is None:
        key = f"comparison|{scn['id']}|{kind}"
        if key not in st.session_state:
            with st.spinner("Calculando N4 con el comparador de estrategias..."):
                st.session_state[key] = scenarios.compare_strategies(base, scn, kind, draws=10)
        n4 = st.session_state[key][1]
    measured.update(kpi_audit.scenario_kpi(scn, kind, n4))
    measured.update({"T8_auc": base["ai"]["auc"], "T8_recall": base["ai"]["recall_unsuitable"]})

    st.subheader("Pruebas del núcleo (T1–T4, Tabla III sintética)")
    mode = st.radio("Alcance", ["quick", "full"], horizontal=True, key="kpi_mode",
                    format_func=lambda m: "Rápida (~10 s)" if m == "quick" else "Completa como el artículo (1–3 min)")
    if st.button("▶️ Ejecutar pruebas del núcleo", type="primary"):
        with st.status("Ejecutando pruebas del núcleo...", expanded=True) as box:
            table, res = kpi_audit.test_scalability(mode, progress=box.write)
            box.write("Factor de aproximación contra el óptimo por fuerza bruta...")
            ap = kpi_audit.test_approximation(mode)
            box.write("Tabla III en regiones sintéticas...")
            t3, r3 = kpi_audit.test_table_iii(mode)
            box.write("Cierre de vías sobre el relieve completo...")
            t3_terr = kpi_audit.test_territory_closure(base, scn["params"])
            box.update(label="Pruebas del núcleo terminadas", state="complete")
        st.session_state["kpi_core"] = (table, t3)
        measured.update({"T1": res["T1_p95_ms"], "T1_V": res["V_max"], "T2_slope": res["T2_slope"],
                         "T2_slope_matrix": res["T2_slope_matrix"], "T2_mb": res["T2_mb"],
                         "T3": res["T3_p95_ms"], "T3_territory": t3_terr, "T4_mean": ap["mean"],
                         "T4_max": ap["max"], "T4_mean_ref": ap["mean_refined"],
                         "N1_synth": r3["N1"], "N2_synth": r3["N2"]})
    if "kpi_core" in st.session_state:
        table, t3 = st.session_state["kpi_core"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=table["V"], y=table["Incremental p50 (ms)"], mode="lines+markers",
                                 name=f"Incremental (pendiente {measured.get('T2_slope', float('nan')):.2f})"))
        matrix_rows = table.dropna(subset=["Matriz (ms)"])
        fig.add_trace(go.Scatter(x=matrix_rows["V"], y=matrix_rows["Matriz (ms)"], mode="lines+markers",
                                 name=f"Matriz (pendiente {measured.get('T2_slope_matrix', float('nan')):.2f})"))
        fig.add_hline(y=config.TARGETS["T1"], line_dash="dash", annotation_text="Meta RNF02: 2 s")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title="Tiempo de ubicación de k = 5 centros",
                          xaxis=dict(type="log", title="Número de comunidades V"),
                          yaxis=dict(type="log", title="Tiempo (ms)"))
        st.plotly_chart(fig, key="scalability", config=PLOTLY_CONFIG)
        st.markdown("**Tabla III en regiones sintéticas** (artículo: 46,5 / 45,0 / 34,3 km)")
        st.dataframe(t3.round(1), hide_index=True)

    st.subheader("Pruebas del sistema (T9, T10, T11)")
    a, b = st.columns(2)
    if a.button("🔌 Probar sin conexión (T11)"):
        with st.spinner("Bloqueando la red y ejecutando el núcleo..."):
            table11, pct = kpi_audit.test_offline(base, scn["params"])
        measured["T11"] = pct
        st.session_state["kpi_t11"] = table11
    if "kpi_t11" in st.session_state:
        a.dataframe(st.session_state["kpi_t11"], hide_index=True)
    with b:
        n = st.slider("Preguntas del banco", 5, 50, 10, key="kpi_n_questions")
        force = st.toggle("Forzar modo guiado (sin modelo)", key="kpi_force")
        if st.button("💬 Probar el asistente (T9 y T10)"):
            with st.status("Preguntando al asistente...", expanded=True) as box:
                table9, r9 = kpi_audit.test_assistant(base, scn, n, force, progress=box.write)
                box.update(label="Banco de preguntas terminado", state="complete")
            measured.update({"T9": r9["T9"], "T9_no_fabrication": r9["T9_no_fabrication"], "T10": r9["T10"],
                             "T9_source": r9["source"]})
            st.session_state["kpi_t9"] = table9
        if "kpi_t9" in st.session_state:
            st.dataframe(st.session_state["kpi_t9"], hide_index=True)
            st.caption("T9 y T10 se miden de verdad con un modelo conectado; en modo guiado miden las plantillas.")

    st.subheader("Mediciones manuales (T5, T6, T7)")
    with st.expander("Lighthouse (T5, T6): cómo medir"):
        st.markdown("En Chrome abre la app → clic derecho → **Inspeccionar** → pestaña **Lighthouse** → modo "
                    "**Móvil** → *Analyze page load*. O en una terminal: "
                    "`npx lighthouse http://localhost:8501 --preset=perf --form-factor=mobile`. "
                    "Escribe los puntajes aquí:")
        c = st.columns(4)
        perf = c[0].number_input("Rendimiento", 0, 100, 0, key="lh_perf")
        acc = c[1].number_input("Accesibilidad", 0, 100, 0, key="lh_acc")
        bp = c[2].number_input("Buenas prácticas", 0, 100, 0, key="lh_bp")
        seo = c[3].number_input("SEO", 0, 100, 0, key="lh_seo")
        if perf or acc or bp or seo:
            measured.update({"T6": float(perf), "T5": f"{acc} / {bp} / {seo}", "T5_min": min(acc, bp, seo)})
    with st.expander("Cuestionario SUS (T7): una respuesta por funcionario"):
        responses = [st.slider(f"{i + 1}. {p}", 1, 5, 3, key=f"sus_{i}",
                               help="1 = totalmente en desacuerdo · 5 = totalmente de acuerdo")
                     for i, p in enumerate(SUS_QUESTIONS)]
        if st.button("Agregar respuesta SUS"):
            st.session_state.setdefault("sus", []).append(kpi_audit.sus_score(responses))
        sus = st.session_state.get("sus", [])
        if sus:
            measured.update({"T7": float(np.mean(sus)), "T7_n": len(sus)})
            st.caption(f"Puntajes: {', '.join(f'{x:.1f}' for x in sus)} · promedio {np.mean(sus):.1f}")

    st.subheader("Tabla II · KPI de negocio (N) y técnicos (T)")
    table2 = kpi_audit.kpi_table(measured)
    st.dataframe(table2, hide_index=True, height=640)
    st.download_button("Descargar Tabla II (CSV)", table2.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"tabla_II_{scn['id']}.csv", mime="text/csv")


def data_page():
    e = state()
    base, scn = e["base"], e["scenario"]
    t = base["territory"]
    st.title("📋 Datos de prueba")
    st.caption("Datos sintéticos calibrados para La Libertad. En el piloto se reemplazan por DANE, IGAC, "
               "Copernicus DEM y SIMMA (ver data_generator.py).")
    pop = t["settlements"].drop(columns=["cell", "row", "col"]).rename(columns=DATA_LABELS)
    roads = t["roads"].drop(columns=["cells"])
    a, b, c = st.tabs([f"Poblados ({len(pop)})", f"Vías ({len(roads)})", f"SIMMA ({len(t['simma'])})"])
    with a:
        st.dataframe(pop, hide_index=True)
        st.download_button("Descargar CSV", pop.to_csv(index=False).encode("utf-8-sig"), "poblados.csv", "text/csv")
    with b:
        roads = roads.assign(status=["cerrada" if i in scn["params"]["closed"] else "abierta"
                                     for i in roads.index]).rename(columns=DATA_LABELS)
        st.dataframe(roads, hide_index=True)
        st.download_button("Descargar CSV", roads.to_csv(index=False).encode("utf-8-sig"), "vias.csv", "text/csv")
    with c:
        st.dataframe(t["simma"].drop(columns=["cell"]).rename(columns=DATA_LABELS), hide_index=True)
    export_cfg = {"scenario": scn["id"], "parameters": {k: v for k, v in scn["params"].items() if k != "id"},
                  "types": {kind: {"centers": r["centers"], "kpi": r["eval"]["kpi"], "steps": r["steps"]}
                         for kind, r in scn["types"].items()}}
    st.download_button("⬇️ Exportar escenario (JSON)", json.dumps(export_cfg, ensure_ascii=False, indent=2, default=str),
                       file_name=f"{scn['id']}.json", mime="application/json")
