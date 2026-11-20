import difflib
import json
import re
import time
import unicodedata

import numpy as np

import config
import scenarios
import terrain_ai


SYSTEM_INSTRUCTIONS = """Eres el asistente de ACCESO, una herramienta de planeación territorial para las alcaldías \
rurales de la Provincia de La Libertad (Boyacá, Colombia), alineada con el ODS 10.
Tu trabajo es EXPLICAR los resultados del motor de ubicación al funcionario. No decides ni calculas.

REGLAS OBLIGATORIAS:
1. Usa ÚNICAMENTE cifras que aparezcan en el RESUMEN DEL ESCENARIO o en los resultados de las herramientas.
2. No calcules, no estimes y no redondees por tu cuenta (ni sumas, ni restas, ni porcentajes). Si necesitas un \
dato que no está, llama a una herramienta. Si ninguna lo da, di claramente que no tienes ese dato.
3. Nunca inventes nombres de veredas, vías o centros.
4. Termina SIEMPRE con la cita del escenario, así: [Escenario ESC-XXXXXX]. Si simulaste otro escenario, cita los dos.
5. Escribe en español sencillo, máximo 150 palabras, con números con punto decimal y sin separador de miles.
6. Los tiempos están en minutos y las distancias en km por el camino real (no en línea recta).
7. Si te preguntan algo ajeno a ACCESO, responde en una frase que solo ayudas con este análisis.

RESUMEN DEL ESCENARIO (datos calculados por el motor de ACCESO):
"""

REPORT_INSTRUCTIONS = (
    "Con los DATOS de abajo, escribe un informe en Markdown con exactamente tres secciones: "
    "1) Dónde se ubicaron los centros y a quién benefician; 2) Qué terrenos descartó la IA y por qué; "
    "3) Cuánto se acorta el peor viaje frente a ubicar en las comunidades más pobladas. "
    "Cierra con una recomendación de dos o tres frases. Usa SOLO las cifras de los datos y cita el escenario."
)


TOOLS = [
    {"type": "function", "function": {
        "name": "scenario_summary",
        "description": "Indicadores y centros del escenario actual (peor viaje, cobertura, centros por tipo).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "explain_center",
        "description": "Explica por qué un centro quedó donde quedó: regla aplicada, terreno, comunidades atendidas.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(config.CENTER_TYPES)},
            "number": {"type": "integer", "description": "Número del centro (1, 2, 3...)"}},
            "required": ["kind", "number"]}}},
    {"type": "function", "function": {
        "name": "query_community",
        "description": "Datos de una vereda o cabecera: tiempo y km a su centro, centro asignado, y si su terreno "
                       "permite construir (con los motivos de la IA de terreno).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Nombre de la vereda o cabecera, p. ej. 'El Cedro'"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "simulate_road_closure",
        "description": "Recalcula el escenario si se cierra una vía o cae un puente (HU-04) y compara antes/después.",
        "parameters": {"type": "object", "properties": {
            "road": {"type": "string", "description": "Nombre del tramo (p. ej. 'Pisba ↔ Paya') o su número"}},
            "required": ["road"]}}},
    {"type": "function", "function": {
        "name": "list_roads",
        "description": "Lista tramos viales con su número, longitud y si tienen puente. Útil antes de simular un cierre.",
        "parameters": {"type": "object", "properties": {
            "bridges_only": {"type": "boolean"},
            "text": {"type": "string", "description": "Filtra por nombre de vereda o municipio"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "change_center_count",
        "description": "Recalcula qué pasaría con otro número de centros de un tipo (qué pasa si...).",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(config.CENTER_TYPES)},
            "k": {"type": "integer", "minimum": 0, "maximum": config.K_MAX_PER_KIND}},
            "required": ["kind", "k"]}}},
    {"type": "function", "function": {
        "name": "compare_strategies",
        "description": "Tabla III: compara ACCESO con ubicar en las más pobladas, al azar, sin IA de terreno y "
                       "con el criterio de cobertura máxima.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(config.CENTER_TYPES)}}, "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "terrain_suitability",
        "description": "Explica la aptitud del terreno (Apto, Condicionado, No apto) en unas coordenadas.",
        "parameters": {"type": "object", "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number"}}, "required": ["lat", "lon"]}}},
]


def r1(x):
    if x is None:
        return None
    x = float(x)
    return "sin ruta" if not np.isfinite(x) else round(x, 1)


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


class Context:

    def __init__(self, base, scenario, recompute=None):
        self.base = base
        self.scenario = scenario
        self.recompute = recompute or (lambda p: scenarios.compute_scenario(base, p))
        self._comparisons = {}

    @property
    def id(self):
        return self.scenario["id"]

    def default_kind(self):
        return next(iter(self.scenario["types"]), next(iter(config.CENTER_TYPES)))

    def compare(self, kind):
        if kind not in self._comparisons:
            self._comparisons[kind] = scenarios.compare_strategies(self.base, self.scenario, kind, draws=10)
        return self._comparisons[kind]


def scenario_summary(ctx):
    scn, base = ctx.scenario, ctx.base
    p = scn["params"]
    roads = base["territory"]["roads"]
    output = {
        "scenario": scn["id"], "travel_mode": p["mode"], "criterion": p["criterion"],
        "refined_to_central_suitable_point": p["refine"],
        "closed_roads": [roads.loc[c, "name"] for c in p["closed"]],
        "total_population": int(base["population"].sum()), "communities": len(base["demand"]),
        "terrain_ai": {"auc": round(base["ai"]["auc"], 3), "recall_unsuitable": round(base["ai"]["recall_unsuitable"], 3)},
        "types": {},
    }
    for kind, r in scn["types"].items():
        kp, kb = r["eval"]["kpi"], scn["bases"][kind]["eval"]["kpi"]
        farthest = r["eval"]["farthest"]
        output["types"][kind] = {
            "centers": [{"number": c["number"], "location": c["location"], "ai_terrain": c["ai_suitability"],
                         "altitude_m": round(c["altitude"]), "slope_pct": r1(c["slope"]),
                         "communities_served": c["communities"], "inhabitants": c["inhabitants"],
                         "worst_trip_min": r1(c["worst_time_min"]), "worst_distance_km": r1(c["worst_km"]),
                         "existing": c["existing"]} for c in r["centers"]],
            "worst_trip_min": r1(kp["max_time_min"]), "worst_distance_km": r1(kp["max_km"]),
            "farthest_community": base["demand"].iloc[farthest]["name"] if farthest is not None else None,
            "avg_time_min": r1(kp["mean_time_min"]),
            "pct_population_under_1h_walk": r1(kp["pct_under_1h_walk"]),
            "pct_population_over_2h": r1(kp["pct_over_2h"]), "pct_population_over_25km": r1(kp["pct_over_25km"]),
            "rural_urban_gap": round(kp["rural_urban_gap"], 2),
            "communities_without_route": kp["no_route"],
            "usual_practice_most_populated": {
                "worst_trip_min": r1(kb["max_time_min"]), "worst_distance_km": r1(kb["max_km"]),
                "pct_population_under_1h_walk": r1(kb["pct_under_1h_walk"])},
            "worst_trip_reduction_min": r1(kb["max_time_min"] - kp["max_time_min"]),
            "worst_trip_reduction_pct": r1(100 * (kb["max_time_min"] - kp["max_time_min"]) / kb["max_time_min"])
            if kb["max_time_min"] > 0 else None,
            "pct_territory_suitable_for_type": r1(r["allowed_pct"]),
        }
    return output


def find_community(ctx, name):
    names = ctx.base["demand"]["name"].tolist()
    cleaned = [normalize(n.replace("Vereda ", "").replace(" (cabecera)", "")) for n in names]
    target_value = normalize(name).replace("vereda ", "").replace("(cabecera)", "").strip()
    if target_value in cleaned:
        return cleaned.index(target_value)
    similar = difflib.get_close_matches(target_value, cleaned, n=1, cutoff=0.72)
    return cleaned.index(similar[0]) if similar else None


def communities_in_text(ctx, text):
    t = normalize(text)
    names = ctx.base["demand"]["name"].tolist()
    found = []
    for i, n in sorted(enumerate(names), key=lambda x: -len(x[1])):
        clean = normalize(n.replace("Vereda ", "").replace(" (cabecera)", ""))
        if re.search(r"\b" + re.escape(clean) + r"\b", t):
            found.append(i)
            t = t.replace(clean, " ")
    return found


def explain_center(ctx, kind, number):
    r = ctx.scenario["types"].get(kind)
    if r is None:
        return {"error": f"El escenario no tiene centros de tipo {kind}."}
    if not 1 <= int(number) <= len(r["centers"]):
        return {"error": f"Solo hay {len(r['centers'])} centros de tipo {kind}."}
    c = r["centers"][int(number) - 1]
    terrain = terrain_ai.explain_cell(ctx.base["territory"], ctx.base["suitability"], ctx.base["prob"], c["node"])
    steps = [p for p in r["steps"] if p.startswith(f"Centro {number} ") or f"Centro {number}:" in p]
    return {"scenario": ctx.id, "type": kind, "number": c["number"], "location": c["location"],
            "applied_rule": c["reason"], "algorithm_steps": steps,
            "moved_from": c["moved_from"], "terrain": terrain,
            "communities_served": c["communities"], "inhabitants": c["inhabitants"],
            "worst_trip_min": r1(c["worst_time_min"]), "worst_distance_km": r1(c["worst_km"])}


def query_community(ctx, name):
    i = find_community(ctx, name)
    if i is None:
        return {"error": f"No encontré la comunidad '{name}'."}
    row = ctx.base["demand"].iloc[i]
    terrain = terrain_ai.explain_cell(ctx.base["territory"], ctx.base["suitability"], ctx.base["prob"], row["cell"])
    output = {"scenario": ctx.id, "community": row["name"], "municipality": row["municipality"],
              "population": int(row["population"]), "altitude_m": int(row["altitude"]),
              "community_terrain": terrain, "by_type": {}}
    for kind, r in ctx.scenario["types"].items():
        ev = r["eval"]
        j = int(ev["assignment"][i])
        allowed = config.CENTER_TYPES[kind]["allowed_classes"]
        output["by_type"][kind] = {
            "time_to_center_min": r1(ev["time"][i]), "distance_to_center_km": r1(ev["km"][i]),
            "walk_time_min": r1(ev["walk"][i]),
            "assigned_center": f"{kind} {j + 1} ({r['centers'][j]['location']})" if j >= 0 else "sin ruta",
            "is_farthest": ev["farthest"] == i,
            "terrain_allows_type": config.CLASSES.index(terrain["suitability"]) in allowed,
            "has_own_center": any(c["node"] == row["cell"] for c in r["centers"]),
        }
    return output


def find_road(ctx, road):
    roads = ctx.base["territory"]["roads"]
    text = str(road).strip()
    if text.isdigit() and int(text) < len(roads):
        return int(text)
    t = normalize(text)
    best, score = None, 0
    for i, name in enumerate(roads["name"]):
        parts = [normalize(p.replace("🌉", "").strip()) for p in name.split("↔")]
        p = sum(1 for part in parts if part and part in t) + 0.5 * ("puente" in t and roads.loc[i, "puentes"] > 0)
        if p > score:
            best, score = i, p
    if best is None and "puente" in t:
        with_bridge = roads[roads["bridges"] > 0].sort_values(["type", "length_km"], ascending=[True, False])
        best = int(with_bridge.index[0]) if len(with_bridge) else None
    return best


def simulate_road_closure(ctx, road):
    t = find_road(ctx, road)
    if t is None:
        return {"error": f"No encontré la vía '{road}'. Usa list_roads para ver los nombres."}
    p = ctx.scenario["params"]
    roads = ctx.base["territory"]["roads"]
    new_params = scenarios.parameters(dict(p["ks"]), p["mode"], p["criterion"], p["refine"],
                                      tuple(p["closed"]) + (t,), p["fixed_centers"])
    start = time.perf_counter()
    new = ctx.recompute(new_params)
    ms = (time.perf_counter() - start) * 1000
    output = {"current_scenario": ctx.id, "simulated_scenario": new["id"],
              "closed_road": roads.loc[t, "name"], "had_bridge": bool(roads.loc[t, "bridges"] > 0),
              "recompute_ms": round(ms), "by_type": {}}
    names = ctx.base["demand"]["name"]
    for kind, r in ctx.scenario["types"].items():
        if kind not in new["types"]:
            continue
        a, d = r["eval"]["kpi"], new["types"][kind]["eval"]["kpi"]
        no_route = [names[i] for i in np.flatnonzero(~np.isfinite(new["types"][kind]["eval"]["time"]))]
        before = {c["location"] for c in r["centers"]}
        after = [c["location"] for c in new["types"][kind]["centers"]]
        output["by_type"][kind] = {
            "worst_trip_before_min": r1(a["max_time_min"]), "worst_trip_after_min": r1(d["max_time_min"]),
            "worst_trip_increase_min": r1(d["max_time_min"] - a["max_time_min"]),
            "worst_distance_before_km": r1(a["max_km"]), "worst_distance_after_km": r1(d["max_km"]),
            "communities_without_route": no_route,
            "centers_changed": set(after) != before, "new_centers": [u for u in after if u not in before],
        }
    return output


def list_roads(ctx, bridges_only=False, text=""):
    roads = ctx.base["territory"]["roads"]
    if bridges_only:
        roads = roads[roads["bridges"] > 0]
    if text:
        t = normalize(text)
        roads = roads[[t in normalize(n) for n in roads["name"]]]
    return {"scenario": ctx.id, "segments": [
        {"number": int(i), "name": f["name"], "type": f["type"], "length_km": r1(f["length_km"]),
         "bridge": bool(f["bridges"] > 0), "is_closed": int(i) in ctx.scenario["params"]["closed"]}
        for i, f in roads.head(25).iterrows()], "total_found": len(roads)}


def change_center_count(ctx, kind, k):
    p = ctx.scenario["params"]
    ks = dict(p["ks"])
    before = ks.get(kind, 0)
    ks[kind] = int(np.clip(int(k), 0, config.K_MAX_PER_KIND))
    new = ctx.recompute(scenarios.parameters(ks, p["mode"], p["criterion"], p["refine"], p["closed"], p["fixed_centers"]))
    output = {"current_scenario": ctx.id, "simulated_scenario": new["id"], "type": kind,
              "k_before": before, "k_after": ks[kind]}
    if kind in ctx.scenario["types"]:
        a = ctx.scenario["types"][kind]["eval"]["kpi"]
        output.update({"worst_trip_before_min": r1(a["max_time_min"]),
                       "pct_under_1h_walk_before": r1(a["pct_under_1h_walk"])})
    if kind in new["types"]:
        d = new["types"][kind]["eval"]["kpi"]
        output.update({"worst_trip_after_min": r1(d["max_time_min"]),
                       "pct_under_1h_walk_after": r1(d["pct_under_1h_walk"]),
                       "centers": [c["location"] for c in new["types"][kind]["centers"]]})
    return output


def compare_strategies(ctx, kind):
    if kind not in ctx.scenario["types"]:
        return {"error": f"El escenario no tiene centros de tipo {kind}."}
    table, n4 = ctx.compare(kind)
    rows = [{k: (r1(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}
            for row in table.to_dict("records")]
    return {"scenario": ctx.id, "type": kind, "strategies": rows,
            "without_ai_pct_centers_unsuitable": r1(n4["pct_without_ai_unsuitable"]),
            "cost_of_avoiding_unsuitable_pct": r1(n4["cost_pct"])}


def terrain_suitability(ctx, lat, lon):
    grid = ctx.base["territory"]["grid"]
    row = int(np.argmin(np.abs(grid["lat"] - float(lat))))
    col = int(np.argmin(np.abs(grid["lon"] - float(lon))))
    return {"scenario": ctx.id, **terrain_ai.explain_cell(
        ctx.base["territory"], ctx.base["suitability"], ctx.base["prob"], row * grid["nx"] + col)}


FUNCTIONS = {
    "scenario_summary": lambda ctx, **a: scenario_summary(ctx),
    "explain_center": lambda ctx, **a: explain_center(ctx, a.get("kind", ctx.default_kind()), a.get("number", 1)),
    "query_community": lambda ctx, **a: query_community(ctx, a.get("name", "")),
    "simulate_road_closure": lambda ctx, **a: simulate_road_closure(ctx, a.get("road", "")),
    "list_roads": lambda ctx, **a: list_roads(ctx, bool(a.get("bridges_only", False)), a.get("text", "")),
    "change_center_count": lambda ctx, **a: change_center_count(ctx, a.get("kind", ctx.default_kind()),
                                                                   a.get("k", 1)),
    "compare_strategies": lambda ctx, **a: compare_strategies(ctx, a.get("kind", ctx.default_kind())),
    "terrain_suitability": lambda ctx, **a: terrain_suitability(ctx, a.get("lat", 0), a.get("lon", 0)),
}


def run_tool(ctx, name, arguments):
    if name not in FUNCTIONS:
        return {"error": f"Herramienta desconocida: {name}"}
    try:
        return FUNCTIONS[name](ctx, **(arguments or {}))
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


NUMBER_PATTERN = re.compile(r"(?<![\w-])(\d{1,3}(?:\.\d{3})+(?![\d.,])|\d+(?:[.,]\d+)?)")


def numbers_in(text):
    text = re.sub(r"ESC-[0-9A-F]{6}", " ", str(text))
    output = []
    for m in NUMBER_PATTERN.findall(text):
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", m) and len(m.split(".")[-1]) == 3 and m.count(".") >= 1 \
                and int(m.split(".")[0]) > 0 and len(m) > 5:
            output.append(float(m.replace(".", "")))
        else:
            output.append(float(m.replace(",", ".")))
    return output


def verify_figures(text, sources):
    known = np.array(numbers_in(json.dumps(sources, ensure_ascii=False, default=str)) or [np.nan])
    unverified = []
    figures = numbers_in(text)
    for x in figures:
        if x <= 10 and float(x).is_integer():
            continue
        tolerance = 0.55 if float(x).is_integer() else 0.051
        if not np.any(np.abs(known - x) <= tolerance):
            unverified.append(x)
    return {"total": len(figures), "unverified": unverified}


def contains_value(text, expected):
    if isinstance(expected, str):
        target_value = normalize(expected.replace("Vereda ", "").replace(" (cabecera)", ""))
        return target_value in normalize(text)
    return any(abs(x - float(expected)) <= 0.55 for x in numbers_in(text))


def read_settings(settings=None):
    a = {"lm_url": config.LM_STUDIO_URL, "lm_model": config.LM_STUDIO_MODEL,
         "cloud_url": config.CLOUD_URL, "cloud_key": config.CLOUD_API_KEY, "cloud_model": config.CLOUD_MODEL,
         "use_cloud": True, "timeout": config.RESPONSE_TIMEOUT_S}
    a.update({k: v for k, v in (settings or {}).items() if v not in (None, "")})
    return a


def create_client(url, key, wait_time):
    from openai import OpenAI
    return OpenAI(base_url=url, api_key=key or "sin-clave", timeout=wait_time, max_retries=0)


def detect_provider(settings=None):
    a = read_settings(settings)
    reason = ""
    try:
        client = create_client(a["lm_url"], config.LM_STUDIO_API_KEY, config.CONNECT_TIMEOUT_S)
        models = [m.id for m in client.models.list().data if "embed" not in m.id.lower()]
        if models:
            model = a["lm_model"] or models[0]
            return {"name": "LM Studio", "url": a["lm_url"], "api_key": config.LM_STUDIO_API_KEY,
                    "model": model, "message": f"Conectado a LM Studio · modelo {model}"}
        reason = "LM Studio está encendido pero no tiene ningún modelo cargado."
    except ImportError:
        reason = "Falta la librería openai (pip install openai)."
    except Exception:
        reason = f"No se encontró LM Studio en {a['lm_url']}."
    if a["use_cloud"] and a["cloud_key"] and a["cloud_model"]:
        return {"name": "Nube", "url": a["cloud_url"], "api_key": a["cloud_key"], "model": a["cloud_model"],
                "message": f"{reason} Se usará el modelo en la nube {a['cloud_model']}."}
    return {"name": None, "message": f"{reason} El chat funciona en modo guiado (sin modelo de lenguaje)."}


def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def is_connection_error(error):
    name = type(error).__name__
    return isinstance(error, OSError) or name in {"APIConnectionError", "APITimeoutError", "ConnectError",
                                                     "ConnectTimeout", "ReadTimeout", "TimeoutException"}


def build_messages(question, history, ctx):
    summary = scenario_summary(ctx)
    system = SYSTEM_INSTRUCTIONS + json.dumps(summary, ensure_ascii=False, indent=1)
    recent = [{"role": m["role"], "content": m["content"]} for m in history[-6:]
              if m.get("role") in ("user", "assistant")]
    return [{"role": "system", "content": system}, *recent, {"role": "user", "content": question}], summary


def respond_with_model(question, history, ctx, provider, settings=None):
    a = read_settings(settings)
    client = create_client(provider["url"], provider["api_key"], float(a["timeout"]))
    messages, summary = build_messages(question, history, ctx)
    sources, used = [summary], []
    use_tools = True
    text = ""
    for _ in range(config.MAX_TOOL_ROUNDS + 1):
        arguments = {"model": provider["model"], "messages": messages, "temperature": config.TEMPERATURE}
        if use_tools:
            arguments.update(tools=TOOLS, tool_choice="auto")
        try:
            response = client.chat.completions.create(**arguments)
        except Exception as error:
            if use_tools and not is_connection_error(error):
                use_tools = False
                continue
            raise
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            text = clean_response(message.content)
            break
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
            {"id": ll.id, "type": "function", "function": {"name": ll.function.name,
                                                           "arguments": ll.function.arguments or "{}"}}
            for ll in calls]})
        for ll in calls:
            try:
                args = json.loads(ll.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(ctx, ll.function.name, args)
            sources.append(result)
            used.append({"tool": ll.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": ll.id,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
    if not text:
        raise RuntimeError("El modelo no entregó una respuesta final.")
    return text, sources, used


def detect_kind(ctx, text):
    t = normalize(text)
    for key, kind in (("hospital", "Hospital"), ("colegio", "Colegio"), ("escuela", "Colegio"),
                      ("educa", "Colegio"), ("salud", "Centro de salud"), ("puesto", "Centro de salud")):
        if key in t and kind in ctx.scenario["types"]:
            return kind
    return ctx.default_kind()


def summary_template(d, kind=None):
    lines = [f"Modo: {d['travel_mode']}. Criterio: {d['criterion']}."]
    for t, v in d["types"].items():
        if kind and t != kind:
            continue
        b = v["usual_practice_most_populated"]
        lines.append(
            f"**{t}** ({len(v['centers'])} centro{'s' if len(v['centers']) != 1 else ''}): el peor viaje es de {v['worst_trip_min']} min "
            f"({v['worst_distance_km']} km por camino), en {v['farthest_community']}. "
            f"Ubicando en las comunidades más pobladas sería de {b['worst_trip_min']} min "
            f"(reducción de {v['worst_trip_reduction_min']} min, {v['worst_trip_reduction_pct']} %). "
            f"Población a menos de 1 h a pie: {v['pct_population_under_1h_walk']} % "
            f"(más pobladas: {b['pct_population_under_1h_walk']} %).")
        for c in v["centers"]:
            lines.append(f"- {t} {c['number']}: {c['location']}, terreno {c['ai_terrain']}, "
                         f"atiende {c['communities_served']} comunidades ({c['inhabitants']} hab).")
    if d["closed_roads"]:
        lines.append("Vías cerradas: " + ", ".join(d["closed_roads"]) + ".")
    return "\n".join(lines)


def respond_offline(question, ctx):
    t = normalize(question)
    kind = detect_kind(ctx, question)
    used = []

    def use_tool(tool, **args):
        result = run_tool(ctx, tool, args)
        used.append({"tool": tool, "args": args, "result": result})
        return result

    communities = communities_in_text(ctx, question)
    number = re.search(r"centro\s*(?:n[uú]mero\s*|#\s*)?(\d+)|(?:numero|#)\s*(\d+)", t)

    if any(p in t for p in ("cierr", "cae", "cayo", "puente", "derrumbe", "bloque")):
        r = use_tool("simulate_road_closure", road=question)
        if "error" in r:
            text = r["error"]
        else:
            parts = [f"Si se cierra **{r['closed_road']}** (recalculado en {r['recompute_ms']} ms, "
                     f"escenario simulado {r['simulated_scenario']}):"]
            for tp, v in r["by_type"].items():
                sr = (" Quedarían sin ruta: " + ", ".join(v["communities_without_route"]) + ".") if v["communities_without_route"] else ""
                change = " Los centros se reubican: " + ", ".join(v["new_centers"]) + "." if v["centers_changed"] \
                    else " Los centros no cambian."
                parts.append(f"- {tp}: el peor viaje pasa de {v['worst_trip_before_min']} a "
                             f"{v['worst_trip_after_min']} min.{sr}{change}")
            text = "\n".join(parts)
    elif any(p in t for p in ("compar", "estrategia", "azar", "habitual", "sin ia")):
        r = use_tool("compare_strategies", kind=kind)
        if "error" in r:
            text = r["error"]
        else:
            col = next(k for k in r["strategies"][0] if k.startswith("Peor viaje"))
            rows = [f"- {e['Estrategia']}: peor viaje {e[col]}, menos de 1 h a pie {e['Pobl. <1 h a pie (%)']} %"
                    for e in r["strategies"]]
            text = (f"Comparación para {kind}:\n" + "\n".join(rows)
                    + f"\nSin IA de terreno, el {r['without_ai_pct_centers_unsuitable']} % de los centros caería en "
                       f"terreno No apto.")
    elif number and "comunidad" not in t:
        n = int(number.group(1) or number.group(2))
        r = use_tool("explain_center", kind=kind, number=n)
        if "error" in r:
            text = r["error"]
        else:
            text = (f"**{kind} {r['number']}** quedó {r['location']}. {r['applied_rule']} "
                    f"Terreno: {r['terrain']['suitability']} ({'; '.join(r['terrain']['reasons'])}). "
                    f"Atiende {r['communities_served']} comunidades ({r['inhabitants']} hab); su peor viaje es "
                    f"de {r['worst_trip_min']} min ({r['worst_distance_km']} km).")
    elif communities:
        r = use_tool("query_community", name=ctx.base["demand"].iloc[communities[0]]["name"])
        parts = [f"**{r['community']}** ({r['municipality']}, {r['population']} hab, {r['altitude_m']} m). "
                 f"Terreno: {r['community_terrain']['suitability']} "
                 f"({'; '.join(r['community_terrain']['reasons'])})."]
        for tp, v in r["by_type"].items():
            note = " Es la comunidad más alejada." if v["is_farthest"] else ""
            if not v["terrain_allows_type"] and not v["has_own_center"]:
                note += f" Su terreno no permite construir un {tp.lower()}."
            parts.append(f"- {tp}: {v['time_to_center_min']} min ({v['distance_to_center_km']} km) "
                         f"hasta {v['assigned_center']}; a pie {v['walk_time_min']} min.{note}")
        text = "\n".join(parts)
    elif any(p in t for p in ("si pongo", "si fueran", "con ", "agrega", "mas centros")) and re.search(r"\d+", t):
        k = int(re.search(r"\d+", t).group())
        r = use_tool("change_center_count", kind=kind, k=k)
        text = (f"Con {r['k_after']} centros de tipo {kind} (escenario {r['simulated_scenario']}), el peor viaje "
                f"pasaría de {r.get('worst_trip_before_min', '–')} a {r.get('worst_trip_after_min', '–')} min y "
                f"la población a menos de 1 h a pie de {r.get('pct_under_1h_walk_before', '–')} % a "
                f"{r.get('pct_under_1h_walk_after', '–')} %.")
    else:
        d = use_tool("scenario_summary")
        text = summary_template(d, kind if any(p in t for p in ("salud", "colegio", "hospital")) else None)
    return text, [u["result"] for u in used], used


def respond(question, history, ctx, settings=None, force_offline=False, provider=None):
    start = time.perf_counter()
    provider = provider or ({"name": None, "message": "Modo guiado forzado."} if force_offline
                            else detect_provider(settings))
    note = ""
    if provider["name"]:
        try:
            text, sources, used = respond_with_model(question, history, ctx, provider, settings)
            source, model = provider["name"], provider["model"]
        except Exception as error:
            note = f"El modelo falló ({type(error).__name__}); respuesta en modo guiado."
            provider = {"name": None}
    if not provider["name"]:
        text, sources, used = respond_offline(question, ctx)
        source, model = "Modo guiado", "plantillas del motor"
    cited = set(re.findall(r"ESC-[0-9A-F]{6}", text))
    if ctx.id not in cited:
        text += f"\n\n[Escenario {ctx.id}]"
    return {"text": text, "source": source, "model": model, "tools": used,
            "latency_s": time.perf_counter() - start, "verification": verify_figures(text, sources),
            "scenario": ctx.id, "note": note}


def guided_questions(ctx):
    kind = ctx.default_kind()
    r = ctx.scenario["types"].get(kind)
    farthest = ctx.base["demand"].iloc[r["eval"]["farthest"]]["name"] if r and r["eval"]["farthest"] is not None else None
    questions = ["Resume el escenario actual",
                 f"¿Por qué el {kind.lower()} número 1 quedó ahí?",
                 f"Compara las estrategias para {kind.lower()}",
                 "¿Qué pasa si se cae el puente principal?"]
    if farthest:
        questions.insert(2, f"¿Cómo queda {farthest.replace('Vereda ', 'la vereda ')}?")
    return questions


def question_bank(ctx, n=50):
    bank = []
    d = scenario_summary(ctx)
    for kind, v in d["types"].items():
        bank += [
            {"question": f"¿Cuál es el peor tiempo de viaje hasta un {kind.lower()}?", "expected": v["worst_trip_min"]},
            {"question": f"¿Cuál es la peor distancia por camino hasta un {kind.lower()}?",
             "expected": v["worst_distance_km"]},
            {"question": f"¿Qué porcentaje de la población queda a menos de 1 h a pie de un {kind.lower()}?",
             "expected": v["pct_population_under_1h_walk"]},
            {"question": f"¿Cuál es la comunidad más alejada de un {kind.lower()}?",
             "expected": v["farthest_community"]},
            {"question": f"¿En qué terreno quedó el {kind.lower()} número 1?",
             "expected": v["centers"][0]["ai_terrain"]},
        ]
    kind = ctx.default_kind()
    ev = ctx.scenario["types"][kind]["eval"]
    for i, row in ctx.base["demand"].iterrows():
        if len(bank) >= n:
            break
        if np.isfinite(ev["time"][i]):
            name = row["name"].replace("Vereda ", "la vereda ")
            bank.append({"question": f"¿A cuántos minutos queda {name} de su {kind.lower()}?",
                         "expected": round(float(ev["time"][i]), 1)})
    return bank[:n]


def report_facts(ctx):
    d = scenario_summary(ctx)
    parts = [f"## Informe ACCESO · {d['scenario']}", summary_template(d)]
    for kind in d["types"]:
        r = compare_strategies(ctx, kind)
        if "error" not in r:
            parts.append(f"Sin IA de terreno, el {r['without_ai_pct_centers_unsuitable']} % de los centros de tipo "
                         f"{kind} caería en terreno No apto.")
    suit_class = ctx.base["suitability"]
    parts.append(f"La IA de terreno clasificó el territorio en: Apto {r1(100 * (suit_class == 0).mean())} %, "
                 f"Condicionado {r1(100 * (suit_class == 1).mean())} %, No apto {r1(100 * (suit_class == 2).mean())} % "
                 f"(AUC {d['terrain_ai']['auc']}, recall No apto {d['terrain_ai']['recall_unsuitable']}).")
    return "\n\n".join(parts)


def generate_report(ctx, settings=None):
    facts = report_facts(ctx)
    provider = detect_provider(settings)
    if provider["name"]:
        try:
            client = create_client(provider["url"], provider["api_key"], max(30.0, float(read_settings(settings)["timeout"])))
            response = client.chat.completions.create(
                model=provider["model"], temperature=config.TEMPERATURE,
                messages=[{"role": "system", "content": SYSTEM_INSTRUCTIONS.split("RESUMEN DEL ESCENARIO")[0]},
                          {"role": "user", "content": REPORT_INSTRUCTIONS + "\n\nDATOS:\n" + facts}])
            text = clean_response(response.choices[0].message.content)
            if text:
                return {"text": text, "source": f"{provider['name']} · {provider['model']}",
                        "verification": verify_figures(text, [facts])}
        except Exception:
            pass
    text = facts + "\n\n### Recomendación\nVerificar en campo los sitios propuestos antes de construir y " \
                     "repetir el cálculo si cambia el estado de alguna vía."
    return {"text": text, "source": "Plantilla local", "verification": verify_figures(text, [facts])}


if __name__ == "__main__":
    base = scenarios.prepare_base()
    ctx = Context(base, scenarios.compute_scenario(base, scenarios.parameters()))
    print(detect_provider()["message"])
    for question in guided_questions(ctx)[:3]:
        r = respond(question, [], ctx)
        print(f"\n> {question}\n[{r['source']} · {r['latency_s']:.2f} s · "
              f"unverified figures: {r['verification']['unverified']}]\n{r['text']}")
