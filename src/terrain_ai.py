import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import config

FEATURES = ["slope", "altitude", "curvature", "tpi", "river_dist_m"]
DISPLAY_NAMES = {
    "slope": "Pendiente (%)",
    "altitude": "Altura (m)",
    "curvature": "Curvatura (cima + / valle −)",
    "tpi": "Posición topográfica TPI (m)",
    "river_dist_m": "Distancia al río (m)",
}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acceso_cache")


def feature_table(territory):
    grid, relief = territory["grid"], territory["relief"]
    rows, cols = np.divmod(np.arange(grid["ny"] * grid["nx"]), grid["nx"])
    table = pd.DataFrame({v: relief[v].ravel() for v in FEATURES})
    table["label"] = territory["labels"].ravel()
    table["row"], table["col"] = rows, cols
    side = max(1, int(round(config.CV_BLOCK_SIZE_KM / grid["cell_km"])))
    table["block"] = (rows // side) * 1000 + (cols // side)
    return table


def new_forest():
    return RandomForestClassifier(
        n_estimators=config.NUM_TREES, max_depth=config.TREE_DEPTH,
        min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=config.SEED,
    )


def train_model(territory):
    table = feature_table(territory)
    sample = table.sample(frac=config.TRAINING_FRACTION, random_state=config.SEED)
    X, y, groups = sample[FEATURES], sample["label"], sample["block"]

    prob_oof = np.zeros((len(sample), 3))
    for learn, exam in GroupKFold(n_splits=config.CV_FOLDS).split(X, y, groups):
        forest = new_forest().fit(X.iloc[learn], y.iloc[learn])
        prob_oof[np.ix_(exam, forest.classes_)] = forest.predict_proba(X.iloc[exam])
    pred_oof = prob_oof.argmax(axis=1)

    recall_per_class = recall_score(y, pred_oof, labels=[0, 1, 2], average=None, zero_division=0)
    model = new_forest().fit(X, y)
    importances = pd.DataFrame({
        "variable": [DISPLAY_NAMES[v] for v in FEATURES],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return {
        "model": model,
        "auc": float(roc_auc_score(y, prob_oof, multi_class="ovr", average="macro")),
        "recall_unsuitable": float(recall_per_class[2]),
        "recall_macro": float(recall_per_class.mean()),
        "recall_by_class": recall_per_class.tolist(),
        "accuracy": float(accuracy_score(y, pred_oof)),
        "confusion_matrix": confusion_matrix(y, pred_oof, labels=[0, 1, 2]).tolist(),
        "importances": importances,
        "samples": len(sample),
        "blocks": int(groups.nunique()),
    }


def config_fingerprint():
    key_list = {k: getattr(config, k) for k in dir(config) if k.isupper() and not k.startswith(("LM_", "CLOUD_"))}
    text = json.dumps(key_list, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def train_or_load(territory):
    route = os.path.join(CACHE_DIR, f"terrain_model_{config_fingerprint()}.joblib")
    try:
        return joblib.load(route)
    except Exception:
        info = train_model(territory)
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            joblib.dump(info, route)
        except OSError:
            pass
        return info


def classify_cells(model, territory):
    table = feature_table(territory)
    grid = territory["grid"]
    prob = np.zeros((len(table), 3))
    prob[:, model.classes_] = model.predict_proba(table[FEATURES])
    suit_class = prob.argmax(axis=1)
    river = (territory["relief"]["main_river"] | territory["relief"]["stream"]).ravel()
    suit_class[river] = 2
    shape = (grid["ny"], grid["nx"])
    return suit_class.reshape(shape), prob.reshape(shape + (3,))


def cell_reasons(variables):
    m = []
    if variables["slope"] > config.SLOPE_UNSUITABLE:
        m.append(f"pendiente de {variables['slope']:.0f} % (riesgo de deslizamiento)")
    elif variables["slope"] > config.SLOPE_CONDITIONAL:
        m.append(f"pendiente de {variables['slope']:.0f} % (requiere terraceo)")
    if variables["tpi"] > config.PEAK_UNSUITABLE_M:
        m.append(f"cima o cuchilla de montaña ({variables['tpi']:.0f} m sobre su entorno)")
    elif variables["tpi"] > config.PEAK_CONDITIONAL_M:
        m.append(f"loma o filo ({variables['tpi']:.0f} m sobre su entorno)")
    if variables["altitude"] > config.ALTITUDE_UNSUITABLE:
        m.append(f"altura de {variables['altitude']:.0f} m (páramo)")
    elif variables["altitude"] > config.ALTITUDE_CONDITIONAL:
        m.append(f"altura de {variables['altitude']:.0f} m (alta montaña)")
    if variables["river_dist_m"] < config.RIVER_UNSUITABLE_M:
        m.append(f"a {variables['river_dist_m']:.0f} m del río (inundación)")
    elif variables["river_dist_m"] < config.RIVER_CONDITIONAL_M:
        m.append(f"a {variables['river_dist_m']:.0f} m del río (franja de amortiguación)")
    return m


def explain_cell(territory, suit_class, prob, cell):
    relief, grid = territory["relief"], territory["grid"]
    row, col = divmod(int(cell), grid["nx"])
    variables = {v: float(relief[v][row, col]) for v in FEATURES}
    reasons = cell_reasons(variables)
    if relief["simma"][row, col]:
        reasons.append("hay un movimiento en masa registrado (SIMMA)")
    return {
        "lat": round(float(grid["lat"][row]), 5), "lon": round(float(grid["lon"][col]), 5),
        "suitability": config.CLASSES[int(suit_class[row, col])],
        "probabilities": {c: round(float(p), 2) for c, p in zip(config.CLASSES, prob[row, col])},
        "variables": {DISPLAY_NAMES[k]: round(v, 1) for k, v in variables.items()},
        "reasons": reasons or ["ningún factor supera los límites de riesgo"],
    }


if __name__ == "__main__":
    import time

    import data_generator

    t = data_generator.generate_territory()
    start = time.perf_counter()
    info = train_model(t)
    print(f"Trained in {time.perf_counter() - start:.1f} s on {info['samples']} cells "
          f"({info['blocks']} spatial blocks)")
    print(f"AUC (macro, spatial CV): {info['auc']:.3f}   target >= {config.TARGETS['T8_auc']}")
    print(f"Recall unsuitable:        {info['recall_unsuitable']:.3f}   target >= {config.TARGETS['T8_recall']}")
    print(f"Accuracy:                 {info['accuracy']:.1%}")
    print("\nFeature importance:")
    print(info["importances"].to_string(index=False))
    suit_class, prob = classify_cells(info["model"], t)
    print("\nTerritory classification:",
          {config.CLASSES[c]: f"{(suit_class == c).mean():.0%}" for c in range(3)})
