import os

import streamlit as st

import config
import scenarios
import ui_components as ui

st.set_page_config(page_title="ACCESO · ODS 10", page_icon="🧭", layout="wide")


SECRET_NAMES = {"ACCESO_CLOUD_URL": "CLOUD_URL", "ACCESO_CLOUD_API_KEY": "CLOUD_API_KEY",
                "ACCESO_CLOUD_MODEL": "CLOUD_MODEL", "ACCESO_LM_STUDIO_URL": "LM_STUDIO_URL"}


def read_secrets():
    try:
        for key in ("ACCESO_CLOUD_URL", "ACCESO_CLOUD_API_KEY", "ACCESO_CLOUD_MODEL", "ACCESO_LM_STUDIO_URL"):
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
                setattr(config, SECRET_NAMES[key], str(st.secrets[key]))
    except Exception:
        pass


@st.cache_resource(show_spinner="Preparando el territorio, la IA de terreno y la red de rutas (solo la primera vez)...")
def load_base():
    return scenarios.prepare_base()


@st.cache_resource(max_entries=64, show_spinner="Calculando el escenario...")
def compute(params):
    return scenarios.compute_scenario(load_base(), params)


def main():
    read_secrets()
    pages = [
        st.Page(ui.map_page, title="Mapa y centros", icon="🗺️", url_path="mapa", default=True),
        st.Page(ui.relief_3d_page, title="Relieve 3D", icon="⛰️", url_path="relieve-3d"),
        st.Page(ui.terrain_ai_page, title="IA de terreno", icon="🧠", url_path="ia-terreno"),
        st.Page(ui.compare_page, title="Comparar estrategias", icon="⚖️", url_path="comparar"),
        st.Page(ui.assistant_page, title="Asistente IA", icon="💬", url_path="asistente"),
        st.Page(ui.kpi_page, title="Auditoría KPI", icon="📊", url_path="kpi"),
        st.Page(ui.data_page, title="Datos", icon="📋", url_path="datos"),
    ]
    page = st.navigation(pages)
    base = load_base()
    params = ui.sidebar(base)
    st.session_state["acceso"] = {"base": base, "params": params, "scenario": compute(params), "compute": compute}
    page.run()


main()
