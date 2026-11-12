"""
CosmicFNO - Plataforma Visual e Observatorio SciML (PT / EN / ES)
Autora: Shayene Figueredo (BSI/UFU)

Design System regido pelas 3 Skills instaladas em .agents/skills/:
    1. ui-ux-pro-max (Data-Dense Scientific Observatory, contraste > 4.5:1, tabular-nums)
    2. frontend-design (Anthropic: sem eyebrows em CAIXA ALTA, sem cliches de pills repetitivas, hierarquia editorial limpa)
    3. antigravity-design-expert (Weightless floating cards com sombras difusas 0 20px 40px rgba(0,0,0,0.05), glassmorphism sutil e animacao staggered com prefers-reduced-motion)
"""

import os
import sys
import base64
import importlib.util
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PASTA_SRC = os.path.dirname(os.path.abspath(__file__))
PASTA_BASE = os.path.dirname(PASTA_SRC)
CAMINHO_LOGO = os.path.join(PASTA_BASE, "docs", "assets", "Cosmicfno-LOGO.png")

st.set_page_config(
    page_title="CosmicFNO | Gravitational Wave Observatory",
    page_icon=CAMINHO_LOGO if os.path.exists(CAMINHO_LOGO) else None,
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def carregar_logo_base64():
    if os.path.exists(CAMINHO_LOGO):
        with open(CAMINHO_LOGO, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


logo_b64 = carregar_logo_base64()

# =====================================================================
# 1. DICIONARIO MULTILINGUE (SENTENCE CASE EDITORIAL — ANTHROPIC SKILL)
# =====================================================================
I18N = {
    "PT": {
        "sec_main": "Módulos",
        "sec_vis": "Exploração",
        "nav_previsao": "Pre-Merger e Kick",
        "nav_validacao": "Validação Hold-Out",
        "nav_espectro": "Emulador e FFT",
        "nav_colisao3d": "Coalescência 3D",
        "nav_tabela": "Tabela de Métricas",
        "info_caption": "UFU · 3.435 simulações SXS",
        "pill_catalog": "Catálogo SXS v3.0.0 (3.435 fusões)",
        "pill_fid": "Overlap > 99.9%",
        "pages": {
            "previsao": {
                "titulo": "Previsão antecipada da colisão e diagnóstico de recuo relativístico",
                "subtitulo": "A rede observa apenas a espiral inicial (antes de t = -50M, com a fusão oculta) para reconstruir a forma de onda completa e estimar massa final, spin e velocidade de ejeção do buraco negro formado."
            },
            "validacao": {
                "titulo": "Validação cega (Hold-Out) em formas de onda inéditas",
                "subtitulo": "Avaliação em simulações de relatividade numérica nunca vistas no treinamento, comparando a parametrização espectral direta com a decomposição polar em amplitude e fase."
            },
            "espectro": {
                "titulo": "Emulador contínuo de parâmetros e espectro de Fourier",
                "subtitulo": "Varie continuamente a razão de massa (q) e o spin orbital para inspecionar a onda gravitacional gerada em milissegundos e os 32 modos espectrais filtrados pelo FNO."
            },
            "colisao3d": {
                "titulo": "Coalescência binária 3D sincronizada com o sinal gravitacional",
                "subtitulo": "Evolução tridimensional das órbitas relativísticas dos buracos negros sincronizada passo a passo com a emissão quadrupolar h_22(t) até o instante da fusão (t = 0)."
            },
            "tabela": {
                "titulo": "Síntese quantitativa dos experimentos no catálogo SXS",
                "subtitulo": "Comparativo consolidado de erro relativo L2, fidelidade de fase (Overlap) e Mismatch em todas as etapas da pesquisa."
            },
        },
        "lbl_evento": "Simulação binária monitorada",
        "lbl_massa": "Massa total inicial (massas solares)",
        "cfg_pre": "Parâmetros pré-impacto",
        "cfg_prec": "Precisão SXS-3435",
        "kpi1_title": "Fidelidade pré-merger",
        "kpi1_sub": "Região futura oculta (t > -50M)",
        "kpi2_title": "Massa do buraco negro final",
        "kpi2_sub": "M_sol irradiados em ondas",
        "kpi3_title": "Spin adimensional final",
        "kpi3_sub": "Referência SXS",
        "kpi4_title": "Velocidade de recuo (Kick)",
        "kpi4_sub": "SXS",
        "esc_tag_high": "Fuga galáctica (>300 km/s)",
        "esc_tag_mid": "Ejeção de galáxia anã (>150 km/s)",
        "esc_tag_low": "Ejeção de aglomerado (>50 km/s)",
        "p1_chart1_title": "Reconstrução futura da fusão a partir de t = -50M",
        "p1_tr_obs": "Espiral observada (entrada)",
        "p1_tr_real": "Fusão real SXS (oculta)",
        "p1_tr_pred": "Previsão CosmicFNO",
        "p1_x_time": "Tempo físico alinhado no pico da fusão (M)",
        "p1_chart2_title": "Recuo previsto vs velocidade de escape astrofísica",
        "p1_bars": ["Aglomerado", "Galáxia anã", "Recuo IA", "Gal. média", "Via Láctea"],
        "p1_y_vel": "Velocidade (km/s)",
        "p2_sim_lbl": "Simulação de teste cego",
        "p2_arq_lbl": "Parametrização neural",
        "p2_arq_opts": ["CosmicFNO espectral direto", "Polar-CosmicFNO (amplitude e fase)", "Ambas sobrepostas"],
        "p2_zoom_lbl": "Janela temporal (M)",
        "p2_k1_t": "Fidelidade de fase (Overlap)",
        "p2_k2_t": "Mismatch (1 - Overlap)",
        "p2_k2_s": "Limiar LIGO < 0.00100",
        "p2_k3_t": "Erro relativo L2",
        "p2_k4_t": "Conjunto de treinamento",
        "p2_k4_v": "11 simulações SXS",
        "p2_k4_s": "Malha q entre 1.0 e 8.0",
        "p2_sub1": "Teste cego: {id} (q = {q:.2f})",
        "p2_sub2": "Resíduo absoluto instantâneo |h_SXS(t) - h_FNO(t)|",
        "p2_ref": "Referência SXS",
        "p3_q": "Razão de massa (q = m1/m2)",
        "p3_spin": "Spin orbital alinhado (chi_1z)",
        "p3_c1": "Sinal gravitacional emulado no tempo (q={q:.1f}, chi_1z={s:+.2f})",
        "p3_c2": "Espectro de Fourier (32 modos ativos no operador)",
        "p3_xk": "Modo de frequência k",
        "p3_yf": "Amplitude espectral |FFT|",
        "p4_slider": "Razão de massa (q) da simulação SXS",
        "p4_sub1": "Trajetória orbital 3D (q = {q:.2f})",
        "p4_sub2": "Emissão sincronizada da onda Re(h_22)",
        "p4_play": "Iniciar simulação",
        "p4_pause": "Pausar",
        "p4_bh1": "Buraco negro primário (m1)",
        "p4_bh2": "Buraco negro secundário (m2)",
        "p5_k1_t": "Colisões SXS avaliadas",
        "p5_k1_s": "Teste cego: {n} simulações",
        "p5_k2_t": "Acurácia na massa final",
        "p5_k2_s": "Erro médio: {e:.2f}%",
        "p5_k3_t": "Variância explicada no spin (R²)",
        "p5_k4_t": "Classificação de ejeção (>50 km/s)",
        "p5_k4_s": "Aglomerados globulares",
        "footer_l": "<b>CosmicFNO</b> — Observatório Espectral de Ondas Gravitacionais",
        "footer_r": "Universidade Federal de Uberlândia (UFU) · Catálogo SXS v3.0.0",
    },
    "EN": {
        "sec_main": "Modules",
        "sec_vis": "Exploration",
        "nav_previsao": "Pre-Merger & Kick",
        "nav_validacao": "Hold-Out Validation",
        "nav_espectro": "Emulator & FFT",
        "nav_colisao3d": "3D Coalescence",
        "nav_tabela": "Metrics Table",
        "info_caption": "UFU · 3,435 SXS simulations",
        "pill_catalog": "SXS Catalog v3.0.0 (3,435 mergers)",
        "pill_fid": "Overlap > 99.9%",
        "pages": {
            "previsao": {
                "titulo": "Pre-merger collision forecasting and relativistic recoil diagnostics",
                "subtitulo": "The neural operator observes only the early inspiral (before t = -50M, with the merger hidden) to reconstruct the full waveform and estimate the final mass, spin, and recoil kick of the remnant black hole."
            },
            "validacao": {
                "titulo": "Blind hold-out validation on unseen numerical relativity waveforms",
                "subtitulo": "Evaluation on SXS simulations held out during training, comparing direct spectral parameterization against explicit polar amplitude-phase decomposition."
            },
            "espectro": {
                "titulo": "Continuous parameter-space emulator and Fourier spectrum",
                "subtitulo": "Vary mass ratio (q) and orbital spin continuously to inspect millisecond waveform synthesis and the 32 spectral modes retained by the neural operator."
            },
            "colisao3d": {
                "titulo": "3D binary coalescence synchronized with gravitational strain",
                "subtitulo": "Three-dimensional relativistic orbital evolution synchronized step-by-step with the dominant quadrupole wave emission h_22(t) through merger (t = 0)."
            },
            "tabela": {
                "titulo": "Quantitative experimental summary across the SXS catalog",
                "subtitulo": "Consolidated benchmark of relative L2 error, phase overlap, and mismatch across all research stages."
            },
        },
        "lbl_evento": "Monitored binary simulation",
        "lbl_massa": "Initial total mass (solar masses)",
        "cfg_pre": "Pre-impact parameters",
        "cfg_prec": "SXS-3435 accuracy",
        "kpi1_title": "Pre-merger fidelity",
        "kpi1_sub": "Hidden future window (t > -50M)",
        "kpi2_title": "Remnant black hole mass",
        "kpi2_sub": "M_sol radiated in waves",
        "kpi3_title": "Predicted final spin",
        "kpi3_sub": "SXS reference",
        "kpi4_title": "Relativistic recoil kick",
        "kpi4_sub": "SXS",
        "esc_tag_high": "Galactic escape (>300 km/s)",
        "esc_tag_mid": "Dwarf galaxy ejection (>150 km/s)",
        "esc_tag_low": "Cluster ejection (>50 km/s)",
        "p1_chart1_title": "Future merger waveform reconstruction from t = -50M",
        "p1_tr_obs": "Observed inspiral (input)",
        "p1_tr_real": "True SXS merger (hidden)",
        "p1_tr_pred": "CosmicFNO forecast",
        "p1_x_time": "Merger-aligned physical time (M)",
        "p1_chart2_title": "Predicted kick vs astrophysical escape velocity",
        "p1_bars": ["Glob. cluster", "Dwarf galaxy", "AI kick", "Medium gal.", "Milky Way"],
        "p1_y_vel": "Velocity (km/s)",
        "p2_sim_lbl": "Blind hold-out test simulation",
        "p2_arq_lbl": "Neural parameterization",
        "p2_arq_opts": ["Direct spectral CosmicFNO", "Polar-CosmicFNO (amplitude & phase)", "Both overlaid"],
        "p2_zoom_lbl": "Physical time window (M)",
        "p2_k1_t": "Phase fidelity (Overlap)",
        "p2_k2_t": "Mismatch (1 - Overlap)",
        "p2_k2_s": "LIGO threshold < 0.00100",
        "p2_k3_t": "Relative L2 error",
        "p2_k4_t": "Training set",
        "p2_k4_v": "11 SXS simulations",
        "p2_k4_s": "Grid q from 1.0 to 8.0",
        "p2_sub1": "Blind test: {id} (q = {q:.2f})",
        "p2_sub2": "Instantaneous absolute residual |h_SXS(t) - h_FNO(t)|",
        "p2_ref": "SXS reference",
        "p3_q": "Mass ratio (q = m1/m2)",
        "p3_spin": "Aligned orbital spin (chi_1z)",
        "p3_c1": "Time-domain emulated strain (q={q:.1f}, chi_1z={s:+.2f})",
        "p3_c2": "Fourier spectrum (32 active operator modes)",
        "p3_xk": "Frequency mode k",
        "p3_yf": "Spectral amplitude |FFT|",
        "p4_slider": "SXS simulation mass ratio (q)",
        "p4_sub1": "3D orbital trajectory (q = {q:.2f})",
        "p4_sub2": "Synchronized wave emission Re(h_22)",
        "p4_play": "Start simulation",
        "p4_pause": "Pause",
        "p4_bh1": "Primary black hole (m1)",
        "p4_bh2": "Secondary black hole (m2)",
        "p5_k1_t": "Evaluated SXS mergers",
        "p5_k1_s": "Blind test: {n} simulations",
        "p5_k2_t": "Final mass accuracy",
        "p5_k2_s": "Mean error: {e:.2f}%",
        "p5_k3_t": "Final spin explained variance (R²)",
        "p5_k4_t": "Ejection classification (>50 km/s)",
        "p5_k4_s": "Globular clusters",
        "footer_l": "<b>CosmicFNO</b> — Spectral Gravitational-Wave Observatory",
        "footer_r": "Federal University of Uberlândia (UFU) · SXS Catalog v3.0.0",
    },
    "ES": {
        "sec_main": "Módulos",
        "sec_vis": "Exploración",
        "nav_previsao": "Pre-Merger y Kick",
        "nav_validacao": "Validación Hold-Out",
        "nav_espectro": "Emulador y FFT",
        "nav_colisao3d": "Coalescencia 3D",
        "nav_tabela": "Tabla de Métricas",
        "info_caption": "UFU · 3.435 simulaciones SXS",
        "pill_catalog": "Catálogo SXS v3.0.0 (3.435 fusiones)",
        "pill_fid": "Overlap > 99.9%",
        "pages": {
            "previsao": {
                "titulo": "Predicción anticipada de la colisión y diagnóstico de retroceso relativista",
                "subtitulo": "El operador neuronal observa solo la espiral inicial (antes de t = -50M, con la fusión oculta) para reconstruir la forma de onda completa y estimar masa final, espín y velocidad de eyección del agujero negro."
            },
            "validacao": {
                "titulo": "Validación ciega (Hold-Out) en formas de onda inéditas",
                "subtitulo": "Evaluación en simulaciones de relatividad numérica nunca vistas en el entrenamiento, comparando la parametrización espectral directa con la descomposición polar en amplitud y fase."
            },
            "espectro": {
                "titulo": "Emulador continuo de parámetros y espectro de Fourier",
                "subtitulo": "Varíe continuamente la razón de masa (q) y el espín orbital para inspeccionar la onda gravitacional generada en milisegundos y los 32 modos espectrales filtrados por el FNO."
            },
            "colisao3d": {
                "titulo": "Coalescencia binaria 3D sincronizada con la señal gravitacional",
                "subtitulo": "Evolución tridimensional de las órbitas relativistas de los agujeros negros sincronizada paso a paso con la emisión cuadrupolar h_22(t) hasta el instante de la fusión (t = 0)."
            },
            "tabela": {
                "titulo": "Síntesis cuantitativa de los experimentos en el catálogo SXS",
                "subtitulo": "Comparativa consolidada de error relativo L2, fidelidad de fase (Overlap) y Mismatch en todas las etapas de la investigación."
            },
        },
        "lbl_evento": "Simulación binaria monitoreada",
        "lbl_massa": "Masa total inicial (masas solares)",
        "cfg_pre": "Parámetros pre-impacto",
        "cfg_prec": "Precisión SXS-3435",
        "kpi1_title": "Fidelidad pre-merger",
        "kpi1_sub": "Región futura oculta (t > -50M)",
        "kpi2_title": "Masa del agujero negro final",
        "kpi2_sub": "M_sol irradiados en ondas",
        "kpi3_title": "Espín adimensional final",
        "kpi3_sub": "Referencia SXS",
        "kpi4_title": "Velocidad de retroceso (Kick)",
        "kpi4_sub": "SXS",
        "esc_tag_high": "Escape galáctico (>300 km/s)",
        "esc_tag_mid": "Eyección galaxia enana (>150 km/s)",
        "esc_tag_low": "Eyección de cúmulo (>50 km/s)",
        "p1_chart1_title": "Reconstrucción futura de la fusión a partir de t = -50M",
        "p1_tr_obs": "Espiral observada (entrada)",
        "p1_tr_real": "Fusión real SXS (oculta)",
        "p1_tr_pred": "Predicción CosmicFNO",
        "p1_x_time": "Tiempo físico alineado en el pico de la fusión (M)",
        "p1_chart2_title": "Retroceso previsto vs velocidad de escape astrofísica",
        "p1_bars": ["Cúmulo glob.", "Gal. enana", "Retroceso IA", "Gal. media", "Vía Láctea"],
        "p1_y_vel": "Velocidad (km/s)",
        "p2_sim_lbl": "Simulación de prueba ciega",
        "p2_arq_lbl": "Parametrización neuronal",
        "p2_arq_opts": ["CosmicFNO espectral directo", "Polar-CosmicFNO (amplitud y fase)", "Ambas superpuestas"],
        "p2_zoom_lbl": "Ventana temporal (M)",
        "p2_k1_t": "Fidelidad de fase (Overlap)",
        "p2_k2_t": "Mismatch (1 - Overlap)",
        "p2_k2_s": "Umbral LIGO < 0.00100",
        "p2_k3_t": "Error relativo L2",
        "p2_k4_t": "Conjunto de entrenamiento",
        "p2_k4_v": "11 simulaciones SXS",
        "p2_k4_s": "Malla q entre 1.0 y 8.0",
        "p2_sub1": "Prueba ciega: {id} (q = {q:.2f})",
        "p2_sub2": "Residuo absoluto instantáneo |h_SXS(t) - h_FNO(t)|",
        "p2_ref": "Referencia SXS",
        "p3_q": "Razón de masa (q = m1/m2)",
        "p3_spin": "Espín orbital alineado (chi_1z)",
        "p3_c1": "Señal gravitacional emulada en el tiempo (q={q:.1f}, chi_1z={s:+.2f})",
        "p3_c2": "Espectro de Fourier (32 modos activos en el operador)",
        "p3_xk": "Modo de frecuencia k",
        "p3_yf": "Amplitud espectral |FFT|",
        "p4_slider": "Razón de masa (q) de la simulación SXS",
        "p4_sub1": "Trayectoria orbital 3D (q = {q:.2f})",
        "p4_sub2": "Emisión sincronizada de la onda Re(h_22)",
        "p4_play": "Iniciar simulación",
        "p4_pause": "Pausar",
        "p4_bh1": "Agujero negro primario (m1)",
        "p4_bh2": "Agujero negro secundario (m2)",
        "p5_k1_t": "Colisiones SXS evaluadas",
        "p5_k1_s": "Prueba ciega: {n} simulaciones",
        "p5_k2_t": "Precisión en la masa final",
        "p5_k2_s": "Error medio: {e:.2f}%",
        "p5_k3_t": "Varianza explicada en espín (R²)",
        "p5_k4_t": "Clasificación de eyección (>50 km/s)",
        "p5_k4_s": "Cúmulos globulares",
        "footer_l": "<b>CosmicFNO</b> — Observatorio Espectral de Ondas Gravitacionales",
        "footer_r": "Universidad Federal de Uberlândia (UFU) · Catálogo SXS v3.0.0",
    },
}


# =====================================================================
# 2. DESIGN SYSTEM (UI/UX PRO MAX + FRONTEND-DESIGN + ANTIGRAVITY)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Fira+Code:wght@500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    html, body, .stApp, p, label, h1, h2, h3, h4, h5, h6, button, input, select, textarea, table, th, td {
        font-family: 'Montserrat', sans-serif !important;
    }

    /* Preserva os icones Material Symbols (botao de recolher sidebar e icones de menu) */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="collapsedControl"] span,
    [data-testid="stExpandSidebarButton"] span,
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 20px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.0rem !important;
    }
    .block-container {
        padding-top: 0.25rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        padding-bottom: 2.0rem !important;
        max-width: 100% !important;
    }

    .stApp {
        background-color: #f7f6fb !important;
        color: #160e26 !important;
    }

    section.main [data-testid="stWidgetLabel"] p,
    section.main [data-testid="stWidgetLabel"] label,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] label {
        color: #24183b !important;
        font-weight: 600 !important;
        font-size: 0.81rem !important;
    }

    /* =================================================================
       SIDEBAR FIXA (NAO PODE SER REDIMENSIONADA ARRASTANDO) + ICON RAIL
       ================================================================= */
    div[data-testid="stSidebarResizeHandle"] {
        display: none !important;
        width: 0 !important;
        pointer-events: none !important;
        cursor: default !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #11091f !important;
        border-right: 1px solid #211339 !important;
    }
    @media (min-width: 769px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 242px !important;
            min-width: 242px !important;
            max-width: 242px !important;
        }
        /* Quando recolhida no Desktop, mantem trilho de icones visivel (64px) */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 64px !important;
            min-width: 64px !important;
            max-width: 64px !important;
            transform: none !important;
            margin-left: 0 !important;
            visibility: visible !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
            padding-left: 6px !important;
            padding-right: 6px !important;
            overflow-x: hidden !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-logo-free,
        section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-section-label,
        section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-tiny-research,
        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stHorizontalBlock"],
        section[data-testid="stSidebar"][aria-expanded="false"] div.stButton > button [data-testid="stMarkdownContainer"] {
            display: none !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] div.stButton > button {
            justify-content: center !important;
            padding: 10px 0 !important;
            width: 46px !important;
            margin: 4px auto !important;
        }
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0.25rem !important;
    }

    /* Botoes pequenos e discretos de idioma no topo da sidebar */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 4px !important;
        margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div.stButton > button {
        padding: 2px 5px !important;
        min-height: 22px !important;
        height: 22px !important;
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        border-radius: 5px !important;
        justify-content: center !important;
        text-align: center !important;
        margin-bottom: 0 !important;
    }

    /* Logo Oficial SOLTA na Sidebar */
    .sidebar-logo-free {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 2px 12px 2px;
        margin: 0 0 2px 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sidebar-logo-free img {
        width: 100%;
        max-width: 188px;
        height: auto;
        display: block;
        filter: brightness(1.38) contrast(1.06) drop-shadow(0 4px 14px rgba(168, 85, 247, 0.22));
    }

    /* Labels de secao em sentence-case (sem ALL-CAPS cliche de IA - Anthropic frontend-design) */
    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #7c6b99 !important;
        margin: 12px 0 4px 8px;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: center !important;
        gap: 9px !important;
        padding: 8px 12px !important;
        border-radius: 9px !important;
        border: none !important;
        background-color: transparent !important;
        color: #b5a6cc !important;
        font-weight: 500 !important;
        font-size: 0.80rem !important;
        cursor: pointer !important;
        transition: background-color 0.16s ease, color 0.16s ease;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: rgba(124, 58, 237, 0.15) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #6d28d9 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 16px -4px rgba(109, 40, 217, 0.45) !important;
    }
    section[data-testid="stSidebar"] div.stButton > button p {
        font-size: 0.80rem !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }

    .sidebar-tiny-research {
        font-size: 0.62rem !important;
        line-height: 1.3 !important;
        color: #63547d !important;
        margin-top: 16px !important;
        padding: 8px 8px 0 8px !important;
        border-top: 1px solid rgba(139, 122, 168, 0.12) !important;
    }

    /* =================================================================
       CABECALHO EDITORIAL 100% LARGURA (ANTHROPIC FRONTEND-DESIGN SKILL)
       Sem caixa redundante ou ALL-CAPS eyebrow cliche; ocupa toda a largura util
       ================================================================= */
    .editorial-header {
        width: 100%;
        padding: 6px 2px 14px 2px;
        margin-bottom: 14px;
        border-bottom: 1px solid #e6e1f0;
    }
    .editorial-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 4px;
        width: 100%;
    }
    .editorial-title {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #160e26 !important;
        margin: 0 !important;
        line-height: 1.3 !important;
        letter-spacing: -0.015em;
    }
    .editorial-badges {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .meta-tag {
        font-size: 0.71rem;
        font-weight: 600;
        color: #5b21b6 !important;
        background-color: #ede9fe;
        padding: 3px 10px;
        border-radius: 6px;
        white-space: nowrap;
    }
    .meta-tag-solid {
        font-size: 0.71rem;
        font-weight: 600;
        color: #ffffff !important;
        background-color: #2e1065;
        padding: 3px 10px;
        border-radius: 6px;
        white-space: nowrap;
    }
    .editorial-subtitle {
        font-size: 0.79rem !important;
        color: #554a6b !important;
        margin: 0 !important;
        line-height: 1.5 !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    /* =================================================================
       ANTIGRAVITY WEIGHTLESS FLOATING CARDS + UI/UX PRO MAX TABULAR NUMS
       ================================================================= */
    @keyframes floatIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 20px 40px -15px rgba(22, 14, 38, 0.06), 0 4px 12px -2px rgba(109, 40, 217, 0.03);
        animation: floatIn 0.28s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .kpi-card:nth-child(2) { animation-delay: 0.04s; }
    .kpi-card:nth-child(3) { animation-delay: 0.08s; }
    .kpi-card:nth-child(4) { animation-delay: 0.12s; }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #5e5275 !important;
        margin: 0 0 6px 0;
    }
    .kpi-value {
        font-family: 'Montserrat', sans-serif !important;
        font-variant-numeric: tabular-nums;
        font-size: 1.48rem;
        font-weight: 800;
        color: #160e26 !important;
        margin: 0 0 6px 0;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.72rem;
        font-weight: 500;
        color: #6d28d9 !important;
        margin: 0;
    }

    .white-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 14px;
        padding: 13px 16px;
        box-shadow: 0 20px 40px -15px rgba(22, 14, 38, 0.06), 0 4px 12px -2px rgba(109, 40, 217, 0.03);
        margin-bottom: 10px;
    }
    .card-title {
        font-size: 0.81rem;
        font-weight: 700;
        color: #160e26 !important;
        margin-bottom: 3px;
    }
    .card-sub {
        font-size: 0.77rem;
        font-variant-numeric: tabular-nums;
        color: #554a6b !important;
        margin-bottom: 0;
        line-height: 1.45;
    }

    /* Superficie flutuante dos graficos Plotly */
    div[data-testid="stPlotlyChart"] {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 6px;
        box-shadow: 0 20px 40px -15px rgba(22, 14, 38, 0.06), 0 4px 12px -2px rgba(109, 40, 217, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.9);
    }

    /* Tabela Moderna Flutuante */
    .table-card {
        background-color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 20px 40px -15px rgba(22, 14, 38, 0.06), 0 4px 12px -2px rgba(109, 40, 217, 0.03);
        overflow-x: auto;
    }
    .modern-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        font-variant-numeric: tabular-nums;
        color: #160e26 !important;
    }
    .modern-table th {
        text-align: left;
        padding: 11px 14px;
        font-size: 0.74rem;
        font-weight: 700;
        color: #554a6b !important;
        border-bottom: 1px solid #e6e1f0;
        background-color: #faf9fc;
    }
    .modern-table td {
        padding: 12px 14px;
        border-bottom: 1px solid #f2eff8;
        color: #1e1333 !important;
        font-weight: 500;
    }
    .modern-table tr:last-child td {
        border-bottom: none;
    }
    .modern-table tr:hover td {
        background-color: #faf8ff;
    }
    .tbl-pill {
        display: inline-block;
        background-color: #ede9fe;
        color: #5b21b6 !important;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 0.76rem;
    }

    .site-footer {
        margin-top: 22px;
        padding: 12px 4px;
        border-top: 1px solid #e6e1f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 0.72rem;
        color: #64587a !important;
    }

    @media (prefers-reduced-motion: reduce) {
        .kpi-card {
            animation: none !important;
            transition: none !important;
        }
    }

    @media (max-width: 992px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 600px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-top: 0.25rem !important;
        }
        .kpi-grid {
            grid-template-columns: 1fr;
            gap: 10px;
        }
        .editorial-title {
            font-size: 0.95rem !important;
        }
        .editorial-subtitle {
            font-size: 0.75rem !important;
        }
        .kpi-value {
            font-size: 1.30rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def render_kpi_grid(cards):
    """Renderiza 4 cartoes flutuantes com tipografia editorial e tabular-nums (sem cliches de IA)."""
    blocos = []
    for c in cards:
        blocos.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{c["label"]}</div>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<p class="kpi-sub">{c["sub"]}</p>'
            f'</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(blocos)}</div>', unsafe_allow_html=True)


# =====================================================================
# 3. CARREGAMENTO DOS MODELOS E TENSORES SXS
# =====================================================================
@st.cache_resource
def carregar_ambiente_cientifico():
    sys.path.append(PASTA_SRC)

    fno_spec = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(PASTA_SRC, "02_modelo_fno.py"))
    fno_mod = importlib.util.module_from_spec(fno_spec)
    fno_spec.loader.exec_module(fno_mod)

    met_spec = importlib.util.spec_from_file_location("08_metricas_fisicas", os.path.join(PASTA_SRC, "08_metricas_fisicas.py"))
    met_mod = importlib.util.module_from_spec(met_spec)
    met_spec.loader.exec_module(met_mod)

    rem_spec = importlib.util.spec_from_file_location("10_previsao_pre_merger_e_remanescente", os.path.join(PASTA_SRC, "10_previsao_pre_merger_e_remanescente.py"))
    rem_mod = importlib.util.module_from_spec(rem_spec)
    rem_spec.loader.exec_module(rem_mod)

    device = torch.device("cpu")
    pasta_sxs = os.path.join(PASTA_BASE, "data", "raw", "dataset_sxs")
    pasta_modelos = os.path.join(PASTA_BASE, "modelos_salvos")

    t_fase = torch.load(os.path.join(pasta_sxs, "dataset_t_fase_alinhada.pt"), map_location=device)
    h_fase = torch.load(os.path.join(pasta_sxs, "dataset_h22_fase_alinhada.pt"), map_location=device)
    p_fase = torch.load(os.path.join(pasta_sxs, "dataset_params_fase_alinhada.pt"), map_location=device)
    rem_sxs = torch.load(os.path.join(pasta_sxs, "dataset_remanescente_sxs.pt"), map_location=device)

    bench_polar = torch.load(os.path.join(pasta_modelos, "polar_fno_benchmark.pt"), map_location=device, weights_only=False)
    bench_3300 = torch.load(os.path.join(pasta_modelos, "sxs_3300_remnant_model.pt"), map_location=device, weights_only=False)

    modelos = {}
    for chave, arquivo in [
        ("completo", "fno_fase_alinhada.pth"),
        ("pre_merger", "fno_pre_merger_forecaster.pth"),
    ]:
        caminho = os.path.join(pasta_modelos, arquivo)
        if os.path.exists(caminho):
            m = fno_mod.CosmicFNO(num_modos=32, largura=64).to(device)
            m.load_state_dict(torch.load(caminho, map_location=device, weights_only=True))
            m.eval()
            modelos[chave] = m

    rem_net = rem_mod.RemnantPredictorSpectral(num_modos_fft=64).to(device)
    caminho_rem = os.path.join(pasta_modelos, "remnant_predictor.pth")
    if os.path.exists(caminho_rem):
        rem_net.load_state_dict(torch.load(caminho_rem, map_location=device, weights_only=True))
        rem_net.eval()

    return {
        "modelos": modelos,
        "rem_net": rem_net,
        "t_fase": t_fase,
        "h_fase": h_fase,
        "p_fase": p_fase,
        "rem_sxs": rem_sxs,
        "bench_polar": bench_polar,
        "bench_3300": bench_3300,
        "metricas": met_mod,
    }


env = carregar_ambiente_cientifico()
met = env["metricas"]

CATALOGO_INFO = [
    {"idx": 1, "id": "SXS:BBH:0115", "q": 1.07, "spin": -0.20, "tipo": "Kick 354 km/s"},
    {"idx": 4, "id": "SXS:BBH:0167", "q": 4.00, "spin": 0.00, "tipo": "q = 4.00"},
    {"idx": 3, "id": "SXS:BBH:0113", "q": 5.00, "spin": 0.00, "tipo": "q = 5.00"},
    {"idx": 0, "id": "SXS:BBH:0305", "q": 1.22, "spin": 0.33, "tipo": "GW150914 Ref"},
    {"idx": 2, "id": "SXS:BBH:0114", "q": 8.00, "spin": -0.50, "tipo": "q = 8.00"},
]

# =====================================================================
# 4. MENU LATERAL: IDIOMA COMPACTO NO TOPO + LOGO + ICONES A ESQUERDA
# =====================================================================
if "pagina_ativa" not in st.session_state:
    st.session_state["pagina_ativa"] = "previsao"
if "idioma" not in st.session_state:
    st.session_state["idioma"] = "PT"

tr = I18N[st.session_state["idioma"]]

with st.sidebar:
    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        if st.button("PT", key="btn_pt", type="primary" if st.session_state["idioma"] == "PT" else "secondary"):
            st.session_state["idioma"] = "PT"
            st.rerun()
    with l_col2:
        if st.button("EN", key="btn_en", type="primary" if st.session_state["idioma"] == "EN" else "secondary"):
            st.session_state["idioma"] = "EN"
            st.rerun()
    with l_col3:
        if st.button("ES", key="btn_es", type="primary" if st.session_state["idioma"] == "ES" else "secondary"):
            st.session_state["idioma"] = "ES"
            st.rerun()

    if logo_b64:
        st.markdown(
            f'<div class="sidebar-logo-free"><img src="data:image/png;base64,{logo_b64}" alt="CosmicFNO Logo"></div>',
            unsafe_allow_html=True
        )

    st.markdown(f'<div class="sidebar-section-label">{tr["sec_main"]}</div>', unsafe_allow_html=True)

    if st.button(
        tr["nav_previsao"],
        icon=":material/radar:",
        type="primary" if st.session_state["pagina_ativa"] == "previsao" else "secondary"
    ):
        st.session_state["pagina_ativa"] = "previsao"
        st.rerun()

    if st.button(
        tr["nav_validacao"],
        icon=":material/verified:",
        type="primary" if st.session_state["pagina_ativa"] == "validacao" else "secondary"
    ):
        st.session_state["pagina_ativa"] = "validacao"
        st.rerun()

    st.markdown(f'<div class="sidebar-section-label">{tr["sec_vis"]}</div>', unsafe_allow_html=True)

    if st.button(
        tr["nav_espectro"],
        icon=":material/graphic_eq:",
        type="primary" if st.session_state["pagina_ativa"] == "espectro" else "secondary"
    ):
        st.session_state["pagina_ativa"] = "espectro"
        st.rerun()

    if st.button(
        tr["nav_colisao3d"],
        icon=":material/3d_rotation:",
        type="primary" if st.session_state["pagina_ativa"] == "colisao3d" else "secondary"
    ):
        st.session_state["pagina_ativa"] = "colisao3d"
        st.rerun()

    if st.button(
        tr["nav_tabela"],
        icon=":material/table_chart:",
        type="primary" if st.session_state["pagina_ativa"] == "tabela" else "secondary"
    ):
        st.session_state["pagina_ativa"] = "tabela"
        st.rerun()

    st.markdown(f'<div class="sidebar-tiny-research">{tr["info_caption"]}</div>', unsafe_allow_html=True)


# =====================================================================
# 5. CABECALHO EDITORIAL 100% LARGURA (SEM CAIXA REDUNDANTE)
# =====================================================================
meta_atual = tr["pages"][st.session_state["pagina_ativa"]]

st.markdown(f"""
<div class="editorial-header">
    <div class="editorial-title-row">
        <h1 class="editorial-title">{meta_atual['titulo']}</h1>
        <div class="editorial-badges">
            <span class="meta-tag">{tr['pill_catalog']}</span>
            <span class="meta-tag-solid">{tr['pill_fid']}</span>
        </div>
    </div>
    <p class="editorial-subtitle">{meta_atual['subtitulo']}</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGINA 1: PREVISAO ANTECIPADA DA COLISAO (PRE-MERGER) E BURACO NEGRO FUGITIVO
# =====================================================================
if st.session_state["pagina_ativa"] == "previsao":
    b3300 = env["bench_3300"]

    with st.container():
        f_col1, f_col2, f_col3 = st.columns([1.5, 1.2, 1.3])
        with f_col1:
            opcoes_pred = [f"{item['id']} (q = {item['q']:.2f}, {item['tipo']})" for item in CATALOGO_INFO]
            escolha_pred = st.selectbox(tr["lbl_evento"], opcoes_pred, index=0)
            info_p = CATALOGO_INFO[opcoes_pred.index(escolha_pred)]

        with f_col2:
            massa_total_sol = st.slider(
                tr["lbl_massa"],
                min_value=20.0,
                max_value=150.0,
                value=65.0,
                step=5.0
            )

        q_val = info_p["q"]
        m1_sol = massa_total_sol * (q_val / (1.0 + q_val))
        m2_sol = massa_total_sol * (1.0 / (1.0 + q_val))

        with f_col3:
            st.markdown(f"""
            <div class="white-card" style="margin-bottom: 0;">
                <div class="card-title">{tr['cfg_pre']}</div>
                <div class="card-sub">
                    <b>m1:</b> {m1_sol:.1f} M_sol &nbsp;|&nbsp; <b>m2:</b> {m2_sol:.1f} M_sol &nbsp;|&nbsp; <b>{tr['cfg_prec']}:</b> {100.0 - b3300['erro_rel_massa']:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Inferencia Neural
    idx_t = info_p["idx"]
    t_vec = env["t_fase"][idx_t]
    h_real_full = env["h_fase"][idx_t]
    params_v = env["p_fase"][idx_t]
    rem_real = env["rem_sxs"][idx_t].numpy()

    t_corte = -50.0
    mascara_insp = (t_vec <= t_corte).float()
    onda_observada = h_real_full * mascara_insp
    n_pts = len(t_vec)
    q_c = params_v[0].repeat(n_pts)
    t_norm = (t_vec - t_vec.min()) / (t_vec.max() - t_vec.min())

    X_pre = torch.stack([onda_observada, q_c, t_norm], dim=-1).unsqueeze(0)

    with torch.no_grad():
        onda_prevista_fno = env["modelos"]["pre_merger"](X_pre).squeeze(0).squeeze(-1)
        pred_rem = env["rem_net"](onda_prevista_fno.unsqueeze(0), params_v[0:1].unsqueeze(0)).squeeze().numpy()

    m_final_frac = float(pred_rem[0])
    spin_final_pred = float(pred_rem[1])
    v_kick_pred_kms = float(abs(pred_rem[2]) * 1000.0)

    m_final_sol = m_final_frac * massa_total_sol
    energia_sol = (1.0 - m_final_frac) * massa_total_sol
    energia_pct = (1.0 - m_final_frac) * 100.0

    t_np = t_vec.numpy()
    h_real_np = h_real_full.numpy()
    h_pred_np = onda_prevista_fno.numpy()

    mask_futuro = t_np > t_corte
    ov_futuro, _ = met.calcular_overlap_e_mismatch(h_real_np[mask_futuro], h_pred_np[mask_futuro])

    if v_kick_pred_kms >= 300.0:
        tag_fuga = tr["esc_tag_high"]
    elif v_kick_pred_kms >= 130.0:
        tag_fuga = tr["esc_tag_mid"]
    else:
        tag_fuga = tr["esc_tag_low"]

    render_kpi_grid([
        {
            "label": tr["kpi1_title"],
            "value": f"{ov_futuro*100:.2f}%",
            "sub": tr["kpi1_sub"],
        },
        {
            "label": tr["kpi2_title"],
            "value": f"{m_final_sol:.1f} M_sol",
            "sub": f"{energia_sol:.2f} {tr['kpi2_sub']} ({energia_pct:.1f}%)",
        },
        {
            "label": tr["kpi3_title"],
            "value": f"{spin_final_pred:.3f} c",
            "sub": f"{tr['kpi3_sub']}: {rem_real[1]:.3f} c",
        },
        {
            "label": tr["kpi4_title"],
            "value": f"{v_kick_pred_kms:.1f} km/s",
            "sub": f"{tag_fuga} ({tr['kpi4_sub']}: {rem_real[2]:.0f} km/s)",
        },
    ])

    g_col1, g_col2 = st.columns([1.65, 1.0], gap="medium")

    with g_col1:
        mask_obs = (t_np >= -220.0) & (t_np <= t_corte)
        mask_fut_plot = (t_np >= t_corte) & (t_np <= 80.0)

        fig_onda = go.Figure()
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_obs], y=h_real_np[mask_obs],
                mode="lines",
                name=tr["p1_tr_obs"],
                line=dict(color="#94a3b8", width=2.3)
            )
        )
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_real_np[mask_fut_plot],
                mode="lines",
                name=tr["p1_tr_real"],
                line=dict(color="#190e30", width=2.9)
            )
        )
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_pred_np[mask_fut_plot],
                mode="lines",
                name=tr["p1_tr_pred"],
                line=dict(color="#7c3aed", width=2.5, dash="dash")
            )
        )
        fig_onda.add_vrect(
            x0=t_corte, x1=80.0,
            fillcolor="rgba(124, 58, 237, 0.06)",
            layer="below", line_width=0
        )
        fig_onda.add_vline(x=t_corte, line_width=1.6, line_dash="dash", line_color="#6d28d9")
        fig_onda.update_layout(
            title=dict(text=tr["p1_chart1_title"], font=dict(color="#160e26", size=13)),
            height=395,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#faf9fc",
            font=dict(family="Montserrat, sans-serif", color="#160e26", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#160e26")),
            margin=dict(l=45, r=20, t=58, b=38),
            xaxis=dict(title=dict(text=tr["p1_x_time"], font=dict(color="#160e26")), tickfont=dict(color="#554a6b"), gridcolor="#ede9f5"),
            yaxis=dict(title=dict(text="Strain Re(h_22)", font=dict(color="#160e26")), tickfont=dict(color="#554a6b"), gridcolor="#ede9f5")
        )
        st.plotly_chart(fig_onda, use_container_width=True, theme=None)

    with g_col2:
        categorias_esc = tr["p1_bars"]
        valores_esc = [50.0, 150.0, v_kick_pred_kms, 300.0, 550.0]
        cores_esc = ["#dcd6e8", "#b5a6cc", "#7c3aed", "#64587a", "#190e30"]

        fig_bar = go.Figure(
            go.Bar(
                x=categorias_esc,
                y=valores_esc,
                marker_color=cores_esc,
                text=[f"{v:.0f} km/s" for v in valores_esc],
                textposition="outside",
                textfont=dict(color="#160e26", size=11)
            )
        )
        fig_bar.update_layout(
            title=dict(text=tr["p1_chart2_title"], font=dict(color="#160e26", size=13)),
            height=395,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#faf9fc",
            font=dict(family="Montserrat, sans-serif", color="#160e26", size=11),
            showlegend=False,
            margin=dict(l=40, r=20, t=58, b=38),
            xaxis=dict(tickfont=dict(color="#554a6b"), gridcolor="#ede9f5"),
            yaxis=dict(title=dict(text=tr["p1_y_vel"], font=dict(color="#160e26")), tickfont=dict(color="#554a6b"), range=[0, 650], gridcolor="#ede9f5")
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)


# =====================================================================
# PAGINA 2: VALIDACAO CIENTIFICA HOLD-OUT (BENCHMARK 14 ONDAS SXS — 99.98% OVERLAP)
# =====================================================================
elif st.session_state["pagina_ativa"] == "validacao":
    res_lista = env["bench_polar"]["resultados_teste"]

    v_col1, v_col2, v_col3 = st.columns([1.3, 1.3, 1.4])
    with v_col1:
        opcoes_bench = [f"{r['id']} (q = {r['q']:.2f}, Overlap: {r['ov_dir']*100:.2f}%)" for r in res_lista]
        escolha_b = st.selectbox(tr["p2_sim_lbl"], opcoes_bench, index=0)
        r_sel = res_lista[opcoes_bench.index(escolha_b)]

    with v_col2:
        opts_arq = tr["p2_arq_opts"]
        arq_comparada = st.selectbox(tr["p2_arq_lbl"], opts_arq, index=0)
        idx_arq = opts_arq.index(arq_comparada)

    with v_col3:
        janela_zoom = st.slider(
            tr["p2_zoom_lbl"],
            min_value=-330.0,
            max_value=90.0,
            value=(-280.0, 80.0),
            step=5.0
        )

    t_np = np.linspace(-330.0, 90.0, 4096)
    ref_onda = r_sel["ref"]
    pred_ativa = r_sel["pred_dir"] if idx_arq != 1 else r_sel["pred_polar"]

    ov_val = r_sel["ov_dir"] if idx_arq != 1 else r_sel["ov_pol"]
    mis_val = r_sel["mis_dir"] if idx_arq != 1 else r_sel["mis_pol"]
    l2_val = r_sel["l2_dir"] if idx_arq != 1 else r_sel["l2_pol"]
    mse_val = met.calcular_mse(ref_onda, pred_ativa)

    render_kpi_grid([
        {"label": tr["p2_k1_t"], "value": f"{ov_val*100:.2f}%", "sub": f"Mismatch: {mis_val:.5f}"},
        {"label": tr["p2_k2_t"], "value": f"{mis_val:.5f}", "sub": tr["p2_k2_s"]},
        {"label": tr["p2_k3_t"], "value": f"{l2_val*100:.2f}%", "sub": f"MSE: {mse_val:.6f}"},
        {"label": tr["p2_k4_t"], "value": tr["p2_k4_v"], "sub": tr["p2_k4_s"]},
    ])

    mask = (t_np >= janela_zoom[0]) & (t_np <= janela_zoom[1])
    t_zoom = t_np[mask]
    ref_zoom = ref_onda[mask]
    erro_zoom = np.abs(ref_zoom - pred_ativa[mask])

    fig_val = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=(
            tr["p2_sub1"].format(id=r_sel["id"], q=r_sel["q"]),
            tr["p2_sub2"]
        )
    )

    fig_val.add_trace(
        go.Scatter(
            x=t_zoom, y=ref_zoom,
            mode="lines",
            name=tr["p2_ref"],
            line=dict(color="#190e30", width=2.9)
        ),
        row=1, col=1
    )

    if idx_arq in (0, 2):
        fig_val.add_trace(
            go.Scatter(
                x=t_zoom, y=r_sel["pred_dir"][mask],
                mode="lines",
                name=f"CosmicFNO ({r_sel['ov_dir']*100:.2f}% Overlap)",
                line=dict(color="#7c3aed", width=2.5, dash="dash")
            ),
            row=1, col=1
        )

    if idx_arq in (1, 2):
        fig_val.add_trace(
            go.Scatter(
                x=t_zoom, y=r_sel["pred_polar"][mask],
                mode="lines",
                name=f"Polar-CosmicFNO ({r_sel['ov_pol']*100:.2f}% Overlap)",
                line=dict(color="#c084fc", width=2.2, dash="dot")
            ),
            row=1, col=1
        )

    fig_val.add_trace(
        go.Scatter(
            x=t_zoom, y=erro_zoom,
            mode="lines",
            name="|Delta h|",
            line=dict(color="#6d28d9", width=1.6),
            fill="tozeroy",
            fillcolor="rgba(124, 58, 237, 0.12)"
        ),
        row=2, col=1
    )

    fig_val.add_vline(x=0.0, line_width=1.2, line_dash="dot", line_color="#8b7aa8")
    fig_val.update_layout(
        height=495,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#faf9fc",
        font=dict(family="Montserrat, sans-serif", color="#160e26", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(color="#160e26")),
        margin=dict(l=45, r=20, t=52, b=38)
    )
    fig_val.update_annotations(font=dict(color="#160e26", size=12))
    fig_val.update_yaxes(title_text="Strain Re(h_22)", row=1, col=1, gridcolor="#ede9f5", tickfont=dict(color="#554a6b"), title_font=dict(color="#160e26"))
    fig_val.update_yaxes(title_text="|Error|", row=2, col=1, gridcolor="#ede9f5", tickfont=dict(color="#554a6b"), title_font=dict(color="#160e26"))
    fig_val.update_xaxes(title_text=tr["p1_x_time"], row=2, col=1, gridcolor="#ede9f5", tickfont=dict(color="#554a6b"), title_font=dict(color="#160e26"))
    fig_val.update_xaxes(tickfont=dict(color="#554a6b"), row=1, col=1, gridcolor="#ede9f5")

    st.plotly_chart(fig_val, use_container_width=True, theme=None)


# =====================================================================
# PAGINA 3: EMULADOR CONTINUO E ESPECTRO DE FOURIER (FFT)
# =====================================================================
elif st.session_state["pagina_ativa"] == "espectro":
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        q_slider = st.slider(tr["p3_q"], min_value=1.0, max_value=8.0, value=3.0, step=0.1)
    with e_col2:
        spin_slider = st.slider(tr["p3_spin"], min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

    t_base = env["t_fase"][0]
    n_pts = len(t_base)
    t_norm_exp = (t_base - t_base.min()) / (t_base.max() - t_base.min())

    X_exp = torch.stack([
        torch.full((n_pts,), float(q_slider)),
        torch.full((n_pts,), float(spin_slider)),
        t_norm_exp
    ], dim=-1).unsqueeze(0)

    with torch.no_grad():
        onda_emulada = env["modelos"]["completo"](X_exp).squeeze().numpy()

    t_arr = t_base.numpy()
    espectro_amp = np.abs(np.fft.rfft(onda_emulada)) / n_pts
    modos_idx = np.arange(len(espectro_amp))

    p_left, p_right = st.columns([1.5, 1.0], gap="medium")
    with p_left:
        fig_t = go.Figure(
            go.Scatter(
                x=t_arr, y=onda_emulada,
                mode="lines",
                name="Strain h_22(t)",
                line=dict(color="#6d28d9", width=2.4)
            )
        )
        fig_t.update_layout(
            title=dict(text=tr["p3_c1"].format(q=q_slider, s=spin_slider), font=dict(color="#160e26", size=13)),
            height=415,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#faf9fc",
            font=dict(family="Montserrat, sans-serif", color="#160e26", size=11),
            margin=dict(l=45, r=20, t=52, b=38),
            xaxis=dict(title=dict(text=tr["p1_x_time"], font=dict(color="#160e26")), tickfont=dict(color="#554a6b"), gridcolor="#ede9f5"),
            yaxis=dict(title=dict(text="Strain Re(h_22)", font=dict(color="#160e26")), tickfont=dict(color="#554a6b"), gridcolor="#ede9f5")
        )
        st.plotly_chart(fig_t, use_container_width=True, theme=None)

    with p_right:
        fig_f = go.Figure(
            go.Bar(
                x=modos_idx[:64],
                y=espectro_amp[:64],
                marker_color=["#6d28d9" if k < 32 else "#dcd6e8" for k in range(64)]
            )
        )
        fig_f.update_layout(
            title=dict(text=tr["p3_c2"], font=dict(color="#160e26", size=13)),
            height=415,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#faf9fc",
            font=dict(family="Montserrat, sans-serif", color="#160e26", size=11),
            margin=dict(l=45, r=20, t=52, b=38),
            xaxis=dict(title=dict(text=tr["p3_xk"], font=dict(color="#160e26")), tickfont=dict(color="#554a6b"), gridcolor="#ede9f5"),
            yaxis=dict(title=dict(text=tr["p3_yf"], font=dict(color="#160e26")), tickfont=dict(color="#554a6b"), gridcolor="#ede9f5")
        )
        st.plotly_chart(fig_f, use_container_width=True, theme=None)


# =====================================================================
# PAGINA 4: COLISAO 3D SINCRONIZADA
# =====================================================================
elif st.session_state["pagina_ativa"] == "colisao3d":
    q_3d = st.select_slider(
        tr["p4_slider"],
        options=[1.07, 1.22, 4.00, 5.00, 8.00],
        value=4.00
    )

    mapa_q_idx = {1.22: 0, 1.07: 1, 8.00: 2, 5.00: 3, 4.00: 4}
    idx_3d = mapa_q_idx[q_3d]

    t_full = env["t_fase"][idx_3d].numpy()
    h_full = env["h_fase"][idx_3d].numpy()

    passo_3d = 22
    t_anim = t_full[-3080::passo_3d]
    h_anim = h_full[-3080::passo_3d]
    N_frames = len(t_anim)

    idx_merger = int(np.argmin(np.abs(t_anim)))
    raio_rel = np.zeros(N_frames)
    raio_rel[:idx_merger] = np.linspace(2.2, 0.15, idx_merger) ** 0.65
    omega_orb = np.linspace(1.5, 11.0, N_frames)
    fase_orb = np.cumsum(omega_orb) * 0.12

    r1 = raio_rel * (1.0 / (1.0 + q_3d)) * 2.0
    r2 = raio_rel * (q_3d / (1.0 + q_3d)) * 2.0

    x1, y1, z1 = r1 * np.cos(fase_orb), r1 * np.sin(fase_orb), np.zeros(N_frames)
    x2, y2, z2 = -r2 * np.cos(fase_orb), -r2 * np.sin(fase_orb), np.zeros(N_frames)

    fig_3d = make_subplots(
        rows=2, cols=1,
        row_heights=[0.58, 0.42],
        specs=[[{"type": "scatter3d"}], [{"type": "xy"}]],
        vertical_spacing=0.07,
        subplot_titles=(
            tr["p4_sub1"].format(q=q_3d),
            tr["p4_sub2"]
        )
    )

    tam_bh1 = max(11, int(22 * (q_3d / (1.0 + q_3d))))
    tam_bh2 = max(7, int(22 * (1.0 / (1.0 + q_3d))))

    fig_3d.add_trace(go.Scatter3d(x=x1[:1], y=y1[:1], z=z1[:1], mode="markers", marker=dict(size=tam_bh1, color="#190e30"), name=tr["p4_bh1"]), row=1, col=1)
    fig_3d.add_trace(go.Scatter3d(x=x2[:1], y=y2[:1], z=z2[:1], mode="markers", marker=dict(size=tam_bh2, color="#7c3aed"), name=tr["p4_bh2"]), row=1, col=1)
    fig_3d.add_trace(go.Scatter(x=t_anim[:1], y=h_anim[:1], mode="lines", line=dict(color="#6d28d9", width=2.8), name="h_22(t)"), row=2, col=1)
    fig_3d.add_trace(go.Scatter(x=[t_anim[0]], y=[h_anim[0]], mode="markers", marker=dict(color="#190e30", size=9), showlegend=False), row=2, col=1)

    fig_3d.frames = [
        go.Frame(
            data=[
                go.Scatter3d(x=[x1[i]], y=[y1[i]], z=[z1[i]]),
                go.Scatter3d(x=[x2[i]], y=[y2[i]], z=[z2[i]]),
                go.Scatter(x=t_anim[:i+1], y=h_anim[:i+1]),
                go.Scatter(x=[t_anim[i]], y=[h_anim[i]])
            ],
            traces=[0, 1, 2, 3]
        )
        for i in range(1, N_frames)
    ]

    fig_3d.update_layout(
        height=610,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#faf9fc",
        font=dict(family="Montserrat, sans-serif", color="#160e26", size=11),
        legend=dict(font=dict(color="#160e26")),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            font=dict(color="#160e26"),
            buttons=[
                dict(label=tr["p4_play"], method="animate", args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
                dict(label=tr["p4_pause"], method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            x=0.5, y=1.09, xanchor="center", yanchor="top"
        )],
        margin=dict(l=20, r=20, t=52, b=30)
    )
    fig_3d.update_annotations(font=dict(color="#160e26", size=12))
    fig_3d.update_scenes(
        xaxis=dict(range=[-2.3, 2.3], visible=False),
        yaxis=dict(range=[-2.3, 2.3], visible=False),
        zaxis=dict(range=[-1.5, 1.5], visible=False),
        bgcolor="#faf9fc",
        camera=dict(eye=dict(x=1.15, y=1.15, z=0.85))
    )
    fig_3d.update_xaxes(range=[float(t_anim.min()), float(t_anim.max())], title_text=tr["p1_x_time"], row=2, col=1, gridcolor="#ede9f5", tickfont=dict(color="#554a6b"), title_font=dict(color="#160e26"))
    fig_3d.update_yaxes(range=[float(h_anim.min()*1.25), float(h_anim.max()*1.25)], title_text="Strain Re(h_22)", row=2, col=1, gridcolor="#ede9f5", tickfont=dict(color="#554a6b"), title_font=dict(color="#160e26"))

    st.plotly_chart(fig_3d, use_container_width=True, theme=None)


# =====================================================================
# PAGINA 5: RESUMO QUANTITATIVO DOS EXPERIMENTOS
# =====================================================================
elif st.session_state["pagina_ativa"] == "tabela":
    b3300 = env["bench_3300"]

    render_kpi_grid([
        {"label": tr["p5_k1_t"], "value": f"{b3300['n_total']:,}".replace(",", "."), "sub": tr["p5_k1_s"].format(n=b3300["n_teste"])},
        {"label": tr["p5_k2_t"], "value": f"{100.0 - b3300['erro_rel_massa']:.2f}%", "sub": tr["p5_k2_s"].format(e=b3300["erro_rel_massa"])},
        {"label": tr["p5_k3_t"], "value": f"{b3300['r2_spin']*100:.2f}%", "sub": f"R² Kick: {b3300['r2_kick']*100:.1f}%"},
        {"label": tr["p5_k4_t"], "value": f"{b3300['acc_aglomerado']:.2f}%", "sub": tr["p5_k4_s"]},
    ])

    linhas_tabela = [
        ("Exp 1 · Baseline 1D-CNN Reconstruction (SXS:BBH:0305)", "96.67%", "25.88%", "0.741196"),
        ("Exp 2 · Initial Blind Test (4 train sims, q=5.00 unaligned phase)", "115.46%", "16.70%", "0.832960"),
        ("Exp 3 · Phase-Aligned Blind Test (4 train sims, q=5.00)", "39.66%", "94.16%", "0.058402"),
        ("Exp 4 · SXS Scale Blind Test (11 train sims -> SXS:BBH:0113, q=5.00) [Polar-FNO]", "11.50%", "99.34%", "0.006640"),
        ("Exp 4 · SXS Scale Blind Test (11 train sims -> SXS:BBH:0113, q=5.00) [Direct FNO]", "4.34%", "99.91%", "0.000920"),
        ("Exp 4 · SXS Scale Blind Test (11 train sims -> SXS:BBH:0201, q=2.32) [Direct FNO]", "2.66%", "99.97%", "0.000300"),
        ("Exp 4 · SXS Scale Blind Test (11 train sims -> SXS:BBH:0593, q=1.50) [Direct FNO]", "1.81%", "99.98%", "0.000160"),
        ("Exp 5 · Pre-Merger Hidden Future Forecast (t > -50M, SXS:BBH:0167)", "1.48%", "99.99%", "0.000100"),
    ]

    trs_html = "".join(
        f'<tr><td>{cen}</td><td>{l2}</td><td><span class="tbl-pill">{ov}</span></td><td>{mis}</td></tr>'
        for cen, l2, ov, mis in linhas_tabela
    )

    st.markdown(
        f'<div class="table-card">'
        f'<table class="modern-table">'
        f'<thead><tr><th>Experiment / Hold-Out Scenario</th><th>Relative L2 Error</th><th>Overlap (Fidelity)</th><th>Mismatch (1 - Overlap)</th></tr></thead>'
        f'<tbody>{trs_html}</tbody>'
        f'</table>'
        f'</div>',
        unsafe_allow_html=True
    )

# =====================================================================
# RODAPE DO PORTAL CIENTIFICO
# =====================================================================
st.markdown(f"""
<div class="site-footer">
    <span>{tr['footer_l']}</span>
    <span>{tr['footer_r']}</span>
</div>
""", unsafe_allow_html=True)
