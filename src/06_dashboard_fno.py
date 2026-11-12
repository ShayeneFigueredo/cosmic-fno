"""
CosmicFNO - Plataforma Visual e Observatorio SciML
Autora: Shayene Figueredo (BSI/UFU)

Descricao:
    Interface responsiva (Desktop e Mobile) com estetica SaaS Clean:
    - Logo oficial CosmicFNO no menu lateral (Sidebar em Azul Marinho Profundo #081226)
    - Destaque em Azul Royal #2563eb, Topbar branca minimalista, fonte Montserrat
    - Layout organizado e adaptavel para telas de celular e computador.
"""

import os
import sys
import time
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
    page_title="CosmicFNO | Lab",
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
# 1. CSS CUSTOMIZADO E RESPONSIVO (DESKTOP + MOBILE, FONTE MONTSERRAT)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, p, div, span, label, h1, h2, h3, h4, h5, h6, button, input, select, textarea, table, th, td {
        font-family: 'Montserrat', sans-serif !important;
    }

    /* Mantem o botao de menu mobile visivel e limpo */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-left: 2.0rem !important;
        padding-right: 2.0rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 100% !important;
    }

    .stApp {
        background-color: #f4f6f9 !important;
        color: #0f172a !important;
    }

    /* SIDEBAR: Azul Marinho Bem Escuro (#081226) */
    section[data-testid="stSidebar"] {
        background-color: #081226 !important;
        border-right: 1px solid #132347 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    /* Container da Logo Oficial na Sidebar */
    .sidebar-logo-box {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 4px 0 22px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    }
    .sidebar-logo-box img {
        width: 100%;
        max-width: 195px;
        height: auto;
        display: block;
    }

    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #64748b !important;
        text-transform: uppercase;
        margin: 18px 0 8px 6px;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 11px 16px !important;
        border-radius: 10px !important;
        border: none !important;
        background-color: transparent !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        transition: all 0.18s ease;
        margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: rgba(37, 99, 235, 0.18) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
    }

    /* TOPBAR Branca Organizada */
    .topbar {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 22px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
    }
    .topbar-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .topbar-greeting {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .topbar-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin: 0;
    }
    .topbar-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-pill {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .avatar-circle {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #081226, #1d4ed8);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.78rem;
        flex-shrink: 0;
    }

    /* Cartoes de Metricas */
    div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 4px solid #1d4ed8 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
    }

    .white-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        margin-bottom: 14px;
    }
    .card-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #081226;
        margin-bottom: 6px;
    }
    .card-sub {
        font-size: 0.84rem;
        color: #64748b;
        margin-bottom: 0;
        line-height: 1.55;
    }

    .verdict-banner {
        background: linear-gradient(90deg, #081226 0%, #1e3a8a 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 16px;
        border-left: 5px solid #3b82f6;
    }
    .verdict-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #93c5fd;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .verdict-main {
        font-size: 0.96rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 3px;
    }
    .verdict-desc {
        font-size: 0.82rem;
        color: #dbeafe;
        margin: 0;
        line-height: 1.5;
    }

    /* RESPONSIVIDADE MOBILE (Celulares e Tablets) */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-top: 0.25rem !important;
        }
        .topbar {
            padding: 12px 14px;
            margin-bottom: 12px;
        }
        .topbar-greeting {
            font-size: 0.98rem;
        }
        .status-pill {
            font-size: 0.68rem;
            padding: 4px 9px;
        }
        .white-card {
            padding: 14px 16px;
        }
        .verdict-banner {
            padding: 13px 15px;
        }
        .verdict-main {
            font-size: 0.88rem;
        }
        div[data-testid="metric-container"] {
            padding: 11px 14px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. CARREGAMENTO DOS MODELOS E TENSORES SXS
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
    {"idx": 1, "id": "SXS:BBH:0115", "q": 1.07, "spin": -0.20, "tipo": "Chute Extremo (354 km/s)"},
    {"idx": 4, "id": "SXS:BBH:0167", "q": 4.00, "spin": 0.00, "tipo": "Colisao Assimetrica (q=4.00)"},
    {"idx": 3, "id": "SXS:BBH:0113", "q": 5.00, "spin": 0.00, "tipo": "Colisao Assimetrica (q=5.00)"},
    {"idx": 0, "id": "SXS:BBH:0305", "q": 1.22, "spin": 0.33, "tipo": "Referencia GW150914"},
    {"idx": 2, "id": "SXS:BBH:0114", "q": 8.00, "spin": -0.50, "tipo": "Razao de Massa Extrema (q=8.00)"},
]

# =====================================================================
# 3. MENU LATERAL COM A LOGO OFICIAL COSMICFNO
# =====================================================================
if "pagina_ativa" not in st.session_state:
    st.session_state["pagina_ativa"] = "previsao"

with st.sidebar:
    if logo_b64:
        st.markdown(
            f'<div class="sidebar-logo-box"><img src="data:image/png;base64,{logo_b64}" alt="CosmicFNO Logo"></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-section-label">MODULOS PRINCIPAIS</div>', unsafe_allow_html=True)

    if st.button("Previsao Pre-Merger e Kick", type="primary" if st.session_state["pagina_ativa"] == "previsao" else "secondary"):
        st.session_state["pagina_ativa"] = "previsao"
        st.rerun()

    if st.button("Validacao Hold-Out (99.9%)", type="primary" if st.session_state["pagina_ativa"] == "validacao" else "secondary"):
        st.session_state["pagina_ativa"] = "validacao"
        st.rerun()

    st.markdown('<div class="sidebar-section-label">EXPLORACAO VISUAL</div>', unsafe_allow_html=True)

    if st.button("Emulador e Espectro FFT", type="primary" if st.session_state["pagina_ativa"] == "espectro" else "secondary"):
        st.session_state["pagina_ativa"] = "espectro"
        st.rerun()

    if st.button("Colisao 3D Sincronizada", type="primary" if st.session_state["pagina_ativa"] == "colisao3d" else "secondary"):
        st.session_state["pagina_ativa"] = "colisao3d"
        st.rerun()

    if st.button("Tabela de Metricas", type="primary" if st.session_state["pagina_ativa"] == "tabela" else "secondary"):
        st.session_state["pagina_ativa"] = "tabela"
        st.rerun()

    st.markdown('<div class="sidebar-section-label">PESQUISA</div>', unsafe_allow_html=True)
    st.caption("3.435 Simulacoes SXS BBH\n14 Formas de Onda Alinhadas\nShayene Figueredo — UFU")


# =====================================================================
# 4. TOPBAR BRANCA ORGANIZADA (RESPONSIVA DESKTOP & MOBILE)
# =====================================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div>
            <p class="topbar-greeting">Olá, Shayene</p>
            <p class="topbar-sub">Plataforma de Emulação Espectral e Diagnóstico de Buracos Negros Binários</p>
        </div>
    </div>
    <div class="topbar-right">
        <span class="status-pill">Catálogo SXS v3.0.0 — 3.435 Colisões</span>
        <div class="avatar-circle">SF</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGINA 1: PREVISAO ANTECIPADA DA COLISAO (PRE-MERGER) E BURACO NEGRO FUGITIVO
# =====================================================================
if st.session_state["pagina_ativa"] == "previsao":
    b3300 = env["bench_3300"]

    # Barra de controle superior organizada (funciona perfeitamente em Desktop e Mobile)
    with st.container():
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Previsão Antecipada da Colisão (Pre-Merger) e Diagnóstico de Fuga Galáctica</div>
            <div class="card-sub">
                O modelo observa apenas a fase inicial da espiral (antes de t = -50M, com o impacto oculto) para prever a onda completa da colisão e estimar a massa final, o spin e a velocidade de ejeção (Recoil Kick) do buraco negro formado.
            </div>
        </div>
        """, unsafe_allow_html=True)

        f_col1, f_col2, f_col3 = st.columns([1.5, 1.2, 1.3])
        with f_col1:
            opcoes_pred = [f"{item['id']} — q = {item['q']:.2f} ({item['tipo']})" for item in CATALOGO_INFO]
            escolha_pred = st.selectbox("Evento Binario Monitorado:", opcoes_pred, index=0)
            info_p = CATALOGO_INFO[opcoes_pred.index(escolha_pred)]

        with f_col2:
            massa_total_sol = st.slider(
                "Massa Total Inicial (Massas Solares):",
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
            <div class="white-card" style="padding: 12px 16px; margin-bottom: 0; border-left: 4px solid #081226;">
                <div class="card-title" style="font-size:0.84rem; margin-bottom: 3px;">Configuração Pré-Impacto</div>
                <div class="card-sub" style="font-size:0.8rem;">
                    <b>m1:</b> {m1_sol:.1f} M_sol | <b>m2:</b> {m2_sol:.1f} M_sol | <b>Precisão SXS-3435:</b> {100.0 - b3300['erro_rel_massa']:.2f}%
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

    # Cartoes de Metricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fidelidade Pre-Merger", f"{ov_futuro*100:.2f}%", "Futuro Oculto (t > -50M)")
    c2.metric("Novo Buraco Negro (M_f)", f"{m_final_sol:.1f} M_sol", f"{energia_sol:.2f} M_sol irradiados ({energia_pct:.1f}%)", delta_color="off")
    c3.metric("Spin Final Previsto", f"{spin_final_pred:.3f} c", f"SXS Real: {rem_real[1]:.3f} c", delta_color="off")
    c4.metric("Chute Relativistico (Kick)", f"{v_kick_pred_kms:.1f} km/s", f"SXS Real: {rem_real[2]:.1f} km/s")

    if v_kick_pred_kms >= 300.0:
        titulo_v = "VEREDITO ASTROFISICO: BURACO NEGRO FUGITIVO INTERGALACTICO"
        desc_v = (
            f"O chute gravitacional previsto ({v_kick_pred_kms:.1f} km/s = {v_kick_pred_kms*3600:,.0f} km/h) "
            f"ultrapassa a velocidade de escape de Aglomerados Globulares (50 km/s), Galaxias Anas (150 km/s) "
            f"e Galaxias Medias (300 km/s). O buraco negro recem-nascido sera ejetado do seu nucleo!"
        )
    elif v_kick_pred_kms >= 130.0:
        titulo_v = "VEREDITO ASTROFISICO: EJECAO DE AGLOMERADOS E GALAXIAS ANAS"
        desc_v = (
            f"O chute gravitacional previsto ({v_kick_pred_kms:.1f} km/s = {v_kick_pred_kms*3600:,.0f} km/h) "
            f"rompe a gravidade de Aglomerados Estelares Globulares (50 km/s) e Galaxias Satelites Anas, "
            f"mas permanece retido no halo de galaxias massivas como a Via Lactea (550 km/s)."
        )
    else:
        titulo_v = "VEREDITO ASTROFISICO: EJECAO DE AGLOMERADO ESTELAR (RETIDO NA GALAXIA)"
        desc_v = (
            f"O chute gravitacional previsto ({v_kick_pred_kms:.1f} km/s) "
            f"supera a velocidade de escape de Aglomerados Globulares (50 km/s), mas o buraco negro "
            f"permanecera preso ao potencial gravitacional da galaxia hospedeira."
        )

    st.markdown(f"""
    <div class="verdict-banner">
        <div class="verdict-title">DIAGNOSTICO DE RECUO RELATIVISTICO (VALIDADO EM {b3300['n_total']} COLISOES SXS — {b3300['acc_aglomerado']:.2f}% ACURACIA)</div>
        <div class="verdict-main">{titulo_v}</div>
        <p class="verdict-desc">{desc_v}</p>
    </div>
    """, unsafe_allow_html=True)

    # Dois graficos em colunas independentes (empilham automaticamente no celular!)
    g_col1, g_col2 = st.columns([1.65, 1.0], gap="medium")

    with g_col1:
        mask_obs = (t_np >= -220.0) & (t_np <= t_corte)
        mask_fut_plot = (t_np >= t_corte) & (t_np <= 80.0)

        fig_onda = go.Figure()
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_obs], y=h_real_np[mask_obs],
                mode="lines",
                name="Espiral Observada (Entrada)",
                line=dict(color="#64748b", width=2.4)
            )
        )
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_real_np[mask_fut_plot],
                mode="lines",
                name="Impacto Real SXS (Oculto)",
                line=dict(color="#081226", width=3.0)
            )
        )
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_pred_np[mask_fut_plot],
                mode="lines",
                name="Previsao Futura CosmicFNO",
                line=dict(color="#2563eb", width=2.5, dash="dash")
            )
        )
        fig_onda.add_vrect(
            x0=t_corte, x1=80.0,
            fillcolor="rgba(37, 99, 235, 0.07)",
            layer="below", line_width=0
        )
        fig_onda.add_vline(x=t_corte, line_width=1.8, line_dash="dash", line_color="#1d4ed8")
        fig_onda.update_layout(
            title="Previsao Antecipada da Colisao (Pre-Merger em t = -50M)",
            height=410,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=45, r=20, t=65, b=40),
            xaxis=dict(title="Tempo Fisico Alinhado (M)", gridcolor="#e2e8f0"),
            yaxis=dict(title="Strain Re(h_22)", gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_onda, use_container_width=True)

    with g_col2:
        categorias_esc = ["Aglomerado", "Galaxia Ana", "CHUTE IA", "Gal. Media", "Via Lactea"]
        valores_esc = [50.0, 150.0, v_kick_pred_kms, 300.0, 550.0]
        cores_esc = ["#cbd5e1", "#94a3b8", "#2563eb", "#64748b", "#081226"]

        fig_bar = go.Figure(
            go.Bar(
                x=categorias_esc,
                y=valores_esc,
                marker_color=cores_esc,
                text=[f"{v:.0f} km/s" for v in valores_esc],
                textposition="outside"
            )
        )
        fig_bar.update_layout(
            title="Chute Relativistico vs Velocidade de Escape",
            height=410,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            showlegend=False,
            margin=dict(l=40, r=20, t=65, b=40),
            xaxis=dict(gridcolor="#e2e8f0"),
            yaxis=dict(title="Velocidade (km/s)", range=[0, 650], gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# =====================================================================
# PAGINA 2: VALIDACAO CIENTIFICA HOLD-OUT (BENCHMARK 14 ONDAS SXS — 99.98% OVERLAP)
# =====================================================================
elif st.session_state["pagina_ativa"] == "validacao":
    res_lista = env["bench_polar"]["resultados_teste"]

    v_col1, v_col2, v_col3 = st.columns([1.3, 1.3, 1.4])
    with v_col1:
        opcoes_bench = [f"{r['id']} — q = {r['q']:.2f} (Overlap: {r['ov_dir']*100:.2f}%)" for r in res_lista]
        escolha_b = st.selectbox("Simulacao Secreta de Teste Cego:", opcoes_bench, index=0)
        r_sel = res_lista[opcoes_bench.index(escolha_b)]

    with v_col2:
        arq_comparada = st.selectbox(
            "Arquitetura Neural Exibida:",
            [
                "CosmicFNO Espectral Direto (Melhor)",
                "Polar-CosmicFNO (Amplitude-Fase)",
                "Ambas Sobrepostas"
            ],
            index=0
        )

    with v_col3:
        janela_zoom = st.slider(
            "Janela de Tempo Fisico (M):",
            min_value=-330.0,
            max_value=90.0,
            value=(-280.0, 80.0),
            step=5.0
        )

    t_np = np.linspace(-330.0, 90.0, 4096)
    ref_onda = r_sel["ref"]
    pred_ativa = r_sel["pred_dir"] if not arq_comparada.startswith("Polar") else r_sel["pred_polar"]

    ov_val = r_sel["ov_dir"] if not arq_comparada.startswith("Polar") else r_sel["ov_pol"]
    mis_val = r_sel["mis_dir"] if not arq_comparada.startswith("Polar") else r_sel["mis_pol"]
    l2_val = r_sel["l2_dir"] if not arq_comparada.startswith("Polar") else r_sel["l2_pol"]
    mse_val = met.calcular_mse(ref_onda, pred_ativa)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overlap (Teste Cego)", f"{ov_val*100:.2f}%", f"Mismatch: {mis_val:.5f}")
    k2.metric("Mismatch (1 - Overlap)", f"{mis_val:.5f}", "Meta LIGO < 0.00100")
    k3.metric("Erro Relativo L2", f"{l2_val*100:.2f}%", f"MSE: {mse_val:.6f}", delta_color="off")
    k4.metric("Amostras de Treino", "11 Simulacoes SXS", "Grade q in [1.0, 8.0]", delta_color="off")

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
            f"Teste Cego Hold-Out: {r_sel['id']} (q = {r_sel['q']:.2f} — Simulacao Nunca Vista no Treino)",
            "Erro Absoluto Instantaneo |h_SXS(t) - h_FNO(t)|"
        )
    )

    fig_val.add_trace(
        go.Scatter(
            x=t_zoom, y=ref_zoom,
            mode="lines",
            name="Onda de Referencia SXS",
            line=dict(color="#081226", width=3.0)
        ),
        row=1, col=1
    )

    if arq_comparada in ("CosmicFNO Espectral Direto (Melhor)", "Ambas Sobrepostas"):
        fig_val.add_trace(
            go.Scatter(
                x=t_zoom, y=r_sel["pred_dir"][mask],
                mode="lines",
                name=f"CosmicFNO Direto ({r_sel['ov_dir']*100:.2f}% Overlap)",
                line=dict(color="#2563eb", width=2.5, dash="dash")
            ),
            row=1, col=1
        )

    if arq_comparada in ("Polar-CosmicFNO (Amplitude-Fase)", "Ambas Sobrepostas"):
        fig_val.add_trace(
            go.Scatter(
                x=t_zoom, y=r_sel["pred_polar"][mask],
                mode="lines",
                name=f"Polar-CosmicFNO ({r_sel['ov_pol']*100:.2f}% Overlap)",
                line=dict(color="#06b6d4", width=2.2, dash="dot")
            ),
            row=1, col=1
        )

    fig_val.add_trace(
        go.Scatter(
            x=t_zoom, y=erro_zoom,
            mode="lines",
            name="Residuo |Delta h|",
            line=dict(color="#0284c7", width=1.6),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)"
        ),
        row=2, col=1
    )

    fig_val.add_vline(x=0.0, line_width=1.2, line_dash="dot", line_color="#64748b")
    fig_val.update_layout(
        height=520,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        margin=dict(l=45, r=20, t=55, b=40)
    )
    fig_val.update_yaxes(title_text="Strain Re(h_22)", row=1, col=1, gridcolor="#e2e8f0")
    fig_val.update_yaxes(title_text="|Erro|", row=2, col=1, gridcolor="#e2e8f0")
    fig_val.update_xaxes(title_text="Tempo Fisico Alinhado pelo Merger (t = 0)", row=2, col=1, gridcolor="#e2e8f0")

    st.plotly_chart(fig_val, use_container_width=True)


# =====================================================================
# PAGINA 3: EMULADOR CONTINUO E ESPECTRO DE FOURIER (FFT)
# =====================================================================
elif st.session_state["pagina_ativa"] == "espectro":
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        q_slider = st.slider("Razao de Massa (q = m1/m2)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
    with e_col2:
        spin_slider = st.slider("Spin Orbital (chi_1z)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

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
                line=dict(color="#1d4ed8", width=2.4)
            )
        )
        fig_t.update_layout(
            title=f"Forma de Onda Emulada no Tempo (q={q_slider:.1f}, chi_1z={spin_slider:+.2f})",
            height=430,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            margin=dict(l=45, r=20, t=55, b=40),
            xaxis=dict(title="Tempo Fisico (M)", gridcolor="#e2e8f0"),
            yaxis=dict(title="Strain Re(h_22)", gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with p_right:
        fig_f = go.Figure(
            go.Bar(
                x=modos_idx[:64],
                y=espectro_amp[:64],
                marker_color=["#1d4ed8" if k < 32 else "#cbd5e1" for k in range(64)]
            )
        )
        fig_f.update_layout(
            title="Espectro de Fourier (32 Modos Ativos no FNO)",
            height=430,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            margin=dict(l=45, r=20, t=55, b=40),
            xaxis=dict(title="Modo de Frequencia k", gridcolor="#e2e8f0"),
            yaxis=dict(title="Amplitude |FFT|", gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_f, use_container_width=True)


# =====================================================================
# PAGINA 4: COLISAO 3D SINCRONIZADA
# =====================================================================
elif st.session_state["pagina_ativa"] == "colisao3d":
    q_3d = st.select_slider(
        "Selecione a Razao de Massa (q) da Simulacao SXS:",
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
            f"Coalescencia Binaria 3D (q = {q_3d:.2f})",
            "Formacao Sincronizada da Onda Gravitacional Re(h_22)"
        )
    )

    tam_bh1 = max(11, int(22 * (q_3d / (1.0 + q_3d))))
    tam_bh2 = max(7, int(22 * (1.0 / (1.0 + q_3d))))

    fig_3d.add_trace(go.Scatter3d(x=x1[:1], y=y1[:1], z=z1[:1], mode="markers", marker=dict(size=tam_bh1, color="#081226"), name="Buraco Negro 1 (m1)"), row=1, col=1)
    fig_3d.add_trace(go.Scatter3d(x=x2[:1], y=y2[:1], z=z2[:1], mode="markers", marker=dict(size=tam_bh2, color="#2563eb"), name="Buraco Negro 2 (m2)"), row=1, col=1)
    fig_3d.add_trace(go.Scatter(x=t_anim[:1], y=h_anim[:1], mode="lines", line=dict(color="#1d4ed8", width=2.8), name="Onda h_22"), row=2, col=1)
    fig_3d.add_trace(go.Scatter(x=[t_anim[0]], y=[h_anim[0]], mode="markers", marker=dict(color="#081226", size=9), showlegend=False), row=2, col=1)

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
        height=640,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Reproduzir Colisao", method="animate", args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
                dict(label="Pausar", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            x=0.5, y=1.09, xanchor="center", yanchor="top"
        )],
        margin=dict(l=20, r=20, t=55, b=30)
    )
    fig_3d.update_scenes(
        xaxis=dict(range=[-2.3, 2.3], visible=False),
        yaxis=dict(range=[-2.3, 2.3], visible=False),
        zaxis=dict(range=[-1.5, 1.5], visible=False),
        bgcolor="#f8fafc",
        camera=dict(eye=dict(x=1.15, y=1.15, z=0.85))
    )
    fig_3d.update_xaxes(range=[float(t_anim.min()), float(t_anim.max())], title_text="Tempo Fisico Alinhado (t = 0 no Merger)", row=2, col=1, gridcolor="#e2e8f0")
    fig_3d.update_yaxes(range=[float(h_anim.min()*1.25), float(h_anim.max()*1.25)], title_text="Strain Re(h_22)", row=2, col=1, gridcolor="#e2e8f0")

    st.plotly_chart(fig_3d, use_container_width=True)


# =====================================================================
# PAGINA 5: RESUMO QUANTITATIVO DOS EXPERIMENTOS
# =====================================================================
elif st.session_state["pagina_ativa"] == "tabela":
    st.markdown("""
    <div class="white-card">
        <div class="card-title">Sintese Experimental — Escalonamento SXS (14 Formas de Onda & 3.435 Colisoes)</div>
        <div class="card-sub">Resultados consolidados gerados pelos modulos 08, 09, 10 e 11.</div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        {
            "Experimento / Cenario (Hold-Out Cego)": [
                "Exp 1 — Reconstrucao Baseline 1D-CNN (SXS:BBH:0305)",
                "Exp 2 — Teste Cego Inicial (4 sims treino, q=5.00 sem alinhar fase)",
                "Exp 3 — Teste Cego Fase Alinhada (4 sims treino, q=5.00)",
                "Exp 4 — Teste Cego Escala SXS (11 sims treino -> SXS:BBH:0113, q=5.00) [Polar-FNO]",
                "Exp 4 — Teste Cego Escala SXS (11 sims treino -> SXS:BBH:0113, q=5.00) [FNO Direto]",
                "Exp 4 — Teste Cego Escala SXS (11 sims treino -> SXS:BBH:0201, q=2.32) [FNO Direto]",
                "Exp 4 — Teste Cego Escala SXS (11 sims treino -> SXS:BBH:0593, q=1.50) [FNO Direto]",
                "Exp 5 — Previsao Pre-Merger Futuro Oculto (t > -50M, SXS:BBH:0167)",
            ],
            "Erro Relativo L2": ["96.67%", "115.46%", "39.66%", "11.50%", "4.34%", "2.66%", "1.81%", "1.48%"],
            "Overlap (Fidelidade)": ["25.88%", "16.70%", "94.16%", "99.34%", "99.91%", "99.97%", "99.98%", "99.99%"],
            "Mismatch (1 - Overlap)": ["0.741196", "0.832960", "0.058402", "0.006640", "0.000920", "0.000300", "0.000160", "0.000100"],
        },
        use_container_width=True,
        hide_index=True
    )
