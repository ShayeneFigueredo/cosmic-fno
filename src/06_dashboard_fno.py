"""
CosmicFNO - Plataforma Visual e Observatorio SciML
Autora: Shayene Figueredo (BSI/UFU)

Descricao:
    Interface com estetica SaaS Clean (Sidebar em Azul Marinho Profundo #081226,
    Destaque em Azul Royal #2563eb, Topbar branca minimalista, fonte Montserrat
    e Area de Trabalho clara #f4f6f9, sem emojis).
    Inclui:
    1. Previsao Pre-Merger e Preditor de Buracos Negros Fugitivos validado em 3.435 colisoes SXS.
    2. Validacao Hold-Out Cega em 14 Simulacoes SXS (99.98% de Overlap / Mismatch 0.00016)
       comparando CosmicFNO Espectral Direto vs Polar-CosmicFNO (Amplitude-Fase).
"""

import os
import sys
import time
import importlib.util
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="CosmicFNO | Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 1. CSS CUSTOMIZADO (Fonte Montserrat + Azul Marinho Profundo, Sem Emojis)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, p, div, span, label, h1, h2, h3, h4, h5, h6, button, input, select, textarea, table, th, td {
        font-family: 'Montserrat', sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0rem !important;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 100% !important;
    }

    .stApp {
        background-color: #f4f6f9 !important;
        color: #0f172a !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #081226 !important;
        border-right: 1px solid #132347 !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 10px 0 26px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 18px;
    }
    .brand-icon {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    .brand-text-main {
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #ffffff !important;
        line-height: 1.0;
    }
    .brand-text-sub {
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 2.2px;
        color: #60a5fa !important;
        margin-top: 3px;
    }
    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #475569 !important;
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
        background-color: rgba(37, 99, 235, 0.15) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
    }

    .topbar {
        background-color: #ffffff;
        margin: 0 -2.2rem 1.8rem -2.2rem;
        padding: 16px 2.4rem;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .topbar-greeting {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .topbar-right {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .status-pill {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
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
        font-size: 0.8rem;
    }

    div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 4px solid #1d4ed8 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
    }

    .white-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #081226;
        margin-bottom: 6px;
    }
    .card-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 0;
    }
    .verdict-banner {
        background: linear-gradient(90deg, #081226 0%, #1e3a8a 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 12px;
        margin-bottom: 16px;
        border-left: 5px solid #3b82f6;
    }
    .verdict-title {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 1.3px;
        color: #93c5fd;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .verdict-main {
        font-size: 1.0rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .verdict-desc {
        font-size: 0.84rem;
        color: #dbeafe;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. CARREGAMENTO DOS MODELOS E TENSORES SXS (INCLUINDO BENCHMARK 14 ONDAS E 3435 COLISOES)
# =====================================================================
@st.cache_resource
def carregar_ambiente_cientifico():
    pasta_src = os.path.dirname(os.path.abspath(__file__))
    pasta_base = os.path.dirname(pasta_src)
    sys.path.append(pasta_src)

    fno_spec = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(pasta_src, "02_modelo_fno.py"))
    fno_mod = importlib.util.module_from_spec(fno_spec)
    fno_spec.loader.exec_module(fno_mod)

    met_spec = importlib.util.spec_from_file_location("08_metricas_fisicas", os.path.join(pasta_src, "08_metricas_fisicas.py"))
    met_mod = importlib.util.module_from_spec(met_spec)
    met_spec.loader.exec_module(met_mod)

    rem_spec = importlib.util.spec_from_file_location("10_previsao_pre_merger_e_remanescente", os.path.join(pasta_src, "10_previsao_pre_merger_e_remanescente.py"))
    rem_mod = importlib.util.module_from_spec(rem_spec)
    rem_spec.loader.exec_module(rem_mod)

    device = torch.device("cpu")
    pasta_sxs = os.path.join(pasta_base, "data", "raw", "dataset_sxs")
    pasta_modelos = os.path.join(pasta_base, "modelos_salvos")

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
# 3. MENU LATERAL (SIDEBAR AZUL MARINHO PROFUNDO SEM EMOJIS)
# =====================================================================
if "pagina_ativa" not in st.session_state:
    st.session_state["pagina_ativa"] = "previsao"

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">CF</div>
        <div>
            <div class="brand-text-main">COSMIC</div>
            <div class="brand-text-sub">FNO LAB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Previsao Pre-Merger e Kick", type="primary" if st.session_state["pagina_ativa"] == "previsao" else "secondary"):
        st.session_state["pagina_ativa"] = "previsao"
        st.rerun()

    if st.button("Validacao Hold-Out (99.9%)", type="primary" if st.session_state["pagina_ativa"] == "validacao" else "secondary"):
        st.session_state["pagina_ativa"] = "validacao"
        st.rerun()

    st.markdown('<div class="sidebar-section-label">LABORATORIO</div>', unsafe_allow_html=True)

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
# 4. TOPBAR BRANCA MINIMALISTA ("Ola, Shayene")
# =====================================================================
st.markdown("""
<div class="topbar">
    <p class="topbar-greeting">Olá, Shayene</p>
    <div class="topbar-right">
        <span class="status-pill">Catalogo SXS v3.0.0 — 3.435 Colisoes Validadas</span>
        <div class="avatar-circle">SF</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGINA 1: PREVISAO ANTECIPADA DA COLISAO (PRE-MERGER) E BURACO NEGRO FUGITIVO
# =====================================================================
if st.session_state["pagina_ativa"] == "previsao":
    b3300 = env["bench_3300"]
    col_ctrl, col_main = st.columns([1.05, 2.95], gap="large")

    with col_ctrl:
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Alerta Pre-Merger & Escape Galactico</div>
            <div class="card-sub">
                A IA observa apenas a fase inicial da espiral (t < -50M) e preve a colisao futura e a velocidade de ejecao do novo buraco negro.
            </div>
        </div>
        """, unsafe_allow_html=True)

        opcoes_pred = [f"{item['id']} — q = {item['q']:.2f} ({item['tipo']})" for item in CATALOGO_INFO]
        escolha_pred = st.selectbox("Evento Binario Monitorado:", opcoes_pred, index=0)
        info_p = CATALOGO_INFO[opcoes_pred.index(escolha_pred)]

        massa_total_sol = st.slider(
            "Massa Total do Sistema (Massas Solares):",
            min_value=20.0,
            max_value=150.0,
            value=65.0,
            step=5.0
        )

        q_val = info_p["q"]
        m1_sol = massa_total_sol * (q_val / (1.0 + q_val))
        m2_sol = massa_total_sol * (1.0 / (1.0 + q_val))

        st.markdown(f"""
        <div class="white-card" style="margin-top: 12px; border-left: 4px solid #081226;">
            <div class="card-title" style="font-size:0.9rem;">Validacao em Escala (3.435 Colisoes SXS)</div>
            <div class="card-sub" style="line-height: 1.75;">
                • <b>Treino / Teste Cego:</b> {b3300['n_treino']} / {b3300['n_teste']} colisoes<br>
                • <b>Precisao Massa Final (M_f):</b> {100.0 - b3300['erro_rel_massa']:.2f}%<br>
                • <b>R² Spin Final (chi_f):</b> {b3300['r2_spin']*100:.2f}%<br>
                • <b>Acerto Ejecao Aglomerado:</b> {b3300['acc_aglomerado']:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_main:
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
            <div class="verdict-title">DIAGNOSTICO DE RECUO RELATIVISTICO (VALIDADO EM 3.435 COLISOES SXS)</div>
            <div class="verdict-main">{titulo_v}</div>
            <p class="verdict-desc">{desc_v}</p>
        </div>
        """, unsafe_allow_html=True)

        fig_pred = make_subplots(
            rows=1, cols=2,
            column_widths=[0.65, 0.35],
            subplot_titles=(
                "Previsao Antecipada da Colisao (Passado Observado vs Impacto Previsto)",
                "Velocidade do Chute vs Velocidade de Escape (km/s)"
            )
        )

        mask_obs = (t_np >= -220.0) & (t_np <= t_corte)
        mask_fut_plot = (t_np >= t_corte) & (t_np <= 80.0)

        fig_pred.add_trace(
            go.Scatter(
                x=t_np[mask_obs], y=h_real_np[mask_obs],
                mode="lines",
                name="Espiral Observada (Entrada da IA)",
                line=dict(color="#64748b", width=2.5)
            ),
            row=1, col=1
        )
        fig_pred.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_real_np[mask_fut_plot],
                mode="lines",
                name="Impacto Real SXS (Oculto para a IA)",
                line=dict(color="#081226", width=3.0)
            ),
            row=1, col=1
        )
        fig_pred.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_pred_np[mask_fut_plot],
                mode="lines",
                name="Previsao Futura CosmicFNO",
                line=dict(color="#2563eb", width=2.6, dash="dash")
            ),
            row=1, col=1
        )

        fig_pred.add_vrect(
            x0=t_corte, x1=80.0,
            fillcolor="rgba(37, 99, 235, 0.07)",
            layer="below", line_width=0,
            row=1, col=1
        )
        fig_pred.add_vline(x=t_corte, line_width=1.8, line_dash="dash", line_color="#1d4ed8", row=1, col=1)

        categorias_esc = ["Aglomerado Estelar", "Galaxia Ana", "CHUTE PREVISTO (IA)", "Galaxia Media", "Via Lactea"]
        valores_esc = [50.0, 150.0, v_kick_pred_kms, 300.0, 550.0]
        cores_esc = ["#cbd5e1", "#94a3b8", "#2563eb", "#64748b", "#081226"]

        fig_pred.add_trace(
            go.Bar(
                x=categorias_esc,
                y=valores_esc,
                marker_color=cores_esc,
                text=[f"{v:.0f} km/s" for v in valores_esc],
                textposition="outside",
                showlegend=False
            ),
            row=1, col=2
        )

        fig_pred.update_layout(
            height=450,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            margin=dict(l=45, r=25, t=65, b=40)
        )
        fig_pred.update_xaxes(title_text="Tempo Fisico (Corte Pre-Merger em t = -50M)", row=1, col=1, gridcolor="#e2e8f0")
        fig_pred.update_yaxes(title_text="Strain Re(h_22)", row=1, col=1, gridcolor="#e2e8f0")
        fig_pred.update_yaxes(title_text="Velocidade (km/s)", range=[0, 650], row=1, col=2, gridcolor="#e2e8f0")

        st.plotly_chart(fig_pred, use_container_width=True)


# =====================================================================
# PAGINA 2: VALIDACAO CIENTIFICA HOLD-OUT (BENCHMARK 14 ONDAS SXS — 99.98% OVERLAP)
# =====================================================================
elif st.session_state["pagina_ativa"] == "validacao":
    res_lista = env["bench_polar"]["resultados_teste"]
    col_ctrl, col_main = st.columns([1.05, 2.95], gap="large")

    with col_ctrl:
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Validacao Hold-Out Cega (14 Simulacoes SXS)</div>
            <div class="card-sub">
                Modelo treinado em 11 simulacoes cobrindo q in [1.0, 8.0] e avaliado em 3 simulacoes 100% ineditas (Teste Cego).
            </div>
        </div>
        """, unsafe_allow_html=True)

        opcoes_bench = [
            f"{r['id']} — q = {r['q']:.2f} (Overlap: {r['ov_dir']*100:.2f}%)"
            for r in res_lista
        ]
        escolha_b = st.selectbox("Simulacao Secreta de Teste Cego:", opcoes_bench, index=0)
        r_sel = res_lista[opcoes_bench.index(escolha_b)]

        arq_comparada = st.radio(
            "Arquitetura Neural Exibida:",
            [
                "CosmicFNO Espectral Direto (Melhor)",
                "Polar-CosmicFNO (Amplitude-Fase)",
                "Ambas Sobrepostas"
            ],
            index=0
        )

        janela_zoom = st.slider(
            "Janela de Tempo Fisico (M):",
            min_value=-330.0,
            max_value=90.0,
            value=(-280.0, 80.0),
            step=5.0
        )

        st.markdown(f"""
        <div class="white-card" style="margin-top: 14px; border-left: 4px solid #081226;">
            <div class="card-title" style="font-size:0.9rem;">Comparativo no Teste Cego ({r_sel['id']})</div>
            <div class="card-sub" style="line-height: 1.75;">
                • <b>Overlap FNO Direto:</b> {r_sel['ov_dir']*100:.2f}%<br>
                • <b>Mismatch FNO Direto:</b> {r_sel['mis_dir']:.5f}<br>
                • <b>Overlap Polar-FNO:</b> {r_sel['ov_pol']*100:.2f}%<br>
                • <b>Erro Relativo L2:</b> {r_sel['l2_dir']*100:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_main:
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
            vertical_spacing=0.07,
            subplot_titles=(
                f"Teste Cego Hold-Out: {r_sel['id']} (q = {r_sel['q']:.2f} — Simulacao Nunca Vista no Treino)",
                "Erro Absoluto Instantaneo |h_SXS(t) - h_FNO(t)|"
            )
        )

        fig_val.add_trace(
            go.Scatter(
                x=t_zoom, y=ref_zoom,
                mode="lines",
                name="Onda de Referencia SXS (Relatividade Numerica)",
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
            height=540,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a"),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
            margin=dict(l=55, r=30, t=60, b=45)
        )
        fig_val.update_yaxes(title_text="Strain Re(h_22)", row=1, col=1, gridcolor="#e2e8f0")
        fig_val.update_yaxes(title_text="|Erro|", row=2, col=1, gridcolor="#e2e8f0")
        fig_val.update_xaxes(title_text="Tempo Fisico Alinhado pelo Merger (t = 0)", row=2, col=1, gridcolor="#e2e8f0")

        st.plotly_chart(fig_val, use_container_width=True)


# =====================================================================
# PAGINA 3: EMULADOR CONTINUO E ESPECTRO DE FOURIER (FFT)
# =====================================================================
elif st.session_state["pagina_ativa"] == "espectro":
    c_left, c_right = st.columns([1, 3], gap="large")

    with c_left:
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Condicionamento Fisico</div>
            <div class="card-sub">Explore continuamente a geracao da forma de onda variando a razao de massa e o spin.</div>
        </div>
        """, unsafe_allow_html=True)

        q_slider = st.slider("Razao de Massa (q = m1/m2)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
        spin_slider = st.slider("Spin Orbital (chi_1z)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

    with c_right:
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

        fig_exp = make_subplots(
            rows=1, cols=2,
            column_widths=[0.62, 0.38],
            subplot_titles=(
                f"Forma de Onda Emulada no Tempo (q={q_slider:.1f}, chi_1z={spin_slider:+.2f})",
                "Espectro de Fourier (32 Modos Ativos no FNO)"
            )
        )

        fig_exp.add_trace(
            go.Scatter(
                x=t_arr, y=onda_emulada,
                mode="lines",
                name="Strain h_22(t)",
                line=dict(color="#1d4ed8", width=2.4)
            ),
            row=1, col=1
        )
        fig_exp.add_trace(
            go.Bar(
                x=modos_idx[:64],
                y=espectro_amp[:64],
                name="|FFT(h_22)|",
                marker_color=["#1d4ed8" if k < 32 else "#cbd5e1" for k in range(64)]
            ),
            row=1, col=2
        )

        fig_exp.update_layout(
            height=500,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a"),
            showlegend=False,
            margin=dict(l=50, r=30, t=55, b=45)
        )
        fig_exp.update_xaxes(title_text="Tempo Fisico (M)", row=1, col=1, gridcolor="#e2e8f0")
        fig_exp.update_yaxes(title_text="Strain Re(h_22)", row=1, col=1, gridcolor="#e2e8f0")
        fig_exp.update_xaxes(title_text="Modo de Frequencia k", row=1, col=2, gridcolor="#e2e8f0")
        fig_exp.update_yaxes(title_text="Amplitude |FFT|", row=1, col=2, gridcolor="#e2e8f0")

        st.plotly_chart(fig_exp, use_container_width=True)


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
        vertical_spacing=0.06,
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
        height=660,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="Montserrat, sans-serif", color="#0f172a"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Reproduzir Colisao", method="animate", args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
                dict(label="Pausar", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            x=0.5, y=1.08, xanchor="center", yanchor="top"
        )],
        margin=dict(l=30, r=30, t=55, b=30)
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

    st.table({
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
    })
