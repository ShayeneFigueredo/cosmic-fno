"""
CosmicFNO - Plataforma Visual e Observatorio SciML (PT / EN / ES)
Autora: Shayene Figueredo (BSI/UFU)

Descricao:
    Portal cientifico responsivo (Desktop e Mobile) em formato de site:
    - Logo oficial CosmicFNO solta (sem fundo/borda branca) no menu lateral (#081226)
    - Cartoes de metricas em HTML com alto contraste (imunes ao modo escuro do navegador)
    - Graficos Plotly com theme=None e fontes escuras nitidas
    - Seletor de idioma instantaneo: Portugues (PT), English (EN) e Espanol (ES)
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
    page_title="CosmicFNO | Gravitational Wave Portal",
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
# 1. DICIONARIO MULTILINGUE COMPLETO (PT / EN / ES)
# =====================================================================
I18N = {
    "PT": {
        "sec_main": "MODULOS PRINCIPAIS",
        "sec_vis": "EXPLORACAO VISUAL",
        "sec_lang": "IDIOMA / LANGUAGE",
        "sec_info": "PESQUISA",
        "nav_previsao": "Previsao Pre-Merger e Kick",
        "nav_validacao": "Validacao Hold-Out (99.9%)",
        "nav_espectro": "Emulador e Espectro FFT",
        "nav_colisao3d": "Colisao 3D Sincronizada",
        "nav_tabela": "Tabela de Metricas",
        "info_caption": "3.435 Simulacoes SXS BBH\n14 Formas de Onda Alinhadas\nUniversidade Federal de Uberlandia (UFU)",
        "pill_catalog": "Catalogo SXS v3.0.0 · 3.435 Colisoes",
        "pill_fid": "Fidelidade > 99.9%",
        "pages": {
            "previsao": {
                "eyebrow": "OBSERVATORIO ESPECTRAL DE ONDAS GRAVITACIONAIS",
                "titulo": "Previsão Antecipada da Colisão (Pre-Merger) e Diagnóstico de Fuga Galáctica",
                "subtitulo": "O operador neural observa apenas a fase inicial da espiral (antes de t = -50M, com o impacto oculto) para prever a forma de onda da fusão e diagnosticar a massa final, o spin e a velocidade de ejeção (Recoil Kick) do buraco negro remanescente."
            },
            "validacao": {
                "eyebrow": "BENCHMARK CIENTIFICO DE RELATIVIDADE NUMERICA",
                "titulo": "Validação Hold-Out Cega em Formas de Onda Inéditas (99.98% de Fidelidade)",
                "subtitulo": "Avaliação rigorosa do modelo em simulações SXS nunca vistas durante o treinamento, comparando a parametrização espectral direta (CosmicFNO Direto) com a decomposição polar explícita (Amplitude-Fase)."
            },
            "espectro": {
                "eyebrow": "LABORATORIO DE INFERENCIA EM TEMPO REAL",
                "titulo": "Emulador Contínuo de Espaço de Parâmetros e Espectro de Fourier (FFT)",
                "subtitulo": "Explore continuamente a razão de massa (q) e o spin orbital para inspecionar a onda gravitacional gerada em milissegundos e os 32 modos espectrais de Fourier filtrados pelo operador neural."
            },
            "colisao3d": {
                "eyebrow": "VISUALIZACAO DINAMICA DE ESPACO-TEMPO",
                "titulo": "Coalescência Binária 3D Sincronizada com o Sinal Gravitacional",
                "subtitulo": "Simulação tridimensional das órbitas relativísticas dos buracos negros sincronizada passo a passo com a emissão do modo dominante quadrupolar h_22(t) até o instante da fusão (t = 0)."
            },
            "tabela": {
                "eyebrow": "RELATORIO CONSOLIDADO DE METRICAS",
                "titulo": "Síntese Quantitativa dos Experimentos — Catálogo SXS v3.0.0",
                "subtitulo": "Comparativo completo de Erro Relativo L2, Fidelidade de Fase (Overlap) e Mismatch em todas as etapas evolutivas da pesquisa."
            },
        },
        # Pagina 1
        "lbl_evento": "Evento Binário Monitorado:",
        "lbl_massa": "Massa Total Inicial (Massas Solares):",
        "cfg_pre": "Configuração Pré-Impacto",
        "cfg_prec": "Precisão SXS-3435",
        "kpi1_title": "Fidelidade Pre-Merger",
        "kpi1_sub": "Futuro Oculto (t > -50M)",
        "kpi2_title": "Novo Buraco Negro (M_f)",
        "kpi2_sub": "M_sol irradiados",
        "kpi3_title": "Spin Final Previsto",
        "kpi3_sub": "SXS Real",
        "kpi4_title": "Chute Relativístico (Kick)",
        "kpi4_sub": "SXS Real",
        "v_header": "DIAGNÓSTICO DE RECUO RELATIVÍSTICO (VALIDADO EM {n} COLISÕES SXS — {acc:.2f}% ACURÁCIA)",
        "v_high_t": "VEREDITO ASTROFÍSICO: BURACO NEGRO FUGITIVO INTERGALÁCTICO",
        "v_high_d": "O chute gravitacional previsto ({v:.1f} km/s = {vh:,.0f} km/h) ultrapassa a velocidade de escape de Aglomerados Globulares (50 km/s), Galáxias Anãs (150 km/s) e Galáxias Médias (300 km/s). O buraco negro recém-nascido será ejetado do seu núcleo!",
        "v_mid_t": "VEREDITO ASTROFÍSICO: EJEÇÃO DE AGLOMERADOS E GALÁXIAS ANÃS",
        "v_mid_d": "O chute gravitacional previsto ({v:.1f} km/s = {vh:,.0f} km/h) rompe a gravidade de Aglomerados Estelares Globulares (50 km/s) e Galáxias Satélites Anãs, mas permanece retido no halo de galáxias massivas como a Via Láctea (550 km/s).",
        "v_low_t": "VEREDITO ASTROFÍSICO: EJEÇÃO DE AGLOMERADO ESTELAR (RETIDO NA GALÁXIA)",
        "v_low_d": "O chute gravitacional previsto ({v:.1f} km/s) supera a velocidade de escape de Aglomerados Globulares (50 km/s), mas o buraco negro permanecerá preso ao potencial gravitacional da galáxia hospedeira.",
        "p1_chart1_title": "Previsão Antecipada da Colisão (Pre-Merger em t = -50M)",
        "p1_tr_obs": "Espiral Observada (Entrada)",
        "p1_tr_real": "Impacto Real SXS (Oculto)",
        "p1_tr_pred": "Previsão Futura CosmicFNO",
        "p1_x_time": "Tempo Físico Alinhado (M)",
        "p1_chart2_title": "Chute Relativístico vs Velocidade de Escape",
        "p1_bars": ["Aglomerado", "Galáxia Anã", "CHUTE IA", "Gal. Média", "Via Láctea"],
        "p1_y_vel": "Velocidade (km/s)",
        # Pagina 2
        "p2_sim_lbl": "Simulação Secreta de Teste Cego:",
        "p2_arq_lbl": "Arquitetura Neural Exibida:",
        "p2_arq_opts": ["CosmicFNO Espectral Direto (Melhor)", "Polar-CosmicFNO (Amplitude-Fase)", "Ambas Sobrepostas"],
        "p2_zoom_lbl": "Janela de Tempo Físico (M):",
        "p2_k1_t": "Overlap (Teste Cego)",
        "p2_k2_t": "Mismatch (1 - Overlap)",
        "p2_k2_s": "Meta LIGO < 0.00100",
        "p2_k3_t": "Erro Relativo L2",
        "p2_k4_t": "Amostras de Treino",
        "p2_k4_v": "11 Simulações SXS",
        "p2_k4_s": "Grade q in [1.0, 8.0]",
        "p2_sub1": "Teste Cego Hold-Out: {id} (q = {q:.2f} — Simulação Inédita)",
        "p2_sub2": "Erro Absoluto Instantâneo |h_SXS(t) - h_FNO(t)|",
        "p2_ref": "Onda de Referência SXS",
        # Pagina 3
        "p3_q": "Razão de Massa (q = m1/m2)",
        "p3_spin": "Spin Orbital (chi_1z)",
        "p3_c1": "Forma de Onda Emulada no Tempo (q={q:.1f}, chi_1z={s:+.2f})",
        "p3_c2": "Espectro de Fourier (32 Modos Ativos no FNO)",
        "p3_xk": "Modo de Frequência k",
        "p3_yf": "Amplitude |FFT|",
        # Pagina 4
        "p4_slider": "Selecione a Razão de Massa (q) da Simulação SXS:",
        "p4_sub1": "Coalescência Binária 3D (q = {q:.2f})",
        "p4_sub2": "Formação Sincronizada da Onda Gravitacional Re(h_22)",
        "p4_play": "Reproduzir Colisão",
        "p4_pause": "Pausar",
        "p4_bh1": "Buraco Negro 1 (m1)",
        "p4_bh2": "Buraco Negro 2 (m2)",
        # Pagina 5
        "p5_k1_t": "Colisões SXS Analisadas",
        "p5_k1_s": "Teste Cego: {n} colisões",
        "p5_k2_t": "Precisão Massa Final (M_f)",
        "p5_k2_s": "Erro Relativo: {e:.2f}%",
        "p5_k3_t": "Correlação Spin Final (R²)",
        "p5_k4_t": "Fuga de Aglomerados",
        "p5_k4_s": "Classificação > 50 km/s",
        "footer_l": "<b>CosmicFNO</b> · Observatório Espectral de Ondas Gravitacionais (Fourier Neural Operators)",
        "footer_r": "Universidade Federal de Uberlândia (UFU) · Catálogo SXS v3.0.0",
    },
    "EN": {
        "sec_main": "CORE MODULES",
        "sec_vis": "VISUAL EXPLORATION",
        "sec_lang": "LANGUAGE / IDIOMA",
        "sec_info": "RESEARCH",
        "nav_previsao": "Pre-Merger & Recoil Kick",
        "nav_validacao": "Hold-Out Validation (99.9%)",
        "nav_espectro": "Emulator & FFT Spectrum",
        "nav_colisao3d": "Synchronized 3D Merger",
        "nav_tabela": "Metrics Benchmark Table",
        "info_caption": "3,435 SXS BBH Simulations\n14 Phase-Aligned Waveforms\nFederal University of Uberlandia (UFU)",
        "pill_catalog": "SXS Catalog v3.0.0 · 3,435 Mergers",
        "pill_fid": "Fidelity > 99.9%",
        "pages": {
            "previsao": {
                "eyebrow": "SPECTRAL GRAVITATIONAL-WAVE OBSERVATORY",
                "titulo": "Pre-Merger Early-Warning Forecasting & Galactic Escape Diagnostics",
                "subtitulo": "The neural operator observes only the early inspiral phase (before t = -50M, with the merger hidden) to forecast the full collision waveform and estimate the final mass, spin, and relativistic recoil kick of the remnant black hole."
            },
            "validacao": {
                "eyebrow": "NUMERICAL RELATIVITY SCIENTIFIC BENCHMARK",
                "titulo": "Blind Hold-Out Validation on Unseen Waveforms (99.98% Overlap)",
                "subtitulo": "Rigorous evaluation on SXS simulations never seen during training, comparing direct spectral parameterization (Direct CosmicFNO) against explicit polar amplitude-phase decomposition."
            },
            "espectro": {
                "eyebrow": "REAL-TIME INFERENCE LABORATORY",
                "titulo": "Continuous Parameter-Space Emulator & Fourier Spectrum (FFT)",
                "subtitulo": "Continuously vary the mass ratio (q) and orbital spin to inspect the millisecond-generated gravitational waveform and the 32 Fourier spectral modes filtered by the neural operator."
            },
            "colisao3d": {
                "eyebrow": "DYNAMIC SPACETIME VISUALIZATION",
                "titulo": "3D Binary Coalescence Synchronized with Gravitational Strain",
                "subtitulo": "Three-dimensional relativistic orbital evolution synchronized step-by-step with the dominant quadrupole wave emission h_22(t) up to the merger instant (t = 0)."
            },
            "tabela": {
                "eyebrow": "CONSOLIDATED METRICS REPORT",
                "titulo": "Quantitative Experimental Summary — SXS Catalog v3.0.0",
                "subtitulo": "Complete comparison of Relative L2 Error, Phase Fidelity (Overlap), and Mismatch across all research milestones."
            },
        },
        "lbl_evento": "Monitored Binary Event:",
        "lbl_massa": "Initial Total Mass (Solar Masses):",
        "cfg_pre": "Pre-Merger Configuration",
        "cfg_prec": "SXS-3435 Accuracy",
        "kpi1_title": "Pre-Merger Fidelity",
        "kpi1_sub": "Hidden Future (t > -50M)",
        "kpi2_title": "Remnant Black Hole (M_f)",
        "kpi2_sub": "M_sol radiated",
        "kpi3_title": "Predicted Final Spin",
        "kpi3_sub": "True SXS",
        "kpi4_title": "Relativistic Recoil Kick",
        "kpi4_sub": "True SXS",
        "v_header": "RELATIVISTIC RECOIL DIAGNOSTICS (VALIDATED ON {n} SXS MERGERS — {acc:.2f}% ACCURACY)",
        "v_high_t": "ASTROPHYSICAL VERDICT: INTERGALACTIC RUNAWAY BLACK HOLE",
        "v_high_d": "The predicted gravitational recoil kick ({v:.1f} km/s = {vh:,.0f} km/h) exceeds the escape velocity of Globular Clusters (50 km/s), Dwarf Galaxies (150 km/s), and Medium Galaxies (300 km/s). The newborn black hole will be ejected from its host core!",
        "v_mid_t": "ASTROPHYSICAL VERDICT: CLUSTER & DWARF GALAXY EJECTION",
        "v_mid_d": "The predicted gravitational recoil kick ({v:.1f} km/s = {vh:,.0f} km/h) breaks free from Globular Star Clusters (50 km/s) and Dwarf Satellite Galaxies, while remaining bound within massive halos like the Milky Way (550 km/s).",
        "v_low_t": "ASTROPHYSICAL VERDICT: STAR CLUSTER EJECTION (GALAXY RETAINED)",
        "v_low_d": "The predicted gravitational recoil kick ({v:.1f} km/s) exceeds the escape velocity of Globular Clusters (50 km/s), but the remnant black hole remains gravitationally bound to its host galaxy.",
        "p1_chart1_title": "Early-Warning Collision Forecast (Pre-Merger at t = -50M)",
        "p1_tr_obs": "Observed Inspiral (Input)",
        "p1_tr_real": "True SXS Merger (Hidden)",
        "p1_tr_pred": "CosmicFNO Future Forecast",
        "p1_x_time": "Merger-Aligned Physical Time (M)",
        "p1_chart2_title": "Relativistic Kick vs Escape Velocity",
        "p1_bars": ["Glob. Cluster", "Dwarf Gal.", "AI KICK", "Medium Gal.", "Milky Way"],
        "p1_y_vel": "Velocity (km/s)",
        "p2_sim_lbl": "Blind Hold-Out Test Simulation:",
        "p2_arq_lbl": "Displayed Neural Architecture:",
        "p2_arq_opts": ["Direct Spectral CosmicFNO (Best)", "Polar-CosmicFNO (Amplitude-Phase)", "Both Overlaid"],
        "p2_zoom_lbl": "Physical Time Window (M):",
        "p2_k1_t": "Overlap (Blind Test)",
        "p2_k2_t": "Mismatch (1 - Overlap)",
        "p2_k2_s": "LIGO Target < 0.00100",
        "p2_k3_t": "Relative L2 Error",
        "p2_k4_t": "Training Set",
        "p2_k4_v": "11 SXS Simulations",
        "p2_k4_s": "Grid q in [1.0, 8.0]",
        "p2_sub1": "Blind Hold-Out Test: {id} (q = {q:.2f} — Unseen Simulation)",
        "p2_sub2": "Instantaneous Absolute Error |h_SXS(t) - h_FNO(t)|",
        "p2_ref": "SXS Reference Waveform",
        "p3_q": "Mass Ratio (q = m1/m2)",
        "p3_spin": "Orbital Spin (chi_1z)",
        "p3_c1": "Time-Domain Emulated Waveform (q={q:.1f}, chi_1z={s:+.2f})",
        "p3_c2": "Fourier Spectrum (32 Active FNO Modes)",
        "p3_xk": "Frequency Mode k",
        "p3_yf": "Amplitude |FFT|",
        "p4_slider": "Select SXS Simulation Mass Ratio (q):",
        "p4_sub1": "3D Binary Coalescence (q = {q:.2f})",
        "p4_sub2": "Synchronized Gravitational Strain Re(h_22)",
        "p4_play": "Play Collision",
        "p4_pause": "Pause",
        "p4_bh1": "Black Hole 1 (m1)",
        "p4_bh2": "Black Hole 2 (m2)",
        "p5_k1_t": "Analyzed SXS Mergers",
        "p5_k1_s": "Blind Test: {n} mergers",
        "p5_k2_t": "Final Mass Accuracy (M_f)",
        "p5_k2_s": "Relative Error: {e:.2f}%",
        "p5_k3_t": "Final Spin Correlation (R²)",
        "p5_k4_t": "Cluster Escape Accuracy",
        "p5_k4_s": "Classification > 50 km/s",
        "footer_l": "<b>CosmicFNO</b> · Spectral Gravitational-Wave Observatory (Fourier Neural Operators)",
        "footer_r": "Federal University of Uberlândia (UFU) · SXS Catalog v3.0.0",
    },
    "ES": {
        "sec_main": "MODULOS PRINCIPALES",
        "sec_vis": "EXPLORACION VISUAL",
        "sec_lang": "IDIOMA / LANGUAGE",
        "sec_info": "INVESTIGACION",
        "nav_previsao": "Prediccion Pre-Merger y Kick",
        "nav_validacao": "Validacion Hold-Out (99.9%)",
        "nav_espectro": "Emulador y Espectro FFT",
        "nav_colisao3d": "Colision 3D Sincronizada",
        "nav_tabela": "Tabla de Metricas",
        "info_caption": "3.435 Simulaciones SXS BBH\n14 Formas de Onda Alineadas\nUniversidad Federal de Uberlandia (UFU)",
        "pill_catalog": "Catálogo SXS v3.0.0 · 3.435 Colisiones",
        "pill_fid": "Fidelidad > 99.9%",
        "pages": {
            "previsao": {
                "eyebrow": "OBSERVATORIO ESPECTRAL DE ONDAS GRAVITACIONALES",
                "titulo": "Predicción Anticipada de la Colisión (Pre-Merger) y Diagnóstico de Escape Galáctico",
                "subtitulo": "El operador neuronal observa únicamente la fase inicial de la espiral (antes de t = -50M, con el impacto oculto) para predecir la onda completa de la fusión y estimar la masa final, el espín y la velocidad de eyección (Recoil Kick) del agujero negro remanente."
            },
            "validacao": {
                "eyebrow": "BENCHMARK CIENTIFICO DE RELATIVIDAD NUMERICA",
                "titulo": "Validación Hold-Out Ciega en Formas de Onda Inéditas (99.98% de Fidelidad)",
                "subtitulo": "Evaluación rigurosa en simulaciones SXS nunca vistas durante el entrenamiento, comparando la parametrización espectral directa (CosmicFNO Directo) con la descomposición polar explícita (Amplitud-Fase)."
            },
            "espectro": {
                "eyebrow": "LABORATORIO DE INFERENCIA EN TIEMPO REAL",
                "titulo": "Emulador Continuo del Espacio de Parámetros y Espectro de Fourier (FFT)",
                "subtitulo": "Explore continuamente la razón de masa (q) y el espín orbital para inspeccionar la onda gravitacional generada en milisegundos y los 32 modos espectrales de Fourier filtrados por el operador neuronal."
            },
            "colisao3d": {
                "eyebrow": "VISUALIZACION DINAMICA DEL ESPACIO-TIEMPO",
                "titulo": "Coalescencia Binaria 3D Sincronizada con la Señal Gravitacional",
                "subtitulo": "Simulación tridimensional de las órbitas relativistas de los agujeros negros sincronizada paso a paso con la emisión del modo dominante cuadrupolar h_22(t) hasta el instante de la fusión (t = 0)."
            },
            "tabela": {
                "eyebrow": "INFORME CONSOLIDADO DE METRICAS",
                "titulo": "Síntesis Cuantitativa de los Experimentos — Catálogo SXS v3.0.0",
                "subtitulo": "Comparativa completa de Error Relativo L2, Fidelidad de Fase (Overlap) y Mismatch en todas las etapas evolutivas de la investigación."
            },
        },
        "lbl_evento": "Evento Binario Monitoreado:",
        "lbl_massa": "Masa Total Inicial (Masas Solares):",
        "cfg_pre": "Configuración Pre-Impacto",
        "cfg_prec": "Precisión SXS-3435",
        "kpi1_title": "Fidelidad Pre-Merger",
        "kpi1_sub": "Futuro Oculto (t > -50M)",
        "kpi2_title": "Nuevo Agujero Negro (M_f)",
        "kpi2_sub": "M_sol irradiados",
        "kpi3_title": "Espín Final Previsto",
        "kpi3_sub": "SXS Real",
        "kpi4_title": "Patada Relativista (Kick)",
        "kpi4_sub": "SXS Real",
        "v_header": "DIAGNÓSTICO DE RETROCESO RELATIVISTA (VALIDADO EN {n} COLISIONES SXS — {acc:.2f}% PRECISIÓN)",
        "v_high_t": "VEREDICTO ASTROFÍSICO: AGUJERO NEGRO FUGITIVO INTERGALÁCTICO",
        "v_high_d": "La patada gravitacional prevista ({v:.1f} km/s = {vh:,.0f} km/h) supera la velocidad de escape de Cúmulos Globulares (50 km/s), Galaxias Enanas (150 km/s) y Galaxias Medianas (300 km/s). ¡El agujero negro recién formado será eyectado de su núcleo!",
        "v_mid_t": "VEREDICTO ASTROFÍSICO: EYECCIÓN DE CÚMULOS Y GALAXIAS ENANAS",
        "v_mid_d": "La patada gravitacional prevista ({v:.1f} km/s = {vh:,.0f} km/h) vence la gravedad de Cúmulos Estelares Globulares (50 km/s) y Galaxias Satélites Enanas, pero permanece retenida en el halo de galaxias masivas como la Vía Láctea (550 km/s).",
        "v_low_t": "VEREDICTO ASTROFÍSICO: EYECCIÓN DE CÚMULO ESTELAR (RETENIDO EN LA GALAXIA)",
        "v_low_d": "La patada gravitacional prevista ({v:.1f} km/s) supera la velocidad de escape de Cúmulos Globulares (50 km/s), pero el agujero negro permanecerá ligado al potencial gravitacional de su galaxia anfitriona.",
        "p1_chart1_title": "Predicción Anticipada de la Colisión (Pre-Merger en t = -50M)",
        "p1_tr_obs": "Espiral Observada (Entrada)",
        "p1_tr_real": "Impacto Real SXS (Oculto)",
        "p1_tr_pred": "Predicción Futura CosmicFNO",
        "p1_x_time": "Tiempo Físico Alineado (M)",
        "p1_chart2_title": "Patada Relativista vs Velocidad de Escape",
        "p1_bars": ["Cúmulo Glob.", "Gal. Enana", "PATADA IA", "Gal. Media", "Vía Láctea"],
        "p1_y_vel": "Velocidad (km/s)",
        "p2_sim_lbl": "Simulación Secreta de Prueba Ciega:",
        "p2_arq_lbl": "Arquitectura Neuronal Mostrada:",
        "p2_arq_opts": ["CosmicFNO Espectral Directo (Mejor)", "Polar-CosmicFNO (Amplitud-Fase)", "Ambas Superpuestas"],
        "p2_zoom_lbl": "Ventana de Tiempo Físico (M):",
        "p2_k1_t": "Overlap (Prueba Ciega)",
        "p2_k2_t": "Mismatch (1 - Overlap)",
        "p2_k2_s": "Meta LIGO < 0.00100",
        "p2_k3_t": "Error Relativo L2",
        "p2_k4_t": "Muestras de Entrenamiento",
        "p2_k4_v": "11 Simulaciones SXS",
        "p2_k4_s": "Malla q in [1.0, 8.0]",
        "p2_sub1": "Prueba Ciega Hold-Out: {id} (q = {q:.2f} — Simulación Inédita)",
        "p2_sub2": "Error Absoluto Instantáneo |h_SXS(t) - h_FNO(t)|",
        "p2_ref": "Onda de Referencia SXS",
        "p3_q": "Razón de Masa (q = m1/m2)",
        "p3_spin": "Espín Orbital (chi_1z)",
        "p3_c1": "Forma de Onda Emulada en el Tiempo (q={q:.1f}, chi_1z={s:+.2f})",
        "p3_c2": "Espectro de Fourier (32 Modos Activos en el FNO)",
        "p3_xk": "Modo de Frecuencia k",
        "p3_yf": "Amplitud |FFT|",
        "p4_slider": "Seleccione la Razón de Masa (q) de la Simulación SXS:",
        "p4_sub1": "Coalescencia Binaria 3D (q = {q:.2f})",
        "p4_sub2": "Formación Sincronizada de la Onda Gravitacional Re(h_22)",
        "p4_play": "Reproducir Colisión",
        "p4_pause": "Pausar",
        "p4_bh1": "Agujero Negro 1 (m1)",
        "p4_bh2": "Agujero Negro 2 (m2)",
        "p5_k1_t": "Colisiones SXS Analizadas",
        "p5_k1_s": "Prueba Ciega: {n} colisiones",
        "p5_k2_t": "Precisión Masa Final (M_f)",
        "p5_k2_s": "Error Relativo: {e:.2f}%",
        "p5_k3_t": "Correlación Espín Final (R²)",
        "p5_k4_t": "Escape de Cúmulos",
        "p5_k4_s": "Clasificación > 50 km/s",
        "footer_l": "<b>CosmicFNO</b> · Observatorio Espectral de Ondas Gravitacionales (Fourier Neural Operators)",
        "footer_r": "Universidad Federal de Uberlândia (UFU) · Catálogo SXS v3.0.0",
    },
}


# =====================================================================
# 2. CSS CUSTOMIZADO E RESPONSIVO (ALTO CONTRASTE + LOGO SOLTA)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, p, div, span, label, h1, h2, h3, h4, h5, h6, button, input, select, textarea, table, th, td {
        font-family: 'Montserrat', sans-serif !important;
    }

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

    /* Forca cor escura visivel nos rotulos de controles na area principal mesmo se o navegador estiver em Dark Mode */
    section.main [data-testid="stWidgetLabel"] p,
    section.main [data-testid="stWidgetLabel"] label,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
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

    /* Logo Oficial SOLTA na Sidebar (sem caixa/borda branca) */
    .sidebar-logo-free {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 2px 18px 2px;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sidebar-logo-free img {
        width: 100%;
        max-width: 215px;
        height: auto;
        display: block;
        filter: brightness(1.35) contrast(1.08) drop-shadow(0 2px 10px rgba(147, 197, 253, 0.18));
    }

    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #64748b !important;
        text-transform: uppercase;
        margin: 16px 0 8px 6px;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
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

    /* CABECALHO DO SITE (HERO BAR INSTITUCIONAL) */
    .site-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1d4ed8;
        border-radius: 14px;
        padding: 20px 26px;
        margin-bottom: 18px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }
    .site-header-left {
        max-width: 780px;
    }
    .site-eyebrow {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: #2563eb !important;
        margin: 0 0 5px 0;
    }
    .site-title {
        font-size: 1.30rem;
        font-weight: 800;
        color: #081226 !important;
        margin: 0 0 6px 0;
        line-height: 1.25;
    }
    .site-subtitle {
        font-size: 0.85rem;
        color: #475569 !important;
        margin: 0;
        line-height: 1.55;
    }
    .site-header-badges {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .status-pill {
        background-color: #eff6ff;
        color: #1d4ed8 !important;
        border: 1px solid #bfdbfe;
        padding: 6px 13px;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .status-pill-dark {
        background-color: #081226;
        color: #f8fafc !important;
        padding: 6px 13px;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* CARTOES DE METRICAS CUSTOMIZADOS (ALTO CONTRASTE GARANTIDO NO FUNDO BRANCO) */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 5px solid #1d4ed8 !important;
        border-radius: 12px;
        padding: 15px 18px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin: 0 0 6px 0;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #081226 !important;
        margin: 0 0 8px 0;
        line-height: 1.15;
    }
    .kpi-badge {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8 !important;
        border: 1px solid #bfdbfe;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
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
        color: #081226 !important;
        margin-bottom: 6px;
    }
    .card-sub {
        font-size: 0.84rem;
        color: #334155 !important;
        margin-bottom: 0;
        line-height: 1.55;
    }

    .verdict-banner {
        background: linear-gradient(90deg, #081226 0%, #1e3a8a 100%);
        color: #ffffff !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 6px;
        margin-bottom: 16px;
        border-left: 5px solid #3b82f6;
    }
    .verdict-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #93c5fd !important;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .verdict-main {
        font-size: 0.96rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 3px;
    }
    .verdict-desc {
        font-size: 0.82rem;
        color: #dbeafe !important;
        margin: 0;
        line-height: 1.5;
    }

    .site-footer {
        margin-top: 28px;
        padding: 16px 20px;
        border-top: 1px solid #cbd5e1;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 0.76rem;
        color: #475569 !important;
    }

    /* RESPONSIVIDADE MOBILE (Celulares e Tablets) */
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
        .site-header {
            padding: 15px 16px;
            margin-bottom: 14px;
        }
        .site-title {
            font-size: 1.06rem;
        }
        .site-subtitle {
            font-size: 0.80rem;
        }
        .status-pill, .status-pill-dark {
            font-size: 0.68rem;
            padding: 4px 10px;
        }
        .kpi-value {
            font-size: 1.35rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def render_kpi_grid(cards):
    """Renderiza 4 cartoes de metricas em HTML com alto contraste garantido no fundo branco."""
    blocos = []
    for c in cards:
        blocos.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{c["label"]}</div>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<span class="kpi-badge">{c["sub"]}</span>'
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
# 4. MENU LATERAL COM LOGO SOLTA E SELETOR DE IDIOMAS (PT / EN / ES)
# =====================================================================
if "pagina_ativa" not in st.session_state:
    st.session_state["pagina_ativa"] = "previsao"
if "idioma" not in st.session_state:
    st.session_state["idioma"] = "PT"

tr = I18N[st.session_state["idioma"]]

with st.sidebar:
    if logo_b64:
        st.markdown(
            f'<div class="sidebar-logo-free"><img src="data:image/png;base64,{logo_b64}" alt="CosmicFNO Logo"></div>',
            unsafe_allow_html=True
        )

    st.markdown(f'<div class="sidebar-section-label">{tr["sec_lang"]}</div>', unsafe_allow_html=True)
    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        if st.button("PT", type="primary" if st.session_state["idioma"] == "PT" else "secondary"):
            st.session_state["idioma"] = "PT"
            st.rerun()
    with l_col2:
        if st.button("EN", type="primary" if st.session_state["idioma"] == "EN" else "secondary"):
            st.session_state["idioma"] = "EN"
            st.rerun()
    with l_col3:
        if st.button("ES", type="primary" if st.session_state["idioma"] == "ES" else "secondary"):
            st.session_state["idioma"] = "ES"
            st.rerun()

    st.markdown(f'<div class="sidebar-section-label">{tr["sec_main"]}</div>', unsafe_allow_html=True)

    if st.button(tr["nav_previsao"], type="primary" if st.session_state["pagina_ativa"] == "previsao" else "secondary"):
        st.session_state["pagina_ativa"] = "previsao"
        st.rerun()

    if st.button(tr["nav_validacao"], type="primary" if st.session_state["pagina_ativa"] == "validacao" else "secondary"):
        st.session_state["pagina_ativa"] = "validacao"
        st.rerun()

    st.markdown(f'<div class="sidebar-section-label">{tr["sec_vis"]}</div>', unsafe_allow_html=True)

    if st.button(tr["nav_espectro"], type="primary" if st.session_state["pagina_ativa"] == "espectro" else "secondary"):
        st.session_state["pagina_ativa"] = "espectro"
        st.rerun()

    if st.button(tr["nav_colisao3d"], type="primary" if st.session_state["pagina_ativa"] == "colisao3d" else "secondary"):
        st.session_state["pagina_ativa"] = "colisao3d"
        st.rerun()

    if st.button(tr["nav_tabela"], type="primary" if st.session_state["pagina_ativa"] == "tabela" else "secondary"):
        st.session_state["pagina_ativa"] = "tabela"
        st.rerun()

    st.markdown(f'<div class="sidebar-section-label">{tr["sec_info"]}</div>', unsafe_allow_html=True)
    st.caption(tr["info_caption"])


# =====================================================================
# 5. CABECALHO DO PORTAL CIENTIFICO (FORMATO DE SITE)
# =====================================================================
meta_atual = tr["pages"][st.session_state["pagina_ativa"]]

st.markdown(f"""
<div class="site-header">
    <div class="site-header-left">
        <p class="site-eyebrow">{meta_atual['eyebrow']}</p>
        <h1 class="site-title">{meta_atual['titulo']}</h1>
        <p class="site-subtitle">{meta_atual['subtitulo']}</p>
    </div>
    <div class="site-header-badges">
        <span class="status-pill">{tr['pill_catalog']}</span>
        <span class="status-pill-dark">{tr['pill_fid']}</span>
    </div>
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
            opcoes_pred = [f"{item['id']} — q = {item['q']:.2f} ({item['tipo']})" for item in CATALOGO_INFO]
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
            <div class="white-card" style="padding: 12px 16px; margin-bottom: 0; border-left: 4px solid #081226;">
                <div class="card-title" style="font-size:0.84rem; margin-bottom: 3px;">{tr['cfg_pre']}</div>
                <div class="card-sub" style="font-size:0.8rem;">
                    <b>m1:</b> {m1_sol:.1f} M_sol | <b>m2:</b> {m2_sol:.1f} M_sol | <b>{tr['cfg_prec']}:</b> {100.0 - b3300['erro_rel_massa']:.2f}%
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

    # Cartoes de Metricas em HTML com Alto Contraste
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
            "sub": f"{tr['kpi4_sub']}: {rem_real[2]:.1f} km/s",
        },
    ])

    if v_kick_pred_kms >= 300.0:
        titulo_v = tr["v_high_t"]
        desc_v = tr["v_high_d"].format(v=v_kick_pred_kms, vh=v_kick_pred_kms * 3600)
    elif v_kick_pred_kms >= 130.0:
        titulo_v = tr["v_mid_t"]
        desc_v = tr["v_mid_d"].format(v=v_kick_pred_kms, vh=v_kick_pred_kms * 3600)
    else:
        titulo_v = tr["v_low_t"]
        desc_v = tr["v_low_d"].format(v=v_kick_pred_kms)

    cabecalho_v = tr["v_header"].format(n=b3300["n_total"], acc=b3300["acc_aglomerado"])

    st.markdown(f"""
    <div class="verdict-banner">
        <div class="verdict-title">{cabecalho_v}</div>
        <div class="verdict-main">{titulo_v}</div>
        <p class="verdict-desc">{desc_v}</p>
    </div>
    """, unsafe_allow_html=True)

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
                line=dict(color="#64748b", width=2.4)
            )
        )
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_real_np[mask_fut_plot],
                mode="lines",
                name=tr["p1_tr_real"],
                line=dict(color="#081226", width=3.0)
            )
        )
        fig_onda.add_trace(
            go.Scatter(
                x=t_np[mask_fut_plot], y=h_pred_np[mask_fut_plot],
                mode="lines",
                name=tr["p1_tr_pred"],
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
            title=dict(text=tr["p1_chart1_title"], font=dict(color="#081226", size=14)),
            height=410,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#0f172a")),
            margin=dict(l=45, r=20, t=65, b=40),
            xaxis=dict(title=dict(text=tr["p1_x_time"], font=dict(color="#0f172a")), tickfont=dict(color="#334155"), gridcolor="#e2e8f0"),
            yaxis=dict(title=dict(text="Strain Re(h_22)", font=dict(color="#0f172a")), tickfont=dict(color="#334155"), gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_onda, use_container_width=True, theme=None)

    with g_col2:
        categorias_esc = tr["p1_bars"]
        valores_esc = [50.0, 150.0, v_kick_pred_kms, 300.0, 550.0]
        cores_esc = ["#cbd5e1", "#94a3b8", "#2563eb", "#64748b", "#081226"]

        fig_bar = go.Figure(
            go.Bar(
                x=categorias_esc,
                y=valores_esc,
                marker_color=cores_esc,
                text=[f"{v:.0f} km/s" for v in valores_esc],
                textposition="outside",
                textfont=dict(color="#081226", size=11)
            )
        )
        fig_bar.update_layout(
            title=dict(text=tr["p1_chart2_title"], font=dict(color="#081226", size=14)),
            height=410,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            showlegend=False,
            margin=dict(l=40, r=20, t=65, b=40),
            xaxis=dict(tickfont=dict(color="#334155"), gridcolor="#e2e8f0"),
            yaxis=dict(title=dict(text=tr["p1_y_vel"], font=dict(color="#0f172a")), tickfont=dict(color="#334155"), range=[0, 650], gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)


# =====================================================================
# PAGINA 2: VALIDACAO CIENTIFICA HOLD-OUT (BENCHMARK 14 ONDAS SXS — 99.98% OVERLAP)
# =====================================================================
elif st.session_state["pagina_ativa"] == "validacao":
    res_lista = env["bench_polar"]["resultados_teste"]

    v_col1, v_col2, v_col3 = st.columns([1.3, 1.3, 1.4])
    with v_col1:
        opcoes_bench = [f"{r['id']} — q = {r['q']:.2f} (Overlap: {r['ov_dir']*100:.2f}%)" for r in res_lista]
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
            line=dict(color="#081226", width=3.0)
        ),
        row=1, col=1
    )

    if idx_arq in (0, 2):
        fig_val.add_trace(
            go.Scatter(
                x=t_zoom, y=r_sel["pred_dir"][mask],
                mode="lines",
                name=f"CosmicFNO ({r_sel['ov_dir']*100:.2f}% Overlap)",
                line=dict(color="#2563eb", width=2.5, dash="dash")
            ),
            row=1, col=1
        )

    if idx_arq in (1, 2):
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
            name="|Delta h|",
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
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(color="#0f172a")),
        margin=dict(l=45, r=20, t=55, b=40)
    )
    fig_val.update_annotations(font=dict(color="#081226", size=13))
    fig_val.update_yaxes(title_text="Strain Re(h_22)", row=1, col=1, gridcolor="#e2e8f0", tickfont=dict(color="#334155"), title_font=dict(color="#0f172a"))
    fig_val.update_yaxes(title_text="|Error|", row=2, col=1, gridcolor="#e2e8f0", tickfont=dict(color="#334155"), title_font=dict(color="#0f172a"))
    fig_val.update_xaxes(title_text=tr["p1_x_time"], row=2, col=1, gridcolor="#e2e8f0", tickfont=dict(color="#334155"), title_font=dict(color="#0f172a"))
    fig_val.update_xaxes(tickfont=dict(color="#334155"), row=1, col=1, gridcolor="#e2e8f0")

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
                line=dict(color="#1d4ed8", width=2.4)
            )
        )
        fig_t.update_layout(
            title=dict(text=tr["p3_c1"].format(q=q_slider, s=spin_slider), font=dict(color="#081226", size=14)),
            height=430,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            margin=dict(l=45, r=20, t=55, b=40),
            xaxis=dict(title=dict(text=tr["p1_x_time"], font=dict(color="#0f172a")), tickfont=dict(color="#334155"), gridcolor="#e2e8f0"),
            yaxis=dict(title=dict(text="Strain Re(h_22)", font=dict(color="#0f172a")), tickfont=dict(color="#334155"), gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_t, use_container_width=True, theme=None)

    with p_right:
        fig_f = go.Figure(
            go.Bar(
                x=modos_idx[:64],
                y=espectro_amp[:64],
                marker_color=["#1d4ed8" if k < 32 else "#cbd5e1" for k in range(64)]
            )
        )
        fig_f.update_layout(
            title=dict(text=tr["p3_c2"], font=dict(color="#081226", size=14)),
            height=430,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Montserrat, sans-serif", color="#0f172a", size=11),
            margin=dict(l=45, r=20, t=55, b=40),
            xaxis=dict(title=dict(text=tr["p3_xk"], font=dict(color="#0f172a")), tickfont=dict(color="#334155"), gridcolor="#e2e8f0"),
            yaxis=dict(title=dict(text=tr["p3_yf"], font=dict(color="#0f172a")), tickfont=dict(color="#334155"), gridcolor="#e2e8f0")
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

    fig_3d.add_trace(go.Scatter3d(x=x1[:1], y=y1[:1], z=z1[:1], mode="markers", marker=dict(size=tam_bh1, color="#081226"), name=tr["p4_bh1"]), row=1, col=1)
    fig_3d.add_trace(go.Scatter3d(x=x2[:1], y=y2[:1], z=z2[:1], mode="markers", marker=dict(size=tam_bh2, color="#2563eb"), name=tr["p4_bh2"]), row=1, col=1)
    fig_3d.add_trace(go.Scatter(x=t_anim[:1], y=h_anim[:1], mode="lines", line=dict(color="#1d4ed8", width=2.8), name="h_22(t)"), row=2, col=1)
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
        legend=dict(font=dict(color="#0f172a")),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            font=dict(color="#081226"),
            buttons=[
                dict(label=tr["p4_play"], method="animate", args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
                dict(label=tr["p4_pause"], method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            x=0.5, y=1.09, xanchor="center", yanchor="top"
        )],
        margin=dict(l=20, r=20, t=55, b=30)
    )
    fig_3d.update_annotations(font=dict(color="#081226", size=13))
    fig_3d.update_scenes(
        xaxis=dict(range=[-2.3, 2.3], visible=False),
        yaxis=dict(range=[-2.3, 2.3], visible=False),
        zaxis=dict(range=[-1.5, 1.5], visible=False),
        bgcolor="#f8fafc",
        camera=dict(eye=dict(x=1.15, y=1.15, z=0.85))
    )
    fig_3d.update_xaxes(range=[float(t_anim.min()), float(t_anim.max())], title_text=tr["p1_x_time"], row=2, col=1, gridcolor="#e2e8f0", tickfont=dict(color="#334155"), title_font=dict(color="#0f172a"))
    fig_3d.update_yaxes(range=[float(h_anim.min()*1.25), float(h_anim.max()*1.25)], title_text="Strain Re(h_22)", row=2, col=1, gridcolor="#e2e8f0", tickfont=dict(color="#334155"), title_font=dict(color="#0f172a"))

    st.plotly_chart(fig_3d, use_container_width=True, theme=None)


# =====================================================================
# PAGINA 5: RESUMO QUANTITATIVO DOS EXPERIMENTOS
# =====================================================================
elif st.session_state["pagina_ativa"] == "tabela":
    b3300 = env["bench_3300"]

    render_kpi_grid([
        {"label": tr["p5_k1_t"], "value": f"{b3300['n_total']:,}".replace(",", "."), "sub": tr["p5_k1_s"].format(n=b3300["n_test"])},
        {"label": tr["p5_k2_t"], "value": f"{100.0 - b3300['erro_rel_massa']:.2f}%", "sub": tr["p5_k2_s"].format(e=b3300["erro_rel_massa"])},
        {"label": tr["p5_k3_t"], "value": f"{b3300['r2_spin']*100:.2f}%", "sub": f"MAE Spin: {b3300['mae_spin']:.4f} c"},
        {"label": tr["p5_k4_t"], "value": f"{b3300['acc_aglomerado']:.2f}%", "sub": tr["p5_k4_s"]},
    ])

    st.dataframe(
        {
            "Experiment / Hold-Out Scenario": [
                "Exp 1 — Baseline 1D-CNN Reconstruction (SXS:BBH:0305)",
                "Exp 2 — Initial Blind Test (4 train sims, q=5.00 unaligned phase)",
                "Exp 3 — Phase-Aligned Blind Test (4 train sims, q=5.00)",
                "Exp 4 — SXS Scale Blind Test (11 train sims -> SXS:BBH:0113, q=5.00) [Polar-FNO]",
                "Exp 4 — SXS Scale Blind Test (11 train sims -> SXS:BBH:0113, q=5.00) [Direct FNO]",
                "Exp 4 — SXS Scale Blind Test (11 train sims -> SXS:BBH:0201, q=2.32) [Direct FNO]",
                "Exp 4 — SXS Scale Blind Test (11 train sims -> SXS:BBH:0593, q=1.50) [Direct FNO]",
                "Exp 5 — Pre-Merger Hidden Future Forecast (t > -50M, SXS:BBH:0167)",
            ],
            "Relative L2 Error": ["96.67%", "115.46%", "39.66%", "11.50%", "4.34%", "2.66%", "1.81%", "1.48%"],
            "Overlap (Fidelity)": ["25.88%", "16.70%", "94.16%", "99.34%", "99.91%", "99.97%", "99.98%", "99.99%"],
            "Mismatch (1 - Overlap)": ["0.741196", "0.832960", "0.058402", "0.006640", "0.000920", "0.000300", "0.000160", "0.000100"],
        },
        use_container_width=True,
        hide_index=True
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
