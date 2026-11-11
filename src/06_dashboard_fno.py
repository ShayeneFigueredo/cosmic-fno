"""
CosmicFNO - Dashboard Interativo
Autora: Shayene Figueredo (BSI/UFU)
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import os
import sys
import time

# Configuração da Página - DEVE SER A PRIMEIRA LINHA
st.set_page_config(page_title="CosmicFNO | Gravitational Waves", layout="wide", page_icon="🌌")

# ==========================================
# 1. ESTILIZAÇÃO VISUAL (CSS Customizado)
# ==========================================
# Cores escolhidas: Fundo cinza escuro, destaques em roxo escuro e azul escuro
st.markdown("""
    <style>
    /* Fundo da tela principal */
    .stApp {
        background-color: #121212;
    }
    
    /* Título principal com gradiente azul/roxo */
    .big-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4cc9f0, #7209b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Subtítulo */
    .sub-title {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 30px;
    }
    
    /* Cartões de métricas */
    div[data-testid="metric-container"] {
        background-color: #1e1e24;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #7209b7;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARREGAMENTO DO MODELO FNO (Cachê para ser rápido)
# ==========================================
@st.cache_resource
def carregar_modelo_e_dados():
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    import importlib.util
    fno_module = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py"))
    modelo_fno = importlib.util.module_from_spec(fno_module)
    fno_module.loader.exec_module(modelo_fno)
    
    device = torch.device("cpu") # Rodamos na CPU no painel pra garantir compatibilidade
    modelo = modelo_fno.CosmicFNO(num_modos=32, largura=64).to(device)
    
    caminho_pesos = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelos_salvos", "fno_real_generalizado.pth")
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=device, weights_only=True))
    modelo.eval()
    
    # Carrega também um vetor de tempo real para o eixo X do gráfico
    caminho_t = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dataset_sxs", "dataset_t.pt")
    t_raw = torch.load(caminho_t)[0] # Pega o tempo da primeira simulacao
    
    return modelo, t_raw, device

modelo, t_raw, device = carregar_modelo_e_dados()

# ==========================================
# 3. INTERFACE DO USUÁRIO
# ==========================================

# Cabeçalho com imagem real da NASA (Telescópio James Webb / Nebulosa Carina)
# Usando imagens reais de domínio público para garantir que não é IA.
st.image("https://stsci-opo.org/STScI-01G7ETPF7DVBJAC42JR5N6EQRH.png", use_container_width=True, caption="Imagem Real: Nebulosa Carina capturada pelo Telescópio James Webb (NASA/ESA/CSA)")

st.markdown('<p class="big-title">CosmicFNO Observatory</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Emulação em Tempo Real de Ondas Gravitacionais (SXS Catalog) via Fourier Neural Operators</p>', unsafe_allow_html=True)

# Layout de Colunas
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 🎛️ Parâmetros Físicos")
    st.markdown("Ajuste as massas dos buracos negros simulados.")
    
    # Sliders para o usuário brincar
    q_input = st.slider("Razão de Massa (q)", min_value=1.0, max_value=8.0, value=2.5, step=0.1, 
                        help="q = m1/m2. Se q=1.0, os buracos negros têm o mesmo peso. Se q=8.0, um é 8x mais pesado.")
    
    spin_input = st.slider("Spin (Rotação Z)", min_value=-0.8, max_value=0.8, value=0.0, step=0.1,
                           help="Velocidade de rotação dos buracos negros.")
    
    st.markdown("---")
    st.markdown("### ⏱️ Performance")
    tempo_supercomp = st.number_input("Tempo estimado em Supercomputador (Dias)", value=14)
    
    gerar = st.button("🚀 Emular Colisão!", use_container_width=True, type="primary")

with col2:
    if gerar or 'ja_gerou' not in st.session_state:
        st.session_state['ja_gerou'] = True
        
        # 1. Preparar a entrada da rede neural
        inicio_inferencia = time.time()
        
        num_pontos = len(t_raw)
        q_tensor = torch.full((num_pontos,), q_input)
        spin_tensor = torch.full((num_pontos,), spin_input)
        
        t_norm = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min())
        
        # [1, 8192, 3]
        X = torch.stack([q_tensor, spin_tensor, t_norm], dim=-1).unsqueeze(0).to(device)
        
        # 2. Rodar a IA
        with torch.no_grad():
            previsao = modelo(X).squeeze().numpy()
            
        tempo_inferencia = time.time() - inicio_inferencia
        
        # 3. Mostrar Métricas (O Tapa na Cara Acadêmico)
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Tempo do Supercomputador", value=f"{tempo_supercomp} dias")
        m2.metric(label="Tempo do CosmicFNO", value=f"{tempo_inferencia:.4f} seg", delta="Instantâneo!", delta_color="normal")
        m3.metric(label="Resolução", value=f"{num_pontos} pontos")
        
        # 4. Desenhar o Gráfico Interativo com Plotly (Fundo escuro, linha neon)
        fig = go.Figure()
        
        # Vamos dar zoom nos últimos 2000 pontos para ver o choque bonito
        zoom_pontos = 2500
        
        fig.add_trace(go.Scatter(
            x=t_raw[-zoom_pontos:].numpy(), 
            y=previsao[-zoom_pontos:], 
            mode='lines',
            name=f'Onda h_22 (q={q_input})',
            line=dict(color='#4cc9f0', width=2.5) # Azul neon
        ))
        
        fig.update_layout(
            title="Sinal da Onda Gravitacional Emulada (Merger & Ringdown)",
            xaxis_title="Tempo (Unidades Geométricas)",
            yaxis_title="Amplitude do Strain (h)",
            template="plotly_dark",
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#121212',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
