"""
CosmicFNO - Plataforma Visual Interativa (Animação 3D Otimizada)
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Versão Otimizada com Renderização Client-Side.
    Em vez de forçar o Python a atualizar a tela frame a frame (o que trava
    notebooks comuns), nós construímos a animação inteira em JavaScript (Plotly Nativo)
    e enviamos de uma só vez para o navegador reproduzir com aceleração de hardware.
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="CosmicFNO | Simulação Visual", layout="wide", page_icon="🌌")

st.markdown("""
    <style>
    .stApp { background-color: #050505; }
    .big-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title { font-size: 1.1rem; color: #a0a0a0; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_onda_real():
    caminho_t = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dataset_sxs", "dataset_t.pt")
    caminho_h = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dataset_sxs", "dataset_h22.pt")
    
    t_raw = torch.load(caminho_t)[0].numpy()
    h_raw = torch.load(caminho_h)[0].numpy()
    
    # OTIMIZAÇÃO EXTREMA: Reduzimos para apenas 150 frames totais.
    # O navegador não aguenta renderizar milhares de pontos 3D no WebGL.
    janela_final = 1500
    passo = 10  # Salta 10 pontos por vez
    
    t = t_raw[-janela_final::passo]
    h = h_raw[-janela_final::passo]
    t = t - t[0]
    
    return t, h

def calcular_posicoes_orbitais(t, h):
    N = len(t)
    x1, y1, z1 = np.zeros(N), np.zeros(N), np.zeros(N)
    x2, y2, z2 = np.zeros(N), np.zeros(N), np.zeros(N)
    
    amplitude = np.abs(h)
    raio = np.linspace(1.5, 0.0, N) 
    
    omega = np.linspace(2.0, 15.0, N)
    angulo = np.cumsum(omega) * (t[1] - t[0])
    
    for i in range(N):
        if i < N * 0.9: 
            r = raio[i]
            x1[i], y1[i] = r * np.cos(angulo[i]), r * np.sin(angulo[i])
            x2[i], y2[i] = -r * np.cos(angulo[i]), -r * np.sin(angulo[i])
        else: 
            vibra = amplitude[i] * 0.1
            x1[i], y1[i], z1[i] = vibra, vibra, 0
            x2[i], y2[i], z2[i] = vibra, vibra, 0
            
    return x1, y1, z1, x2, y2, z2

st.markdown('<p class="big-title">CosmicFNO: The Black Hole Dance</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Animação renderizada no cliente (Navegador) para performance em notebooks convencionais.</p>', unsafe_allow_html=True)

t, h = carregar_onda_real()
x1, y1, z1, x2, y2, z2 = calcular_posicoes_orbitais(t, h)

# Em vez de um loop no Python que trava a máquina, vamos usar o sistema de FRAMES nativo do Plotly.
# O Python empacota tudo num arquivo só, manda pro navegador, e o navegador toca como se fosse um vídeo.

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.6, 0.4],
    specs=[[{"type": "scatter3d"}], [{"type": "xy"}]],
    subplot_titles=("Órbita Espacial 3D", "Emissão da Onda Gravitacional")
)

# Traços iniciais (Frame 0)
fig.add_trace(go.Scatter3d(x=[x1[0]], y=[y1[0]], z=[z1[0]], mode='markers', marker=dict(size=10, color='#00f2fe'), name="BN 1"), row=1, col=1)
fig.add_trace(go.Scatter3d(x=[x2[0]], y=[y2[0]], z=[z2[0]], mode='markers', marker=dict(size=10, color='#4facfe'), name="BN 2"), row=1, col=1)
fig.add_trace(go.Scatter(x=t[:1], y=h[:1], mode='lines', line=dict(color='#8A2BE2', width=3), name="Strain"), row=2, col=1)
fig.add_trace(go.Scatter(x=[t[0]], y=[h[0]], mode='markers', marker=dict(color='red', size=8), showlegend=False), row=2, col=1)

# Construindo os frames (Toda a animação pré-calculada)
frames = []
for i in range(1, len(t)):
    frames.append(go.Frame(
        data=[
            go.Scatter3d(x=[x1[i]], y=[y1[i]], z=[z1[i]]),
            go.Scatter3d(x=[x2[i]], y=[y2[i]], z=[z2[i]]),
            go.Scatter(x=t[:i], y=h[:i]),
            go.Scatter(x=[t[i]], y=[h[i]])
        ],
        traces=[0, 1, 2, 3] # Diz para o Plotly quais linhas da figura original ele deve atualizar
    ))

fig.frames = frames

# Adicionando botão de Play Nativo
fig.update_layout(
    height=800,
    template="plotly_dark",
    plot_bgcolor='#050505',
    paper_bgcolor='#050505',
    showlegend=False,
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="▶️ PLAY", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])],
        x=0.5, y=1.1, xanchor="center", yanchor="top"
    )],
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.update_scenes(
    xaxis=dict(range=[-2, 2], visible=False),
    yaxis=dict(range=[-2, 2], visible=False),
    zaxis=dict(range=[-2, 2], visible=False),
    camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
)

fig.update_yaxes(range=[min(h)*1.2, max(h)*1.2], row=2, col=1)
fig.update_xaxes(range=[0, max(t)], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)
