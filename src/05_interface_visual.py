"""
CosmicFNO - Módulo 05: Interface Visual (Web App)
Autora: Shayene Figueredo (TCC / Sistemas de Informação - UFU)
Descrição:
    Interface web interativa para o modelo FNO. 
    Permite ao usuário (ou a banca do TCC) alterar as massas dos buracos negros 
    em tempo real e ver a IA gerando a onda gravitacional instantaneamente.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib.util

# --- Configuração de Importação dos Módulos Anteriores ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def carregar_modulos():
    # Importa 01
    sxs_module = importlib.util.spec_from_file_location("01_extracao_e_analise_sxs", os.path.join(os.path.dirname(__file__), "01_extracao_e_analise_sxs.py"))
    extracao = importlib.util.module_from_spec(sxs_module)
    sxs_module.loader.exec_module(extracao)
    
    # Importa 02
    fno_module = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py"))
    modelo_fno = importlib.util.module_from_spec(fno_module)
    fno_module.loader.exec_module(modelo_fno)
    
    return extracao.gerar_onda_gravitacional, modelo_fno.CosmicFNO

gerar_onda, CosmicFNO = carregar_modulos()

# --- Carregamento da IA (Cache para não travar a tela) ---
@st.cache_resource
def carregar_ia():
    modelo = CosmicFNO(num_modos=16, largura=64)
    caminho_pesos = os.path.join(os.path.dirname(__file__), "modelos_salvos", "cosmic_fno_pesos.pth")
    if os.path.exists(caminho_pesos):
        modelo.load_state_dict(torch.load(caminho_pesos, map_location=torch.device('cpu')))
    # Tenta dois caminhos: relativo ao script ou relativo a pasta onde o terminal rodou (raiz)
    caminho_pesos_1 = os.path.join(os.path.dirname(__file__), "modelos_salvos", "cosmic_fno_pesos.pth")
    caminho_pesos_2 = os.path.join(os.getcwd(), "modelos_salvos", "cosmic_fno_pesos.pth")
    
    caminho_escolhido = None
    if os.path.exists(caminho_pesos_1):
        caminho_escolhido = caminho_pesos_1
    elif os.path.exists(caminho_pesos_2):
        caminho_escolhido = caminho_pesos_2

    if caminho_escolhido:
        modelo.load_state_dict(torch.load(caminho_escolhido, map_location=torch.device('cpu'), weights_only=True))
        modelo.eval()
        return modelo
    else:
        return None

modelo_ia = carregar_ia()

# --- Construção da Interface Visual (Frontend) ---
st.set_page_config(page_title="Cosmic FNO - TCC", page_icon="🌌", layout="wide")

st.title("🌌 Cosmic FNO: Simulador de Ondas Gravitacionais")
st.markdown("""
Esta aplicação demonstra o uso de **Fourier Neural Operators (FNO)** para prever a fase *Ringdown* 
da coalescência de Buracos Negros. Altere as massas no menu lateral e veja a inferência da IA em tempo real!
""")

# Menu Lateral (Controles)
st.sidebar.header("🎛️ Parâmetros da Colisão")
m1 = st.sidebar.slider("Massa do Buraco Negro 1 (M☉)", min_value=10.0, max_value=100.0, value=30.0, step=1.0)
m2 = st.sidebar.slider("Massa do Buraco Negro 2 (M☉)", min_value=10.0, max_value=100.0, value=40.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido por Shayene Figueredo (BSI/UFU)")

if modelo_ia is None:
    st.error("⚠️ Cérebro da IA não encontrado! Rode o script de treinamento (03) primeiro.")
else:
    # --- Processamento ---
    with st.spinner("A IA está calculando a deformação do espaço-tempo..."):
        # 1. Matemática Pura (Gabarito)
        tempo_real, onda_real = gerar_onda(m1=m1, m2=m2)
        
        # 2. IA FNO
        num_pontos = len(tempo_real)
        t_vetor = torch.linspace(0, 1, num_pontos)
        m1_vetor = torch.full((num_pontos,), m1)
        m2_vetor = torch.full((num_pontos,), m2)
        entrada_ia = torch.stack([m1_vetor, m2_vetor, t_vetor], dim=-1).unsqueeze(0).to(torch.float32)
        
        with torch.no_grad():
            onda_ia = modelo_ia(entrada_ia).squeeze().numpy()

    # --- Gráfico Interativo ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tempo_real, onda_real, label="Física Tradicional (Equações)", color="#1f77b4", linewidth=2)
    ax.plot(tempo_real, onda_ia, label="Previsão Cosmic FNO (IA)", color="#ff7f0e", linestyle="--", linewidth=2)
    
    ax.set_title(f"Colisão: {m1} M☉ + {m2} M☉")
    ax.set_xlabel("Tempo (segundos)")
    ax.set_ylabel("Strain (h)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)
    
    # --- Métricas de Performance ---
    col1, col2 = st.columns(2)
    col1.metric("Resolução da Simulação", f"{num_pontos} pontos")
    col2.metric("Status da IA", "Online e Invariante 🟢")

