"""
CosmicFNO - Treinamento em Lote (Generalização)
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Este script treina a rede FNO usando todo o banco de dados (as 5 simulações).
    O objetivo é que a rede não apenas decore uma onda, mas aprenda as regras 
    físicas: como a onda muda quando a massa (q) e o spin mudam?
    Isso é o que vai dar poder ao nosso futuro Dashboard!
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys

# Importa o modelo FNO
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
fno_module = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py"))
modelo_fno = importlib.util.module_from_spec(fno_module)
fno_module.loader.exec_module(modelo_fno)
CosmicFNO = modelo_fno.CosmicFNO

def treinar_lote():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")
    
    pasta_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dataset_sxs")
    caminho_t = os.path.join(pasta_data, "dataset_t.pt")
    caminho_h22 = os.path.join(pasta_data, "dataset_h22.pt")
    caminho_params = os.path.join(pasta_data, "dataset_params.pt")
    
    # Carrega os tensores das 5 simulacoes
    t_raw = torch.load(caminho_t) # [5, 8192]
    h22_raw = torch.load(caminho_h22) # [5, 8192]
    params = torch.load(caminho_params) # [5, 3] -> (q, spin1, spin2)
    
    num_sims = t_raw.shape[0]
    num_pontos = t_raw.shape[1]
    print(f"[INFO] Carregadas {num_sims} simulacoes com {num_pontos} pontos cada no tempo.")
    
    # O nosso modelo FNO (02_modelo_fno.py) aceita exatamente 3 variáveis (features).
    # Vamos passar: Razão de Massa (q), Spin 1, e Tempo Normalizado.
    q = params[:, 0].unsqueeze(1).repeat(1, num_pontos) # Expande para [5, 8192]
    spin1 = params[:, 1].unsqueeze(1).repeat(1, num_pontos) # Expande para [5, 8192]
    
    # Normalizando o tempo (de 0 a 1) para a rede neural não ficar confusa com tempos grandes
    t_min = t_raw.min(dim=1, keepdim=True).values
    t_max = t_raw.max(dim=1, keepdim=True).values
    t_norm = (t_raw - t_min) / (t_max - t_min)
    
    # Empilhando tudo: [Lote, Tempo, Features] -> [5, 8192, 3]
    X = torch.stack([q, spin1, t_norm], dim=-1).to(device)
    Y = h22_raw.unsqueeze(-1).to(device)
    
    # Inicializa o Cerebro (Modelo FNO)
    modelo = CosmicFNO(num_modos=32, largura=64).to(device)
    otimizador = optim.Adam(modelo.parameters(), lr=0.002)
    funcao_perda = nn.MSELoss()
    
    epocas = 200
    print(f"================ Iniciando Treinamento ({epocas} epocas) ================")
    for epoca in range(epocas):
        otimizador.zero_grad()
        
        # O FNO tenta prever as 5 ondas AO MESMO TEMPO
        previsao = modelo(X)
        erro = funcao_perda(previsao, Y)
        
        erro.backward()
        otimizador.step()
        
        if (epoca + 1) % 20 == 0 or epoca == 0:
            print(f"-> Epoca [{epoca+1:03d}/{epocas}] | Erro Medio (MSE): {erro.item():.6f}")
            
    print("================ Treinamento Concluido! ================")
    
    # Salvando os "pesos" (O cerebro treinado) no disco
    pasta_modelos = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelos_salvos")
    os.makedirs(pasta_modelos, exist_ok=True)
    caminho_pesos = os.path.join(pasta_modelos, "fno_real_generalizado.pth")
    torch.save(modelo.state_dict(), caminho_pesos)
    print(f"[OK] Cerebro (Pesos do Modelo) salvo em: {caminho_pesos}")
    
    # Gerando Grafico de Validacao (Para provar que a IA consegue distinguir as massas)
    modelo.eval()
    with torch.no_grad():
        prev_final = modelo(X).cpu().squeeze().numpy()
        y_real = Y.cpu().squeeze().numpy()
        t_np = t_raw.cpu().numpy()
        
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot Simulação 1 (q mais leve)
    axs[0].plot(t_np[0, -1500:], y_real[0, -1500:], label=f"Real (q={params[0,0]:.2f})", color='purple', linewidth=2)
    axs[0].plot(t_np[0, -1500:], prev_final[0, -1500:], label="FNO Previsto", color='orange', linestyle='dashed', linewidth=2)
    axs[0].set_title(f"A IA desenhando a Simulação 1 (Massa q={params[0,0]:.2f})")
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot Simulação 5 (q mais pesado)
    axs[1].plot(t_np[4, -1500:], y_real[4, -1500:], label=f"Real (q={params[4,0]:.2f})", color='green', linewidth=2)
    axs[1].plot(t_np[4, -1500:], prev_final[4, -1500:], label="FNO Previsto", color='red', linestyle='dashed', linewidth=2)
    axs[1].set_title(f"A IA desenhando a Simulação 5 (Massa q={params[4,0]:.2f})")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    caminho_grafico = os.path.join(pasta_data, "resultado_generalizacao.png")
    plt.savefig(caminho_grafico)
    print(f"[OK] Grafico de comparacao salvo em: {caminho_grafico}")

if __name__ == "__main__":
    treinar_lote()
