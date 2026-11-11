"""
CosmicFNO - Treinamento com Validação Hold-Out
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Este script corrige a falha de 'Data Leakage'. Ele divide o dataset alinhado
    em Treino (4 simulações) e Teste (1 simulação).
    O erro é reportado na simulação de teste, provando a generalização (interpolação)
    em uma massa que a rede nunca viu.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
fno_module = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py"))
modelo_fno = importlib.util.module_from_spec(fno_module)
fno_module.loader.exec_module(modelo_fno)
CosmicFNO = modelo_fno.CosmicFNO

def treinar_holdout():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")
    
    pasta_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dataset_sxs")
    
    # Carregando os tensores ALINHADOS
    t_raw = torch.load(os.path.join(pasta_data, "dataset_t_alinhado.pt"))
    h22_raw = torch.load(os.path.join(pasta_data, "dataset_h22_alinhado.pt"))
    params = torch.load(os.path.join(pasta_data, "dataset_params_alinhado.pt"))
    
    num_sims = t_raw.shape[0]
    num_pontos = t_raw.shape[1]
    
    # Formatando as Features
    q = params[:, 0].unsqueeze(1).repeat(1, num_pontos)
    spin1 = params[:, 1].unsqueeze(1).repeat(1, num_pontos)
    
    t_min = t_raw.min(dim=1, keepdim=True).values
    t_max = t_raw.max(dim=1, keepdim=True).values
    t_norm = (t_raw - t_min) / (t_max - t_min)
    
    X = torch.stack([q, spin1, t_norm], dim=-1)
    Y = h22_raw.unsqueeze(-1)
    
    # --- SPLIT RIGOROSO (HOLD-OUT) ---
    # Vamos usar a simulação de índice 3 (SXS:BBH:0113, q=2.0) como TESTE
    # e as outras 4 como TREINO.
    idx_teste = 3
    indices_treino = [i for i in range(num_sims) if i != idx_teste]
    
    X_treino = X[indices_treino].to(device)
    Y_treino = Y[indices_treino].to(device)
    
    X_teste = X[idx_teste:idx_teste+1].to(device) # Mantém a dimensão do lote
    Y_teste = Y[idx_teste:idx_teste+1].to(device)
    
    print(f"[INFO] Treinando com {len(indices_treino)} simulações. Testando na massa q={params[idx_teste, 0]:.2f}")
    
    # Modelo e Otimização
    modelo = CosmicFNO(num_modos=32, largura=64).to(device)
    otimizador = optim.Adam(modelo.parameters(), lr=0.002)
    funcao_perda = nn.MSELoss()
    
    epocas = 250
    print(f"================ Iniciando Treinamento ({epocas} epocas) ================")
    for epoca in range(epocas):
        modelo.train()
        otimizador.zero_grad()
        
        previsao_treino = modelo(X_treino)
        erro_treino = funcao_perda(previsao_treino, Y_treino)
        
        erro_treino.backward()
        otimizador.step()
        
        # Avaliação no Teste (O que importa para a ciência)
        if (epoca + 1) % 50 == 0 or epoca == 0:
            modelo.eval()
            with torch.no_grad():
                previsao_teste = modelo(X_teste)
                erro_teste = funcao_perda(previsao_teste, Y_teste)
            print(f"-> Epoca [{epoca+1:03d}/{epocas}] | Erro Treino: {erro_treino.item():.6f} | Erro Teste (Inédito): {erro_teste.item():.6f}")
            
    print("================ Treinamento Concluido! ================")
    
    # Gráfico de Validação Científica (A rede previu a onda que nunca viu?)
    modelo.eval()
    with torch.no_grad():
        prev_final = modelo(X_teste).cpu().squeeze().numpy()
        y_real = Y_teste.cpu().squeeze().numpy()
        t_np = t_raw[idx_teste].cpu().numpy()
        
    plt.figure(figsize=(10, 5))
    # Plotamos tudo (agora são 8000 pontos alinhados, então focamos do centro pro final)
    idx_centro = num_pontos // 2
    plt.plot(t_np[idx_centro-1500:idx_centro+1500], y_real[idx_centro-1500:idx_centro+1500], label=f"Gabarito Real (q={params[idx_teste,0]:.2f})", color='purple', linewidth=2)
    plt.plot(t_np[idx_centro-1500:idx_centro+1500], prev_final[idx_centro-1500:idx_centro+1500], label="Predição do FNO (Interpolação)", color='orange', linestyle='dashed', linewidth=2)
    plt.title(f"Validação Científica (Hold-Out): Predição de massa inédita (q={params[idx_teste,0]:.2f})")
    plt.xlabel("Tempo Alinhado (t=0 é o Merger)")
    plt.ylabel("Strain h_22")
    plt.legend()
    plt.grid(True)
    
    caminho_grafico = os.path.join(pasta_data, "resultado_holdout_alinhado.png")
    plt.savefig(caminho_grafico)
    print(f"[OK] Gráfico de validação salvo em: {caminho_grafico}")

if __name__ == "__main__":
    treinar_holdout()
