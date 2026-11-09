"""
CosmicFNO - Treinamento com Dado Astrofísico Real
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Este script prova que o nosso modelo FNO (Fourier Neural Operator)
    consegue processar tensores extraídos diretamente de simulações de supercomputadores.
    Como temos apenas 1 onda por enquanto (SXS:BBH:0305), a IA fará uma tarefa
    chamada "Representation Learning" (Aprendizado de Representação) / Overfitting.
    Isso testa se o modelo tem capacidade matemática de absorver a física de uma onda real.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Importa nosso modelo FNO (Módulo 02)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
fno_module = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py"))
modelo_fno = importlib.util.module_from_spec(fno_module)
fno_module.loader.exec_module(modelo_fno)
CosmicFNO = modelo_fno.CosmicFNO

def treinar_sinal_real(epocas=100, learning_rate=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo selecionado: {device}")
    
    # 1. Carregando os tensores REAIS gerados no script 01b
    pasta_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    caminho_t = os.path.join(pasta_data, "SXS_BBH_0305_tempo.pt")
    caminho_h = os.path.join(pasta_data, "SXS_BBH_0305_h22.pt")
    
    t_real = torch.load(caminho_t)[-8192:]
    h_real = torch.load(caminho_h)[-8192:]
    num_pontos = len(t_real)
    print(f"[INFO] Dados reais carregados. Pontos no tempo: {num_pontos}")
    
    # 2. Formatando os dados para a rede FNO
    # A simulação 0305 (GW150914) possui Razão de Massa (q) = 1.22 e Spin = 0.33
    q_tensor = torch.full((num_pontos,), 1.22)
    spin_tensor = torch.full((num_pontos,), 0.33)
    
    # Normalizando o tempo para ficar entre 0 e 1 (Redes Neurais preferem assim)
    t_norm = (t_real - t_real.min()) / (t_real.max() - t_real.min())
    
    # Empilhando: (q, spin, tempo) -> [13746, 3]
    X_entrada = torch.stack([q_tensor, spin_tensor, t_norm], dim=-1)
    
    # Adicionando a dimensão do Lote (Batch) no início. Fica: [1, 13746, 3]
    X_entrada = X_entrada.unsqueeze(0).to(device)
    
    # O alvo (y) é a própria onda. Fica: [1, 13746, 1]
    Y_alvo = h_real.unsqueeze(0).unsqueeze(-1).to(device)
    
    # 3. Preparando o Modelo
    print("[INFO] Inicializando a arquitetura FNO...")
    modelo = CosmicFNO(num_modos=32, largura=64).to(device)
    otimizador = optim.Adam(modelo.parameters(), lr=learning_rate)
    funcao_perda = nn.MSELoss()
    
    # 4. Loop de Treinamento
    print(f"================ Iniciando Treinamento ({epocas} épocas) ================")
    historico_perdas = []
    
    for epoca in range(epocas):
        otimizador.zero_grad()
        
        # O FNO tenta prever/reconstruir a onda real
        previsao = modelo(X_entrada)
        
        # Calcula o Erro
        erro = funcao_perda(previsao, Y_alvo)
        
        # Ajusta os pesos (Backpropagation)
        erro.backward()
        otimizador.step()
        
        historico_perdas.append(erro.item())
        
        if (epoca + 1) % 10 == 0 or epoca == 0:
            print(f"-> Época [{epoca+1:03d}/{epocas}] | Erro (MSE): {erro.item():.8f}")
            
    print("================ Treinamento Concluído! ================")
    
    # 5. Salvando o Resultado Gráfico (Previsão vs Real)
    modelo.eval()
    with torch.no_grad():
        previsao_final = modelo(X_entrada).cpu().squeeze().numpy()
        h_real_np = h_real.numpy()
        t_np = t_real.numpy()
        
    plt.figure(figsize=(12, 5))
    # Para visualizar melhor, vamos plotar os últimos 2000 pontos (onde ocorre a colisão)
    plt.plot(t_np[-2000:], h_real_np[-2000:], label="Sinal Real (Astrofísico)", color='purple', alpha=0.7, linewidth=2)
    plt.plot(t_np[-2000:], previsao_final[-2000:], label="Reconstrução do FNO", color='orange', linestyle='dashed', linewidth=2)
    plt.title("CosmicFNO aprendendo a Física da Simulação SXS:BBH:0305")
    plt.xlabel("Tempo (M)")
    plt.ylabel("Strain h_22")
    plt.legend()
    plt.grid(True)
    
    caminho_grafico = os.path.join(pasta_data, "resultado_treino_real.png")
    plt.savefig(caminho_grafico)
    print(f"[OK] Gráfico comparativo salvo em: {caminho_grafico}")

if __name__ == "__main__":
    treinar_sinal_real()

