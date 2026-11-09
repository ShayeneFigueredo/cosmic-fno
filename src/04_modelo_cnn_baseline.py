"""
CosmicFNO - Modelo Baseline: CNN 1D (Rede Neural Convolucional)
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Este script implementa uma Rede Neural Convolucional 1D (CNN 1D).
    As CNNs são excelentes para imagens e sequências, mas têm dificuldade 
    em manter a resolução contínua (são presas a grades locais).
    Usaremos este modelo como o "Baseline" (Ponto de Partida/Comparação)
    para provar na Iniciação Científica que o FNO é superior.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

class BaselineCNN1D(nn.Module):
    def __init__(self):
        super(BaselineCNN1D, self).__init__()
        # A entrada tem 3 canais (q, spin, tempo)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (Batch, Channels, Length)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def treinar_cnn_baseline(epocas=100, learning_rate=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo selecionado: {device}")
    
    pasta_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    caminho_t = os.path.join(pasta_data, "SXS_BBH_0305_tempo.pt")
    caminho_h = os.path.join(pasta_data, "SXS_BBH_0305_h22.pt")
    
    # Cortamos para os mesmos últimos 8192 pontos (como fizemos no ajuste do FNO)
    t_real = torch.load(caminho_t)[-8192:]
    h_real = torch.load(caminho_h)[-8192:]
    num_pontos = len(t_real)
    
    # Preparando dados
    q_tensor = torch.full((num_pontos,), 1.22)
    spin_tensor = torch.full((num_pontos,), 0.33)
    t_norm = (t_real - t_real.min()) / (t_real.max() - t_real.min())
    
    # Empilhando: (q, spin, tempo) -> [8192, 3]
    X_entrada = torch.stack([q_tensor, spin_tensor, t_norm], dim=-1)
    
    # IMPORTANTE: CNNs no PyTorch esperam (Batch, Channels, Length). 
    # O FNO esperava (Batch, Length, Channels). Precisamos transpor (permute).
    X_entrada = X_entrada.unsqueeze(0).permute(0, 2, 1).to(device) # Fica: [1, 3, 8192]
    
    # Alvo: [1, 1, 8192]
    Y_alvo = h_real.unsqueeze(0).unsqueeze(1).to(device)
    
    print("[INFO] Inicializando a arquitetura CNN 1D Baseline...")
    modelo = BaselineCNN1D().to(device)
    otimizador = optim.Adam(modelo.parameters(), lr=learning_rate)
    funcao_perda = nn.MSELoss()
    
    print(f"================ Iniciando Treinamento CNN ({epocas} épocas) ================")
    for epoca in range(epocas):
        otimizador.zero_grad()
        previsao = modelo(X_entrada)
        erro = funcao_perda(previsao, Y_alvo)
        erro.backward()
        otimizador.step()
        
        if (epoca + 1) % 10 == 0 or epoca == 0:
            print(f"-> Época [{epoca+1:03d}/{epocas}] | Erro (MSE): {erro.item():.8f}")
            
    print("================ Treinamento Concluído! ================")
    
    modelo.eval()
    with torch.no_grad():
        previsao_final = modelo(X_entrada).cpu().squeeze().numpy()
        h_real_np = h_real.numpy()
        t_np = t_real.numpy()
        
    plt.figure(figsize=(12, 5))
    plt.plot(t_np[-2000:], h_real_np[-2000:], label="Sinal Real", color='purple', alpha=0.7, linewidth=2)
    plt.plot(t_np[-2000:], previsao_final[-2000:], label="Reconstrução da CNN 1D", color='red', linestyle='dashed', linewidth=2)
    plt.title("Modelo Baseline (CNN 1D) tentando aprender a Física")
    plt.xlabel("Tempo (M)")
    plt.ylabel("Strain h_22")
    plt.legend()
    plt.grid(True)
    
    caminho_grafico = os.path.join(pasta_data, "resultado_treino_cnn.png")
    plt.savefig(caminho_grafico)
    print(f"[OK] Gráfico comparativo da CNN salvo em: {caminho_grafico}")

if __name__ == "__main__":
    treinar_cnn_baseline()
