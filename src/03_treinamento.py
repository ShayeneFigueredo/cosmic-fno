"""
CosmicFNO - Módulo 03: Loop de Treinamento
Autora: Shayene Figueredo (TCC / Sistemas de Informação - UFU)
Descrição:
    Este script carrega o dataset gerado (Passo 1), o modelo FNO (Passo 2) e
    realiza o processo de otimização (treinamento) usando PyTorch.
    A função de perda utilizada é o MSE (Mean Squared Error).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
fno_module = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py"))
modelo_fno = importlib.util.module_from_spec(fno_module)
fno_module.loader.exec_module(modelo_fno)
CosmicFNO = modelo_fno.CosmicFNO

def preparar_dados(caminho_X="dataset/X_massas.pt", caminho_Y="dataset/y_ondas.pt", batch_size=32):
    """
    Carrega os tensores do disco, formata para a IA e divide em pacotes (batches).
    """
    print("[INFO] Carregando dataset do disco...")
    X_massas = torch.load(caminho_X) # Formato: [1000, 2]
    y_ondas = torch.load(caminho_Y)   # Formato: [1000, 2048]
    
    num_amostras = X_massas.size(0)
    num_pontos_tempo = y_ondas.size(1)
    
    # O modelo FNO espera a entrada no formato: (Lote, Tempo, Features(m1, m2, t))
    # Precisamos criar o vetor de tempo "t" e junta-lo com as massas.
    t_vetor = torch.linspace(0, 1, num_pontos_tempo).unsqueeze(0).repeat(num_amostras, 1) # [1000, 2048]
    
    # Expandimos as massas para combinar com o tempo (cada ponto no tempo precisa saber a massa)
    m1_expandido = X_massas[:, 0].unsqueeze(1).repeat(1, num_pontos_tempo)
    m2_expandido = X_massas[:, 1].unsqueeze(1).repeat(1, num_pontos_tempo)
    
    # Empilha tudo: (m1, m2, tempo)
    # Resultado final de X: [1000, 2048, 3]
    X_final = torch.stack([m1_expandido, m2_expandido, t_vetor], dim=-1)
    
    # O PyTorch exige que a saida Y também tenha uma dimensao extra no final [1000, 2048, 1]
    y_final = y_ondas.unsqueeze(-1)
    
    # Cria o DataLoader (um iterador otimizado que entrega os dados em lotes)
    dataset = TensorDataset(X_final, y_final)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"[OK] Dados preparados! Total de lotes (batches): {len(dataloader)}")
    return dataloader

def treinar_modelo(epocas=10, batch_size=32, learning_rate=0.001):
    """
    Funcao principal de treinamento.
    """
    print("[INFO] Configurando dispositivo (CPU/GPU)...")
    # Usa a Placa de Video se existir, caso contrario usa Processador (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo selecionado: {device}")
    
    # Carrega dados
    dataloader = preparar_dados(batch_size=batch_size)
    
    # Inicializa o Modelo
    modelo = CosmicFNO(num_modos=16, largura=64).to(device)
    
    # Otimizador (Adam) e Funcao de Perda (MSE)
    otimizador = optim.Adam(modelo.parameters(), lr=learning_rate)
    funcao_perda = nn.MSELoss()
    
    print("-" * 50)
    print(f"[INFO] Iniciando Treinamento ({epocas} epocas)...")
    print("-" * 50)
    
    # Loop de epocas (quantas vezes a IA vai ver o dataset inteiro)
    for epoca in range(epocas):
        modelo.train()
        perda_total = 0.0
        inicio_tempo = time.time()
        
        # Iterando sobre os lotes (batches)
        for lote_X, lote_Y in dataloader:
            # Envia os dados para a memoria correta (CPU ou GPU)
            lote_X = lote_X.to(device)
            lote_Y = lote_Y.to(device)
            
            # 1. Zera os gradientes do passo anterior
            otimizador.zero_grad()
            
            # 2. Faz a previsao (A IA tenta adivinhar a onda)
            previsao = modelo(lote_X)
            
            # 3. Calcula o erro (distancia entre a previsao e a onda real)
            erro = funcao_perda(previsao, lote_Y)
            
            # 4. Backpropagation (A IA aprende onde errou e ajusta as matematicas internas)
            erro.backward()
            
            # 5. Atualiza os pesos
            otimizador.step()
            
            perda_total += erro.item()
            
        # Calcula a media de erro da epoca
        perda_media = perda_total / len(dataloader)
        tempo_epoca = time.time() - inicio_tempo
        
        print(f"[EPOCA {epoca+1:02d}/{epocas}] Erro Medio (MSE): {perda_media:.6f} | Tempo: {tempo_epoca:.2f}s")
        
    print("-" * 50)
    print("[OK] Treinamento concluido!")
    
    # Salva o "cerebro" treinado no disco
    os.makedirs("modelos_salvos", exist_ok=True)
    caminho_modelo = "modelos_salvos/cosmic_fno_pesos.pth"
    torch.save(modelo.state_dict(), caminho_modelo)
    print(f"[INFO] Pesos do modelo salvos em: {caminho_modelo}")

if __name__ == "__main__":
    # Roda um treinamento rápido de 10 épocas para teste
    treinar_modelo(epocas=10, batch_size=32)

