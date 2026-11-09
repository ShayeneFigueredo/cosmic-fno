"""
CosmicFNO - Módulo 01: Engenharia de Dados & Geração de Dataset
Autora: Shayene Figueredo (TCC / Sistemas de Informação - UFU)
Descrição:
    Este script atua como nosso "motor de dados". Ele gera milhares de simulações 
    simplificadas de ondas gravitacionais variando as massas dos buracos negros,
    e salva os resultados em Tensores (PyTorch) para treinarmos a Inteligência Artificial.
"""

import os
import numpy as np
import torch  # Biblioteca principal de Inteligência Artificial que usaremos

def gerar_onda_gravitacional(m1=30.0, m2=30.0, fs=2048, duracao_total=1.0, t_merger=0.8):
    """
    Simula a onda gravitacional. 
    Retorna o vetor de tempo (t) e a amplitude da onda (h).
    """
    n_pontos = int(duracao_total * fs)
    t = np.linspace(0, duracao_total, n_pontos)
    t_relativo = t - t_merger
    M_total = m1 + m2
    
    # Frequência e decaimento da onda baseados na massa total
    f_qnm = 248.0 * (60.0 / M_total)
    tau = 0.005 * (M_total / 60.0)

    # Criando um array vazio de zeros para a onda
    onda = np.zeros_like(t)
    
    # Fase Ringdown (t > 0): A onda vibra e vai morrendo aos poucos (decaimento exponencial)
    # OBS: Este é um modelo matemático simplificado (toy model) para a prova de conceito.
    mask_ringdown = t_relativo >= 0
    onda[mask_ringdown] = np.exp(-t_relativo[mask_ringdown] / tau) * np.sin(2 * np.pi * f_qnm * t_relativo[mask_ringdown])
    
    return t, onda

def gerar_dataset(num_amostras=1000, pasta_saida="dataset"):
    """
    Gera milhares de ondas e salva em formato Tensor para a IA.
    """
    print(f"Iniciando geracao do dataset com {num_amostras} amostras...")
    
    # Cria a pasta 'dataset' se ela não existir
    os.makedirs(pasta_saida, exist_ok=True)
    
    # Listas para guardar nossas massas (X) e ondas (Y)
    entradas_X = []
    saidas_Y = []
    
    for i in range(num_amostras):
        # Sorteia massas entre 10 e 100 massas solares
        m1 = np.random.uniform(10, 100)
        m2 = np.random.uniform(10, 100)
        
        # Gera a onda
        tempo, onda = gerar_onda_gravitacional(m1=m1, m2=m2)
        
        # Guarda os parâmetros (m1 e m2) que geraram a onda
        entradas_X.append([m1, m2])
        saidas_Y.append(onda)
        
        if (i + 1) % 200 == 0:
            print(f"{i + 1} ondas geradas...")

    # Converte as listas do Python para Tensores do PyTorch (o formato que a IA entende)
    tensor_X = torch.tensor(entradas_X, dtype=torch.float32)
    tensor_Y = torch.tensor(saidas_Y, dtype=torch.float32)
    
    # Salva os tensores em arquivos binários na pasta dataset
    caminho_X = os.path.join(pasta_saida, "X_massas.pt")
    caminho_Y = os.path.join(pasta_saida, "y_ondas.pt")
    
    torch.save(tensor_X, caminho_X)
    torch.save(tensor_Y, caminho_Y)
    
    print(f"Dataset concluido! Arquivos salvos em: {pasta_saida}")
    print(f"Formato de X (Entradas): {tensor_X.shape} -> {num_amostras} exemplos com 2 características (m1, m2)")
    print(f"Formato de Y (Saídas): {tensor_Y.shape} -> {num_amostras} exemplos com {len(onda)} pontos no tempo")

# Se executarmos este arquivo diretamente, ele roda a função principal
if __name__ == "__main__":
    gerar_dataset(num_amostras=1000)
