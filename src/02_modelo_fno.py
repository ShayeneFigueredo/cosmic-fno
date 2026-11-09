"""
CosmicFNO - Módulo 02: Arquitetura do Modelo FNO
Autora: Shayene Figueredo (TCC / Sistemas de Informação - UFU)
Descrição:
    Implementação de um Fourier Neural Operator 1D usando PyTorch.
    Esta rede não aprende ponto a ponto. Ela converte os dados para o domínio
    das frequências (usando FFT), filtra as frequências mais importantes (modos),
    aprende a física da onda, e converte de volta para o tempo (IFFT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CamadaFourier1D(nn.Module):
    """
    O coração do FNO. Esta camada faz a Transformada de Fourier,
    aplica os pesos da rede neural nas frequências, e faz a Transformada Inversa.
    """
    def __init__(self, canais_entrada, canais_saida, num_modos):
        super(CamadaFourier1D, self).__init__()
        self.canais_entrada = canais_entrada
        self.canais_saida = canais_saida
        self.num_modos = num_modos # Quantas frequências (ingredientes) principais vamos guardar

        # Pesos da rede neural para os números complexos (frequências)
        # É aqui que a IA "aprende" a física da onda
        escala = 1 / (canais_entrada * canais_saida)
        self.pesos = nn.Parameter(escala * torch.rand(canais_entrada, canais_saida, self.num_modos, dtype=torch.cfloat))

    def forward(self, x):
        # 1. Aplica a Transformada Rápida de Fourier (FFT) para ir do TEMPO para a FREQUÊNCIA
        x_frequencia = torch.fft.rfft(x)

        # Prepara um tensor vazio para guardar o resultado
        saida_frequencia = torch.zeros(x.size(0), self.canais_saida, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # 2. Multiplica apenas as frequências mais importantes (num_modos) pelos pesos da IA
        # A notação 'einsum' é uma forma otimizada de multiplicar matrizes em tensores
        saida_frequencia[:, :, :self.num_modos] = torch.einsum('bix,iox->box', 
                                                                x_frequencia[:, :, :self.num_modos], 
                                                                self.pesos)

        # 3. Aplica a Transformada Inversa (IFFT) para voltar da FREQUÊNCIA para o TEMPO
        x_tempo = torch.fft.irfft(saida_frequencia, n=x.size(-1))
        
        return x_tempo

class CosmicFNO(nn.Module):
    """
    A arquitetura completa do Cosmic FNO.
    Ela segue 3 passos: Elevar (Lift) -> Camadas de Fourier -> Projetar (Project)
    """
    def __init__(self, num_modos=16, largura=64):
        super(CosmicFNO, self).__init__()
        self.num_modos = num_modos
        self.largura = largura

        # Passo A: Camada Linear que "Eleva" nossa entrada simples (massas e tempo)
        # para um espaço de muitas dimensões (largura = 64) para a IA ter espaço para pensar.
        # Entrada: 3 características (m1, m2, tempo t)
        self.elevar = nn.Linear(3, self.largura) 

        # Passo B: As camadas mágicas de Fourier (vamos usar 4 camadas seguidas)
        self.fourier0 = CamadaFourier1D(self.largura, self.largura, self.num_modos)
        self.fourier1 = CamadaFourier1D(self.largura, self.largura, self.num_modos)
        self.fourier2 = CamadaFourier1D(self.largura, self.largura, self.num_modos)
        self.fourier3 = CamadaFourier1D(self.largura, self.largura, self.num_modos)

        # Camadas locais (convencionais) que rodam em paralelo com as de Fourier
        self.local0 = nn.Conv1d(self.largura, self.largura, 1)
        self.local1 = nn.Conv1d(self.largura, self.largura, 1)
        self.local2 = nn.Conv1d(self.largura, self.largura, 1)
        self.local3 = nn.Conv1d(self.largura, self.largura, 1)

        # Passo C: Projetar os 64 canais de pensamento de volta para 1 único valor (a amplitude da onda)
        self.projetar1 = nn.Linear(self.largura, 128)
        self.projetar2 = nn.Linear(128, 1)

    def forward(self, x):
        # x entra no formato (lote, pontos_no_tempo, 3_caracteristicas)
        
        # Passo A
        x = self.elevar(x)
        
        # Transpõe porque o PyTorch prefere que os canais fiquem no meio para Convoluções
        x = x.permute(0, 2, 1)

        # Passo B (Aplica as 4 camadas de Fourier com funções de ativação GELU)
        x1 = self.fourier0(x) + self.local0(x)
        x = F.gelu(x1)

        x2 = self.fourier1(x) + self.local1(x)
        x = F.gelu(x2)

        x3 = self.fourier2(x) + self.local2(x)
        x = F.gelu(x3)

        x4 = self.fourier3(x) + self.local3(x)

        # Transpõe de volta
        x = x4.permute(0, 2, 1)

        # Passo C
        x = self.projetar1(x)
        x = F.gelu(x)
        x = self.projetar2(x)

        return x

if __name__ == "__main__":
    # Teste rápido para ver se a arquitetura compila corretamente
    print("[INFO] Inicializando o modelo Cosmic FNO...")
    modelo = CosmicFNO()
    
    # Criamos um "dado falso" só para testar a rede (1 onda, 2048 pontos, 3 caracteristicas: m1, m2, t)
    teste_input = torch.randn(1, 2048, 3)
    
    # Passamos o dado pela rede
    teste_output = modelo(teste_input)
    
    print("[OK] Sucesso! O modelo processou os dados sem erros de arquitetura.")
    print(f"Formato da Entrada: {teste_input.shape} (Lote, Tempo, Features)")
    print(f"Formato da Saida: {teste_output.shape} (Lote, Tempo, Amplitude da Onda)")
