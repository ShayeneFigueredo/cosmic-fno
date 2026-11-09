"""
CosmicFNO - Módulo 04: Validação e Visualização
Autora: Shayene Figueredo (TCC / Sistemas de Informação - UFU)
Descrição:
    Este script carrega o modelo treinado e pede para ele simular uma
    onda gravitacional de massas que ele nunca viu antes.
    Em seguida, plota o resultado comparando com a física real (matemática pura),
    comprovando visualmente que o Fourier Neural Operator funciona.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util

# Import 01
sxs_module = importlib.util.spec_from_file_location("01_extracao_e_analise_sxs", os.path.join(os.path.dirname(__file__), "01_extracao_e_analise_sxs.py"))
extracao = importlib.util.module_from_spec(sxs_module)
sxs_module.loader.exec_module(extracao)
gerar_onda_gravitacional = extracao.gerar_onda_gravitacional

# Import 02
fno_module = importlib.util.spec_from_file_location("02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py"))
modelo_fno = importlib.util.module_from_spec(fno_module)
fno_module.loader.exec_module(modelo_fno)
CosmicFNO = modelo_fno.CosmicFNO

def rodar_validacao():
    print("[INFO] Iniciando o processo de validacao e comparacao...")
    
    # 1. ESCOLHEMOS UM CASO NOVO (Que a IA não decorou)
    # Por exemplo: Colisão de um Buraco Negro de 37 e outro de 62 massas solares.
    m1_teste = 37.0
    m2_teste = 62.0
    print(f"[INFO] Caso Teste: m1={m1_teste}, m2={m2_teste}")
    
    # 2. GERAMOS O "GABARITO" (Ground Truth / Física Matemática Pura)
    tempo_real, onda_real = gerar_onda_gravitacional(m1=m1_teste, m2=m2_teste)
    
    # 3. PREPARAMOS O DADO PARA A IA (O "Prompt")
    num_pontos = len(tempo_real)
    t_vetor = torch.linspace(0, 1, num_pontos)
    m1_vetor = torch.full((num_pontos,), m1_teste)
    m2_vetor = torch.full((num_pontos,), m2_teste)
    
    # Empilha e cria o formato (Lote=1, Tempo=2048, Features=3)
    entrada_ia = torch.stack([m1_vetor, m2_vetor, t_vetor], dim=-1).unsqueeze(0).to(torch.float32)
    
    # 4. CARREGAMOS A IA TREINADA
    modelo = CosmicFNO(num_modos=16, largura=64)
    caminho_pesos = "modelos_salvos/cosmic_fno_pesos.pth"
    
    try:
        modelo.load_state_dict(torch.load(caminho_pesos, map_location=torch.device('cpu')))
        print("[OK] Cerebro da IA carregado com sucesso!")
    except FileNotFoundError:
        print("[ERRO] Arquivo de pesos nao encontrado. Treine o modelo primeiro (Script 03).")
        return
        
    modelo.eval() # Modo de Avaliação (desliga o aprendizado)
    
    # 5. PEDIMOS PARA A IA SIMULAR (Inferência rápida)
    with torch.no_grad():
        saida_ia = modelo(entrada_ia)
        onda_ia = saida_ia.squeeze().numpy() # Tira do formato Tensor para um Array normal
        
    # 6. PLOTAMOS O GRÁFICO DE COMPARAÇÃO
    print("[INFO] Gerando grafico para a monografia...")
    plt.figure(figsize=(12, 6))
    
    # Gráfico do Gabarito Físico (Azul)
    plt.plot(tempo_real, onda_real, label="Calculo Fisico (Equacoes)", color="blue", linewidth=2)
    
    # Gráfico da IA (Vermelho, Pontilhado)
    plt.plot(tempo_real, onda_ia, label="Previsao do Cosmic FNO", color="red", linestyle="--", linewidth=2)
    
    plt.title(f"Teste FNO - Colisao de Buracos Negros ({m1_teste} + {m2_teste} Massas Solares)")
    plt.xlabel("Tempo (segundos)")
    plt.ylabel("Deformacao do Espaco-Tempo (h)")
    plt.legend()
    plt.grid(True)
    
    # Salva o gráfico
    os.makedirs("resultados", exist_ok=True)
    caminho_grafico = "resultados/grafico_validacao.png"
    plt.savefig(caminho_grafico)
    print(f"[OK] Grafico salvo com sucesso em: {caminho_grafico}")

if __name__ == "__main__":
    rodar_validacao()

