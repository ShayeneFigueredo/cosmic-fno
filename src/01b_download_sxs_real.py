"""
CosmicFNO - ExtraÃ§Ã£o de Dados AstrofÃ­sicos Reais (CatÃ¡logo SXS)
Autora: Shayene Figueredo (BSI/UFU)

DescriÃ§Ã£o:
    Este script se conecta ao catÃ¡logo pÃºblico da colaboraÃ§Ã£o SXS (Simulating eXtreme Spacetimes).
    Em vez de usar dados gerados por equaÃ§Ãµes sintÃ©ticas, nÃ³s baixamos um sinal 
    real de uma simulaÃ§Ã£o de coalescÃªncia de Buracos Negros calculada por supercomputadores.
    
    AtenÃ§Ã£o: O script foca unicamente na extraÃ§Ã£o do modo esfÃ©rico harmÃ´nico principal (h_22),
    removendo os demais modos subdominantes para facilitar o aprendizado inicial da nossa rede FNO.
"""

import sxs
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def baixar_e_processar_sxs_real(sxs_id="SXS:BBH:0305", pasta_saida=None):
    if pasta_saida is None:
        # Pega a pasta TCC/data/raw de forma absoluta, nÃ£o importando de onde o terminal roda
        pasta_saida = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    """
    Faz o download do Extrapolated Waveform real do catÃ¡logo SXS.
    """
    print(f"ðŸš€ Iniciando conexÃ£o com o banco de dados SXS para a simulaÃ§Ã£o {sxs_id}...")
    
    # Cria a pasta para salvar os dados brutos se nÃ£o existir
    os.makedirs(pasta_saida, exist_ok=True)
    
    try:
        # 1. DOWNLOAD DA SIMULAÃ‡ÃƒO
        # O sxs.load baixa o arquivo h5 contendo a onda completa.
        # "rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N2.dir" Ã© o padrÃ£o 
        # para a onda gravitacional (strain h) limpa e extrapolada para o infinito.
        print("â³ Baixando a onda gravitacional real (isso pode levar alguns minutos na 1Âª vez)...")
        sim = sxs.load(sxs_id) # Carrega os metadados da simulaÃ§Ã£o
        waveform = sim.h  # Puxa diretamente o Strain extrapolado mais potente disponÃ­vel
        
        # 2. EXTRAÃ‡ÃƒO DO MODO h_22
        # A onda vem com vÃ¡rias frequÃªncias acopladas. NÃ³s usamos o modo l=2, m=2.
        print("ðŸŽ¯ Extraindo o modo dominante (h_22)...")
        idx_22 = waveform.index(2, 2)
        
        # 3. SEPARAÃ‡ÃƒO DE TEMPO E AMPLITUDE
        # O tempo da simulaÃ§Ã£o (t) e a amplitude real do sinal no modo 2,2
        t = waveform.t
        h22_complexo = waveform.data[:, idx_22]
        
        # Como o sinal Ã© complexo (tem parte Real e ImaginÃ¡ria para modelar fase e amplitude),
        # nÃ³s pegamos o valor absoluto (ou a parte Real) para o nosso primeiro teste de ML.
        h22_real = np.real(h22_complexo)
        
        print("âœ… Dados extraÃ­dos com sucesso!")
        print(f"-> Total de pontos no tempo: {len(t)}")
        print(f"-> DuraÃ§Ã£o da simulaÃ§Ã£o: de t={t[0]:.2f} atÃ© t={t[-1]:.2f}")
        
        # 4. SALVANDO OS DADOS PARA A NOSSA REDE
        # Convertendo para Tensores do PyTorch
        tensor_t = torch.tensor(t, dtype=torch.float32)
        tensor_h22 = torch.tensor(h22_real, dtype=torch.float32)
        
        caminho_t = os.path.join(pasta_saida, f"{sxs_id.replace(':','_')}_tempo.pt")
        caminho_h22 = os.path.join(pasta_saida, f"{sxs_id.replace(':','_')}_h22.pt")
        
        torch.save(tensor_t, caminho_t)
        torch.save(tensor_h22, caminho_h22)
        print(f"ðŸ’¾ Tensores salvos na pasta: {pasta_saida}")
        
        # 5. VISUALIZAÃ‡ÃƒO (A Prova de que deu certo)
        print("ðŸ“Š Gerando grÃ¡fico do sinal astrofÃ­sico real...")
        plt.figure(figsize=(10, 4))
        plt.plot(t, h22_real, label=f"Modo $h_{{22}}$ ({sxs_id})", color="purple")
        plt.title(f"Sinal Real (Relatividade NumÃ©rica) - {sxs_id}")
        plt.xlabel("Tempo (Unidades GeomÃ©tricas)")
        plt.ylabel("Strain (h)")
        plt.grid(True)
        plt.legend()
        
        caminho_grafico = os.path.join(pasta_saida, f"{sxs_id.replace(':','_')}_plot.png")
        plt.savefig(caminho_grafico)
        print(f"ðŸ–¼ï¸ GrÃ¡fico salvo em: {caminho_grafico}")
        
    except Exception as e:
        print(f"âŒ Ocorreu um erro durante o download ou processamento: {e}")

if __name__ == "__main__":
    # SHAYENE: Este Ã© o ID da simulaÃ§Ã£o. O SXS:BBH:0305 Ã© a simulaÃ§Ã£o famosa 
    # do primeiro evento detectado pelo LIGO em 2015 (a colisÃ£o de buracos negros GW150914).
    # VocÃª pode trocar isso no futuro para baixar outras colisÃµes!
    baixar_e_processar_sxs_real(sxs_id="SXS:BBH:0305")


