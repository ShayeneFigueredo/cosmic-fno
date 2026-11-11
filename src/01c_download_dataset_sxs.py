"""
CosmicFNO - Extração de Múltiplas Simulações (Dataset Real)
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Este script baixa um conjunto (batch) de várias simulações de colisões de
    buracos negros do catálogo SXS. Precisamos de vários exemplos com massas 
    e spins diferentes para que a nossa rede neural (FNO) aprenda a generalizar,
    e não apenas decore uma única onda.
"""

import sxs
import numpy as np
import os
import torch

def baixar_dataset_sxs(lista_ids, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)
    
    dataset_tempo = []
    dataset_h22 = []
    dataset_parametros = [] # (q, spin1, spin2)
    
    print(f"[INFO] Iniciando download de {len(lista_ids)} simulacoes...")
    
    for sxs_id in lista_ids:
        print(f"\n-> Processando {sxs_id}...")
        try:
            sim = sxs.load(sxs_id, ignore_deprecation=True)
            
            # Puxando os metadados (Razão de massa q, e spins)
            # Para simplificar, pegamos a magnitude do spin no eixo z
            q = sim.metadata.reference_mass_ratio
            spin1_z = sim.metadata.reference_dimensionless_spin1[2]
            spin2_z = sim.metadata.reference_dimensionless_spin2[2]
            
            # Puxando a onda
            waveform = sim.h
            idx_22 = waveform.index(2, 2)
            t = waveform.t
            h22_real = np.real(waveform.data[:, idx_22])
            
            # Para redes neurais, é bom que todas as ondas tenham o mesmo tamanho
            # Vamos padronizar pegando os últimos 8192 pontos (onde ocorre a colisão)
            if len(t) >= 8192:
                t = t[-8192:]
                h22_real = h22_real[-8192:]
                
                dataset_tempo.append(torch.tensor(t, dtype=torch.float32))
                dataset_h22.append(torch.tensor(h22_real, dtype=torch.float32))
                dataset_parametros.append(torch.tensor([q, spin1_z, spin2_z], dtype=torch.float32))
                print(f"[OK] {sxs_id} extraído com sucesso! (q={q:.2f})")
            else:
                print(f"[AVISO] {sxs_id} ignorado (sinal muito curto: {len(t)} pontos)")
                
        except Exception as e:
            print(f"[ERRO] Falha ao baixar {sxs_id}: {e}")
            
    # Salvando o dataset consolidado
    print("\n[INFO] Salvando tensores do dataset...")
    torch.save(torch.stack(dataset_tempo), os.path.join(pasta_saida, "dataset_t.pt"))
    torch.save(torch.stack(dataset_h22), os.path.join(pasta_saida, "dataset_h22.pt"))
    torch.save(torch.stack(dataset_parametros), os.path.join(pasta_saida, "dataset_params.pt"))
    
    print("[SUCESSO] Dataset criado com sucesso!")

if __name__ == "__main__":
    # Lista de simulações com diferentes razões de massa (q)
    # CUIDADO: Começamos com apenas 5 para não sobrecarregar sua internet e HD!
    simulacoes_alvo = [
        "SXS:BBH:0305", # GW150914 (q=1.22)
        "SXS:BBH:0115", # q=1.0
        "SXS:BBH:0114", # q=1.5
        "SXS:BBH:0113", # q=2.0
        "SXS:BBH:0167"  # q=3.0
    ]
    
    pasta_destino = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dataset_sxs")
    baixar_dataset_sxs(simulacoes_alvo, pasta_destino)

