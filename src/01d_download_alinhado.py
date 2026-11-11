"""
CosmicFNO - Extração e Alinhamento Físico de Simulações
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Este script baixa um conjunto de simulações do catálogo SXS e aplica
    o rigor científico necessário: ALINHAMENTO TEMPORAL.
    Em vez de cortar os "últimos N pontos" aleatoriamente, nós encontramos
    o pico de amplitude (o momento exato do Merger) e definimos esse ponto
    como t=0. Isso garante que a rede neural aprenda a física da onda,
    e não perca capacidade tentando adivinhar "quando" a colisão ocorre.
"""

import sxs
import numpy as np
import os
import torch

def baixar_e_alinhar_dataset(lista_ids, pasta_saida, janela_antes=4000, janela_depois=1000):
    os.makedirs(pasta_saida, exist_ok=True)
    
    dataset_tempo = []
    dataset_h22 = []
    dataset_parametros = [] # (q, spin1, spin2)
    
    print(f"[INFO] Iniciando download e alinhamento de {len(lista_ids)} simulacoes...")
    
    for sxs_id in lista_ids:
        print(f"\n-> Processando {sxs_id}...")
        try:
            # ignore_deprecation=True garante que pegamos a versão mais recente e corrigida
            sim = sxs.load(sxs_id, ignore_deprecation=True)
            
            q = sim.metadata.reference_mass_ratio
            spin1_z = sim.metadata.reference_dimensionless_spin1[2]
            spin2_z = sim.metadata.reference_dimensionless_spin2[2]
            
            waveform = sim.h
            idx_22 = waveform.index(2, 2)
            t = waveform.t
            h22_complexo = waveform.data[:, idx_22]
            
            # 1. Encontrar o pico de amplitude (O momento do Merger)
            amplitude = np.abs(h22_complexo)
            idx_pico = np.argmax(amplitude)
            
            # 2. Alinhamento Temporal (t=0 no pico)
            t_alinhado = t - t[idx_pico]
            
            # 3. Recorte da Janela (Garantindo que todas as ondas tenham o mesmo tamanho)
            # Pegamos 'janela_antes' pontos antes do pico e 'janela_depois' pontos depois
            inicio = idx_pico - janela_antes
            fim = idx_pico + janela_depois
            
            if inicio >= 0 and fim < len(t):
                t_recorte = t_alinhado[inicio:fim]
                h22_recorte = np.real(h22_complexo[inicio:fim])
                
                dataset_tempo.append(torch.tensor(t_recorte, dtype=torch.float32))
                dataset_h22.append(torch.tensor(h22_recorte, dtype=torch.float32))
                dataset_parametros.append(torch.tensor([q, spin1_z, spin2_z], dtype=torch.float32))
                print(f"[OK] {sxs_id} alinhado com sucesso! (q={q:.2f})")
            else:
                print(f"[AVISO] {sxs_id} ignorado (sinal muito curto para a janela exigida)")
                
        except Exception as e:
            print(f"[ERRO] Falha ao processar {sxs_id}: {e}")
            
    print("\n[INFO] Salvando tensores do dataset alinhado...")
    torch.save(torch.stack(dataset_tempo), os.path.join(pasta_saida, "dataset_t_alinhado.pt"))
    torch.save(torch.stack(dataset_h22), os.path.join(pasta_saida, "dataset_h22_alinhado.pt"))
    torch.save(torch.stack(dataset_parametros), os.path.join(pasta_saida, "dataset_params_alinhado.pt"))
    
    print("[SUCESSO] Dataset alinhado criado com sucesso!")

if __name__ == "__main__":
    # Vamos usar as mesmas 5 simulações para manter a consistência
    simulacoes_alvo = [
        "SXS:BBH:0305", # q=1.22
        "SXS:BBH:0115", # q=1.0
        "SXS:BBH:0114", # q=1.5
        "SXS:BBH:0113", # q=2.0
        "SXS:BBH:0167"  # q=3.0
    ]
    
    pasta_destino = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dataset_sxs")
    baixar_e_alinhar_dataset(simulacoes_alvo, pasta_destino)

