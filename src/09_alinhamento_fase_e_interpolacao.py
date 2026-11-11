"""
CosmicFNO - Módulo 09: Alinhamento de Fase Orbital, Grade Física Uniforme e Interpolação
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Corrige duas falhas fundamentais de pré-processamento físico em Relatividade Numérica:
    1. Invariância de Fase Orbital (Phase Alignment):
       No catálogo SXS, cada simulação inicia com uma fase azimutal arbitrária phi_0.
       Em t = t_pico (Merger), rotacionamos o modo complexo h_22(t) por exp(-i * phi(t_pico))
       para garantir phi(t=0) = 0 em todas as simulações.
    2. Grade de Tempo Físico Comum (Uniform Time Grid Interpolation):
       Como os solucionadores numéricos usam passo de tempo adaptativo (dt variável),
       interpolamos todas as simulações em uma grade física estritamente idêntica
       t in [-330M, +90M] com N = 4096 pontos.
"""

import os
import sys
import sxs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import importlib.util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

fno_spec = importlib.util.spec_from_file_location(
    "02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py")
)
fno_mod = importlib.util.module_from_spec(fno_spec)
fno_spec.loader.exec_module(fno_mod)
CosmicFNO = fno_mod.CosmicFNO

metricas_spec = importlib.util.spec_from_file_location(
    "08_metricas_fisicas", os.path.join(os.path.dirname(__file__), "08_metricas_fisicas.py")
)
metricas_mod = importlib.util.module_from_spec(metricas_spec)
metricas_spec.loader.exec_module(metricas_mod)
relatorio_metricas = metricas_mod.relatorio_metricas


def construir_dataset_alinhado_tempo_e_fase(lista_ids, t_inicio=-330.0, t_fim=90.0, num_pontos=4096):
    """
    Carrega simulações SXS do cache local, alinha t=0 no pico do envelope |h_22|,
    rotaciona a fase para phi(0)=0 e interpola na grade uniforme [t_inicio, t_fim].
    """
    grade_t_comum = np.linspace(t_inicio, t_fim, num_pontos, dtype=np.float32)
    ondas_real = []
    parametros = []

    print("[INFO] Construindo dataset com Alinhamento de Tempo E Fase Orbital (phi_0 = 0)...")
    for sxs_id in lista_ids:
        sim = sxs.load(sxs_id, ignore_deprecation=True)
        q = float(sim.metadata.reference_mass_ratio)
        spin1_z = float(sim.metadata.reference_dimensionless_spin1[2])
        spin2_z = float(sim.metadata.reference_dimensionless_spin2[2])

        wf = sim.h
        idx_22 = wf.index(2, 2)
        t = np.asarray(wf.t, dtype=np.float64)
        h22 = np.asarray(wf.data[:, idx_22], dtype=np.complex128)

        # 1. Alinhamento temporal pelo pico da amplitude (Envelope |h_22|)
        env = np.abs(h22)
        idx_pico = int(np.argmax(env))
        t_shift = t - t[idx_pico]

        # 2. Alinhamento de Fase no Merger: h_alinhado(t) = h_22(t) * exp(-i * phi_pico)
        fase_pico = np.angle(h22[idx_pico])
        h22_fase_alinhada = h22 * np.exp(-1j * fase_pico)

        # 3. Interpolação cúbica/linear na grade física uniforme comum
        parte_real_interp = np.interp(grade_t_comum, t_shift, np.real(h22_fase_alinhada)).astype(np.float32)

        ondas_real.append(torch.tensor(parte_real_interp))
        parametros.append(torch.tensor([q, spin1_z, spin2_z], dtype=torch.float32))
        print(f"  -> {sxs_id}: q={q:.2f}, spin1_z={spin1_z:+.2f} | h_real(t=0) = {np.real(h22_fase_alinhada[idx_pico]):+.4f}")

    t_tensor = torch.tensor(grade_t_comum).unsqueeze(0).repeat(len(lista_ids), 1)
    h_tensor = torch.stack(ondas_real)
    p_tensor = torch.stack(parametros)
    return t_tensor, h_tensor, p_tensor


def rodar_experimento_alinhamento():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pasta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_sxs = os.path.join(pasta_base, "data", "raw", "dataset_sxs")

    simulacoes_alvo = [
        "SXS:BBH:0305",  # idx 0: q=1.22
        "SXS:BBH:0115",  # idx 1: q=1.07
        "SXS:BBH:0114",  # idx 2: q=8.00
        "SXS:BBH:0113",  # idx 3: q=5.00 (Sem spin!)
        "SXS:BBH:0167",  # idx 4: q=4.00 (Sem spin!)
    ]

    t_raw, h22_raw, params = construir_dataset_alinhado_tempo_e_fase(simulacoes_alvo)

    # Salvando o novo dataset fisicamente consistente
    torch.save(t_raw, os.path.join(pasta_sxs, "dataset_t_fase_alinhada.pt"))
    torch.save(h22_raw, os.path.join(pasta_sxs, "dataset_h22_fase_alinhada.pt"))
    torch.save(params, os.path.join(pasta_sxs, "dataset_params_fase_alinhada.pt"))

    num_sims, num_pontos = t_raw.shape
    q = params[:, 0].unsqueeze(1).repeat(1, num_pontos)
    spin1 = params[:, 1].unsqueeze(1).repeat(1, num_pontos)
    t_norm = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min())

    X = torch.stack([q, spin1, t_norm], dim=-1)
    Y = h22_raw.unsqueeze(-1)

    # Vamos testar tanto na simulação idx=3 (q=5.0, interpolada entre q=4.0 e q=8.0)
    # quanto na simulação idx=4 (q=4.0, interpolada entre q=1.22 e q=5.0)
    for idx_teste, nome_cenario in [(3, "Teste Hold-Out q=5.00"), (4, "Teste Hold-Out q=4.00")]:
        indices_treino = [i for i in range(num_sims) if i != idx_teste]
        X_treino = X[indices_treino].to(device)
        Y_treino = Y[indices_treino].to(device)
        X_teste = X[idx_teste:idx_teste+1].to(device)
        Y_teste = Y[idx_teste:idx_teste+1].to(device)

        torch.manual_seed(42)
        modelo = CosmicFNO(num_modos=32, largura=64).to(device)
        otimizador = optim.Adam(modelo.parameters(), lr=0.002)
        loss_fn = nn.MSELoss()

        for _ in range(250):
            modelo.train()
            otimizador.zero_grad()
            loss = loss_fn(modelo(X_treino), Y_treino)
            loss.backward()
            otimizador.step()

        modelo.eval()
        with torch.no_grad():
            pred_teste = modelo(X_teste).cpu().squeeze().numpy()
            ref_teste = Y_teste.cpu().squeeze().numpy()
            t_np = t_raw[idx_teste].numpy()

        metricas = relatorio_metricas(f"PÓS-ALINHAMENTO DE FASE | {nome_cenario}", ref_teste, pred_teste)

        if idx_teste == 3:
            # Plota a comparação impressionante Antes vs Depois para q=5.00
            plt.figure(figsize=(11, 5))
            plt.plot(t_np, ref_teste, label="Referência SXS (q=5.00, Fase Alinhada)", color="#4b0082", linewidth=2)
            plt.plot(t_np, pred_teste, label="Predição FNO Hold-Out (Inédito)", color="#ff8c00", linestyle="--", linewidth=2)
            plt.title(
                f"Hold-Out Após Alinhamento de Fase e Grade Comum (q=5.00)\n"
                f"MSE: {metricas['mse']:.6f} | Erro Rel. L2: {metricas['rel_l2']*100:.1f}% | "
                f"Overlap: {metricas['overlap']*100:.2f}% (Mismatch: {metricas['mismatch']:.4f})",
                fontweight="bold"
            )
            plt.xlabel("Tempo Físico Uniforme (t = 0 no Merger)")
            plt.ylabel("Strain Re(h_22)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            caminho_fig = os.path.join(pasta_sxs, "resultado_holdout_fase_alinhada.png")
            plt.savefig(caminho_fig, dpi=150)
            plt.close()
            print(f"[OK] Gráfico salvo em: {caminho_fig}")


if __name__ == "__main__":
    rodar_experimento_alinhamento()

