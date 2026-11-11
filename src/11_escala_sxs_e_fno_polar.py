"""
CosmicFNO - Modulo 11: Escalonamento do Catalogo SXS (3.302 Simulacoes) e
                       Inovacao Arquitetural Polar-CosmicFNO (Amplitude-Fase)
Autora: Shayene Figueredo (BSI/UFU)

Descricao:
    Este modulo eleva o projeto CosmicFNO ao nivel de artigo cientifico completo:
    1. Experimento em Escala Completa no Catalogo SXS v3.0.0 (3.302 Colisoes BBH):
       Treina (80% = 2.641 colisoes) e avalia em Teste Cego (20% = 661 colisoes ineditas)
       um preditor neural profundo para estimar a Massa Final (M_f), o Spin Final (chi_f)
       e a Velocidade do Chute Relativistico (v_kick em km/s, ate >3.000 km/s).
    2. Expansao do Dataset de Formas de Onda h_22(t) (Grade Densa de Massas q in [1.0, 8.0]):
       Baixa e alinha temporalmente e em fase um conjunto expandido de simulacoes SXS.
    3. Inovacao Matematica — Polar-CosmicFNO (Decomposicao Espectral Amplitude-Fase):
       Em vez de interpolar diretamente a onda oscilatoria Re(h_22(t)) (que sofre
       interferencia destrutiva de fase entre massas distintas), o Polar-CosmicFNO
       opera no espaco das funcoes monotonicas suaves de Envoltoria de Amplitude A(t)
       e Fase Orbital Acumulada phi(t), reconstruindo h_22(t) = A(t) * cos(phi(t)).
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
CamadaFourier1D = fno_mod.CamadaFourier1D
CosmicFNO = fno_mod.CosmicFNO

met_spec = importlib.util.spec_from_file_location(
    "08_metricas_fisicas", os.path.join(os.path.dirname(__file__), "08_metricas_fisicas.py")
)
met_mod = importlib.util.module_from_spec(met_spec)
met_spec.loader.exec_module(met_mod)

VELOCIDADE_LUZ_KMS = 299792.458


# =====================================================================
# 1. INOVACAO ARQUITETURAL: POLAR-COSMICFNO (OPERADOR AMPLITUDE-FASE)
# =====================================================================
class PolarCosmicFNO(nn.Module):
    """
    Operador Neural de Fourier em Coordenadas Polares Fisicas:
        Mapeia (q(t), chi_1z(t), t_norm) -> (A(t), phi(t))
        e reconstroi fisicamente o modo quadrupolar:
        Re(h_22(t)) = A(t) * cos(phi(t))
    """
    def __init__(self, num_modos=32, largura=64):
        super().__init__()
        self.elevar = nn.Linear(3, largura)
        self.fourier0 = CamadaFourier1D(largura, largura, num_modos)
        self.fourier1 = CamadaFourier1D(largura, largura, num_modos)
        self.fourier2 = CamadaFourier1D(largura, largura, num_modos)
        self.fourier3 = CamadaFourier1D(largura, largura, num_modos)

        self.local0 = nn.Conv1d(largura, largura, 1)
        self.local1 = nn.Conv1d(largura, largura, 1)
        self.local2 = nn.Conv1d(largura, largura, 1)
        self.local3 = nn.Conv1d(largura, largura, 1)

        self.projetar1 = nn.Linear(largura, 128)
        # Saida dupla: Canal 0 = Amplitude normalizada A(t), Canal 1 = Fase normalizada phi(t)
        self.projetar2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.elevar(x)
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.gelu(self.fourier0(x) + self.local0(x))
        x = torch.nn.functional.gelu(self.fourier1(x) + self.local1(x))
        x = torch.nn.functional.gelu(self.fourier2(x) + self.local2(x))
        x = torch.nn.functional.gelu(self.fourier3(x) + self.local3(x))
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.gelu(self.projetar1(x))
        return self.projetar2(x)


# =====================================================================
# 2. PREDITOR EM ESCALA COMPLETA (3.302 COLISOES DO CATALOGO SXS)
# =====================================================================
class SXS3300RemnantNet(nn.Module):
    """
    Rede Neural Profunda com Conexoes Residuais treinada nas >3.000 simulacoes
    de Relatividade Numerica do catalogo SXS para prever:
        - Massa Remanescente (M_f / M)
        - Spin Final Magnitude (chi_f)
        - Velocidade do Chute Relativistico (v_kick em km/s)
    a partir de 11 parametros fisicos pre-colisao (q, eta, spins 3D de ambos os BNs).
    """
    def __init__(self, in_features=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


def treinar_catalogo_sxs_3300(pasta_sxs, pasta_modelos):
    print("\n=====================================================================")
    print("PARTE 1: TREINAMENTO EM ESCALA REAL NAS >3.000 COLISOES DO CATALOGO SXS")
    print("=====================================================================")

    sims = sxs.load("simulations")
    df = sims.dataframe

    # Filtra todas as simulacoes de Buracos Negros Binarios (SXS:BBH) validas e com dados completos
    colunas_entrada = [
        "reference_mass_ratio",
        "reference_chi_eff",
        "reference_chi1_perp",
        "reference_chi2_perp",
        "reference_dimensionless_spin1_x",
        "reference_dimensionless_spin1_y",
        "reference_dimensionless_spin1_z",
        "reference_dimensionless_spin2_x",
        "reference_dimensionless_spin2_y",
        "reference_dimensionless_spin2_z",
    ]
    colunas_saida = [
        "remnant_mass",
        "remnant_dimensionless_spin_mag",
        "remnant_velocity_mag",
    ]

    bbh = df[
        df.index.str.startswith("SXS:BBH:")
        & (~df["deprecated"])
        & (df["reference_eccentricity"] < 0.01)
        & (df["reference_mass_ratio"] <= 10.0)
    ].dropna(subset=colunas_entrada + colunas_saida)

    print(f"[INFO] Total de Simulacoes SXS BBH extraidas e validadas: {len(bbh)} colisoes!")

    X_np = bbh[colunas_entrada].values.astype(np.float32)
    q_col = X_np[:, 0:1]
    eta_col = q_col / ((1.0 + q_col) ** 2)  # Razao de massa simetrica eta in (0, 0.25]
    X_full = np.hstack([X_np, eta_col])     # 11 features fisicas

    Y_np = bbh[colunas_saida].values.astype(np.float32)
    # Converte velocidade de unidades de c para milhares de km/s
    Y_np[:, 2] = Y_np[:, 2] * VELOCIDADE_LUZ_KMS / 1000.0

    # Divisao 80% Treino / 20% Teste Cego (Hold-Out)
    np.random.seed(42)
    indices = np.random.permutation(len(X_full))
    n_treino = int(0.80 * len(X_full))
    idx_tr, idx_te = indices[:n_treino], indices[n_treino:]

    # Normalizacao Z-score calculada APENAS no treino (Zero Data Leakage!)
    x_mean = X_full[idx_tr].mean(axis=0, keepdims=True)
    x_std = X_full[idx_tr].std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_full - x_mean) / x_std

    X_tr = torch.tensor(X_norm[idx_tr], dtype=torch.float32)
    Y_tr = torch.tensor(Y_np[idx_tr], dtype=torch.float32)
    X_te = torch.tensor(X_norm[idx_te], dtype=torch.float32)
    Y_te = torch.tensor(Y_np[idx_te], dtype=torch.float32)

    print(f"[INFO] Divisao Hold-Out: {len(X_tr)} colisoes de TREINO | {len(X_te)} colisoes INEDITAS de TESTE CEGO")

    torch.manual_seed(42)
    modelo_3300 = SXS3300RemnantNet(in_features=11)
    otimizador = optim.Adam(modelo_3300.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=800)
    loss_fn = nn.SmoothL1Loss()

    for epoca in range(800):
        modelo_3300.train()
        otimizador.zero_grad()
        pred = modelo_3300(X_tr)
        # Pesos balanceados para as 3 grandezas fisicas
        perda = (
            10.0 * loss_fn(pred[:, 0], Y_tr[:, 0])
            + 5.0 * loss_fn(pred[:, 1], Y_tr[:, 1])
            + 2.0 * loss_fn(pred[:, 2], Y_tr[:, 2])
        )
        perda.backward()
        otimizador.step()
        scheduler.step()

    modelo_3300.eval()
    with torch.no_grad():
        pred_te = modelo_3300(X_te).numpy()
        real_te = Y_te.numpy()

    # Calculo de Metricas no Conjunto de Teste Cego (600+ colisoes nunca vistas!)
    mae_massa = float(np.mean(np.abs(pred_te[:, 0] - real_te[:, 0])))
    erro_rel_massa = float(np.mean(np.abs(pred_te[:, 0] - real_te[:, 0]) / real_te[:, 0]) * 100.0)

    mae_spin = float(np.mean(np.abs(pred_te[:, 1] - real_te[:, 1])))
    r2_spin = float(1.0 - np.sum((real_te[:, 1] - pred_te[:, 1])**2) / np.sum((real_te[:, 1] - np.mean(real_te[:, 1]))**2))

    mae_kick_kms = float(np.mean(np.abs(pred_te[:, 2] - real_te[:, 2])) * 1000.0)
    r2_kick = float(1.0 - np.sum((real_te[:, 2] - pred_te[:, 2])**2) / np.sum((real_te[:, 2] - np.mean(real_te[:, 2]))**2))

    # Acuracia na classificacao de ejecao de aglomerados (> 50 km/s) e galaxias anas (> 150 km/s)
    acc_aglomerado = float(np.mean((pred_te[:, 2]*1000 > 50.0) == (real_te[:, 2]*1000 > 50.0)) * 100.0)
    acc_galaxia_ana = float(np.mean((pred_te[:, 2]*1000 > 150.0) == (real_te[:, 2]*1000 > 150.0)) * 100.0)

    print("\n--- RESULTADOS NO TESTE CEGO ({} COLISOES INEDITAS DO SXS) ---".format(len(X_te)))
    print(f"  1. Massa do Buraco Negro Final (M_f) : Erro Medio = {mae_massa:.5f} M (Erro Relativo: {erro_rel_massa:.2f}%)")
    print(f"  2. Spin Final do Buraco Negro (chi_f): MAE = {mae_spin:.4f} | R² = {r2_spin*100:.2f}%")
    print(f"  3. Velocidade do Chute (v_kick)      : MAE = {mae_kick_kms:.1f} km/s | R² = {r2_kick*100:.2f}%")
    print(f"  4. Acuracia de Ejecao Astrofisica    : {acc_aglomerado:.2f}% (Aglomerados >50 km/s) | {acc_galaxia_ana:.2f}% (Galaxias Anas >150 km/s)")

    # Salva o modelo e estatisticas de normalizacao para uso no Dashboard
    torch.save({
        "state_dict": modelo_3300.state_dict(),
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_std": torch.tensor(x_std, dtype=torch.float32),
        "n_total": len(X_full),
        "n_treino": len(X_tr),
        "n_teste": len(X_te),
        "erro_rel_massa": erro_rel_massa,
        "r2_spin": r2_spin,
        "r2_kick": r2_kick,
        "mae_kick_kms": mae_kick_kms,
        "acc_aglomerado": acc_aglomerado,
        "acc_galaxia_ana": acc_galaxia_ana,
        "real_te": torch.tensor(real_te, dtype=torch.float32),
        "pred_te": torch.tensor(pred_te, dtype=torch.float32),
    }, os.path.join(pasta_modelos, "sxs_3300_remnant_model.pt"))


# =====================================================================
# 3. EXPANSAO DE ONDAS SXS E COMPARATIVO FNO DIRETO VS POLAR-COSMICFNO
# =====================================================================
def expandir_ondas_e_treinar_polar_fno(pasta_sxs, pasta_modelos):
    print("\n=====================================================================")
    print("PARTE 2: EXPANSAO DE FORMAS DE ONDA E INOVACAO POLAR-COSMICFNO (A, phi)")
    print("=====================================================================")

    # Selecionamos 12 simulacoes cobrindo sistematicamente q de 1.0 a 8.0
    lista_ids_expandida = [
        "SXS:BBH:0304",  # q = 1.00
        "SXS:BBH:0115",  # q = 1.07
        "SXS:BBH:0198",  # q = 1.20
        "SXS:BBH:0305",  # q = 1.22
        "SXS:BBH:0593",  # q = 1.50
        "SXS:BBH:1354",  # q = 1.83
        "SXS:BBH:1222",  # q = 2.00
        "SXS:BBH:0201",  # q = 2.32
        "SXS:BBH:0292",  # q = 3.00
        "SXS:BBH:0167",  # q = 4.00
        "SXS:BBH:0113",  # q = 5.00
        "SXS:BBH:0192",  # q = 6.58
        "SXS:BBH:0188",  # q = 7.19
        "SXS:BBH:0114",  # q = 8.00
    ]

    grade_t = np.linspace(-330.0, 90.0, 4096, dtype=np.float32)
    ondas_re = []
    ondas_amp = []
    ondas_fase = []
    params_lista = []
    ids_validos = []

    for sxs_id in lista_ids_expandida:
        try:
            sim = sxs.load(sxs_id, ignore_deprecation=True)
            q = float(sim.metadata.reference_mass_ratio)
            s1z = float(sim.metadata.reference_dimensionless_spin1[2])
            s2z = float(sim.metadata.reference_dimensionless_spin2[2])

            wf = sim.h
            idx_22 = wf.index(2, 2)
            t = np.asarray(wf.t, dtype=np.float64)
            h22 = np.asarray(wf.data[:, idx_22], dtype=np.complex128)

            env = np.abs(h22)
            idx_pico = int(np.argmax(env))
            t_shift = t - t[idx_pico]

            if t_shift[0] <= -330.0 and t_shift[-1] >= 90.0:
                fase_pico = np.angle(h22[idx_pico])
                h22_alinhada = h22 * np.exp(-1j * fase_pico)

                # Decomposicao Polar: Amplitude A(t) e Fase Acumulada Desdobrada phi(t)
                amp_t = np.abs(h22_alinhada)
                fase_t = np.unwrap(np.angle(h22_alinhada))
                # Garante que em t=0 (idx_pico) a fase desdobrada seja exatamente 0.0
                fase_t = fase_t - fase_t[idx_pico]

                re_interp = np.interp(grade_t, t_shift, np.real(h22_alinhada)).astype(np.float32)
                amp_interp = np.interp(grade_t, t_shift, amp_t).astype(np.float32)
                fase_interp = np.interp(grade_t, t_shift, fase_t).astype(np.float32)

                ondas_re.append(torch.tensor(re_interp))
                ondas_amp.append(torch.tensor(amp_interp))
                ondas_fase.append(torch.tensor(fase_interp))
                params_lista.append(torch.tensor([q, s1z, s2z], dtype=torch.float32))
                ids_validos.append(sxs_id)
                print(f"  [OK] {sxs_id} carregada e decomposta em (A, phi) | q = {q:.2f}, spin1_z = {s1z:+.2f}")
        except Exception as e:
            print(f"  [AVISO] {sxs_id} pulada: {e}")

    t_tensor = torch.tensor(grade_t).unsqueeze(0).repeat(len(ids_validos), 1)
    h_re_tensor = torch.stack(ondas_re)
    h_amp_tensor = torch.stack(ondas_amp)
    h_fase_tensor = torch.stack(ondas_fase)
    p_tensor = torch.stack(params_lista)

    torch.save({
        "ids": ids_validos,
        "t": t_tensor,
        "h_re": h_re_tensor,
        "h_amp": h_amp_tensor,
        "h_fase": h_fase_tensor,
        "params": p_tensor,
    }, os.path.join(pasta_sxs, "dataset_polar_expandido.pt"))

    # Hold-Out Rigoroso: Vamos separar 3 simulacoes inteiras de massas diferentes para TESTE CEGO
    # Ex: q=1.50 (SXS:BBH:0593), q=2.32 (SXS:BBH:0201) e q=5.00 (SXS:BBH:0113)
    ids_teste_alvo = {"SXS:BBH:0593", "SXS:BBH:0201", "SXS:BBH:0113"}
    idx_teste = [i for i, sid in enumerate(ids_validos) if sid in ids_teste_alvo]
    idx_treino = [i for i in range(len(ids_validos)) if i not in idx_teste]

    num_sims, num_pts = t_tensor.shape
    q_feat = p_tensor[:, 0].unsqueeze(1).repeat(1, num_pts)
    s_feat = p_tensor[:, 1].unsqueeze(1).repeat(1, num_pts)
    t_norm = (t_tensor - t_tensor.min()) / (t_tensor.max() - t_tensor.min())

    X = torch.stack([q_feat, s_feat, t_norm], dim=-1)
    Y_re = h_re_tensor.unsqueeze(-1)

    # Escala da fase (para ficar na ordem de ~[-1, 1] durante o treinamento)
    escala_fase = 100.0
    Y_polar = torch.stack([h_amp_tensor, h_fase_tensor / escala_fase], dim=-1)

    # 1. Treina o FNO Direto (tradicional, prevendo Re(h_22)) nas simulacoes de treino
    torch.manual_seed(42)
    fno_direto = CosmicFNO(num_modos=32, largura=64)
    opt_dir = optim.Adam(fno_direto.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()

    for _ in range(300):
        opt_dir.zero_grad()
        loss = loss_fn(fno_direto(X[idx_treino]), Y_re[idx_treino])
        loss.backward()
        opt_dir.step()

    # 2. Treina a nossa inovacao: Polar-CosmicFNO (prevendo Amplitude A(t) e Fase phi(t))
    torch.manual_seed(42)
    fno_polar = PolarCosmicFNO(num_modos=32, largura=64)
    opt_pol = optim.Adam(fno_polar.parameters(), lr=0.002)

    for _ in range(300):
        opt_pol.zero_grad()
        saida_pol = fno_polar(X[idx_treino])
        loss = loss_fn(saida_pol, Y_polar[idx_treino])
        loss.backward()
        opt_pol.step()

    fno_direto.eval()
    fno_polar.eval()

    print("\n--- COMPARATIVO HOLD-OUT CEGO: FNO DIRETO vs POLAR-COSMICFNO (INOVACAO) ---")
    resultados_comparativos = []
    with torch.no_grad():
        for i in idx_teste:
            sid = ids_validos[i]
            qv = p_tensor[i, 0].item()
            ref_onda = h_re_tensor[i].numpy()

            pred_dir = fno_direto(X[i:i+1]).squeeze().numpy()

            saida_p = fno_polar(X[i:i+1]).squeeze().numpy()
            amp_pred = saida_p[:, 0]
            fase_pred = saida_p[:, 1] * escala_fase
            pred_polar = amp_pred * np.cos(fase_pred)

            ov_dir, mis_dir = met_mod.calcular_overlap_e_mismatch(ref_onda, pred_dir)
            l2_dir = met_mod.calcular_erro_relativo_l2(ref_onda, pred_dir)

            ov_pol, mis_pol = met_mod.calcular_overlap_e_mismatch(ref_onda, pred_polar)
            l2_pol = met_mod.calcular_erro_relativo_l2(ref_onda, pred_polar)

            print(f"\nSimulacao Inedita de Teste: {sid} (q = {qv:.2f})")
            print(f"  [FNO Direto]       Overlap: {ov_dir*100:.2f}% | Mismatch: {mis_dir:.5f} | Erro Rel. L2: {l2_dir*100:.2f}%")
            print(f"  [Polar-CosmicFNO]  Overlap: {ov_pol*100:.2f}% | Mismatch: {mis_pol:.5f} | Erro Rel. L2: {l2_pol*100:.2f}%")

            resultados_comparativos.append({
                "id": sid, "q": qv,
                "ov_dir": ov_dir, "mis_dir": mis_dir, "l2_dir": l2_dir,
                "ov_pol": ov_pol, "mis_pol": mis_pol, "l2_pol": l2_pol,
                "ref": ref_onda, "pred_dir": pred_dir, "pred_polar": pred_polar
            })

    torch.save({
        "fno_direto": fno_direto.state_dict(),
        "fno_polar": fno_polar.state_dict(),
        "escala_fase": escala_fase,
        "resultados_teste": resultados_comparativos,
    }, os.path.join(pasta_modelos, "polar_fno_benchmark.pt"))


if __name__ == "__main__":
    pasta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_sxs = os.path.join(pasta_base, "data", "raw", "dataset_sxs")
    pasta_modelos = os.path.join(pasta_base, "modelos_salvos")
    os.makedirs(pasta_sxs, exist_ok=True)
    os.makedirs(pasta_modelos, exist_ok=True)

    treinar_catalogo_sxs_3300(pasta_sxs, pasta_modelos)
    expandir_ondas_e_treinar_polar_fno(pasta_sxs, pasta_modelos)



def gerar_grafico_benchmark_14sims():
    pasta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench = torch.load(os.path.join(pasta_base, "modelos_salvos", "polar_fno_benchmark.pt"), weights_only=False)
    res = bench["resultados_teste"]
    t = np.linspace(-330.0, 90.0, 4096)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, r in zip(axes, res):
        sid = r["id"]
        qv = r["q"]
        ov_d = r["ov_dir"] * 100.0
        mis_d = r["mis_dir"]
        ov_p = r["ov_pol"] * 100.0
        ax.plot(t, r["ref"], color="#081226", lw=2.3, label=f"Referencia SXS ({sid}, q={qv:.2f})")
        ax.plot(t, r["pred_dir"], color="#2563eb", lw=1.9, ls="--", label=f"CosmicFNO Direto (Overlap: {ov_d:.2f}%, Mismatch: {mis_d:.5f})")
        ax.plot(t, r["pred_polar"], color="#06b6d4", lw=1.5, ls=":", label=f"Polar-CosmicFNO (Overlap: {ov_p:.2f}%)")
        ax.set_ylabel("Strain Re(h_22)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].set_title("Validacao Hold-Out Cega em 3 Simulacoes Ineditas (Treino: 11 Simulacoes SXS Alinhadas)", fontweight="bold")
    axes[-1].set_xlabel("Tempo Fisico Alinhado pelo Merger (t = 0)")
    plt.tight_layout()
    caminho_out = os.path.join(pasta_base, "docs", "assets", "benchmark_14sims_holdout.png")
    plt.savefig(caminho_out, dpi=160)
    plt.close()
    print(f"[OK] Figura salva em: {caminho_out}")
