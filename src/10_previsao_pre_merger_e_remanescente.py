"""
CosmicFNO - Modulo 10: Previsao Antecipada da Colisao (Pre-Merger Early Warning)
                       e Predicao do Buraco Negro Remanescente (Kick Relativistico)
Autora: Shayene Figueredo (BSI/UFU)

Descricao:
    Implementa duas aplicacoes preditivas de alto impacto astrofisico:
    1. Pre-Merger Early-Warning Forecaster (FNO):
       A rede neural observa APENAS a fase inicial da espiral (Inspiral, ex: t < -50M),
       com o futuro mascarado (zerado), e preve a forma de onda completa da colisao
       futura (Merger em t=0 e Ringdown em t > 0) antes do impacto ocorrer.
    2. Predicao do Buraco Negro Remanescente e Chute Gravitacional (Recoil Kick):
       Extrai dos metadados oficiais do catalogo SXS (Relatividade Numerica) e preve
       a partir da onda gravitacional:
       - Massa Final do Buraco Negro (M_f) e Energia Irradiada em Ondas (E_rad = Delta_m * c^2)
       - Spin Final Adimensional (chi_f)
       - Velocidade do Chute Relativistico (v_kick em km/s), determinando se o buraco negro
         recem-nascido escapa da gravidade de aglomerados estelares ou galaxias.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import importlib.util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

fno_spec = importlib.util.spec_from_file_location(
    "02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py")
)
fno_mod = importlib.util.module_from_spec(fno_spec)
fno_spec.loader.exec_module(fno_mod)
CosmicFNO = fno_mod.CosmicFNO

met_spec = importlib.util.spec_from_file_location(
    "08_metricas_fisicas", os.path.join(os.path.dirname(__file__), "08_metricas_fisicas.py")
)
met_mod = importlib.util.module_from_spec(met_spec)
met_spec.loader.exec_module(met_mod)

VELOCIDADE_LUZ_KMS = 299792.458  # km/s


class RemnantPredictorSpectral(nn.Module):
    """
    Rede Neural Espectral que analisa a forma de onda gravitacional completa h_22(t)
    (via espectro de Fourier + caracteristicas globais de assimetria) para prever:
        [0] Massa Final Remanescente (M_f / M_total)
        [1] Spin Final Adimensional (chi_f)
        [2] Velocidade do Chute Relativistico (v_kick em centenas de km/s)
    """
    def __init__(self, num_modos_fft=64):
        super().__init__()
        self.num_modos_fft = num_modos_fft
        # Entrada: 64 modos de amplitude FFT + 3 estatisticas fisicas da onda (pico, energia, assimetria)
        self.rede = nn.Sequential(
            nn.Linear(num_modos_fft + 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )

    def forward(self, onda_h22, q_est):
        # onda_h22: [Batch, N_pontos]
        fft_mag = torch.abs(torch.fft.rfft(onda_h22, dim=-1))[:, :self.num_modos_fft] / onda_h22.shape[-1]
        pico_max = onda_h22.abs().max(dim=-1, keepdim=True).values
        energia = torch.mean(onda_h22 ** 2, dim=-1, keepdim=True)
        eta = q_est / ((1.0 + q_est) ** 2)  # Razao de massa simetrica
        features = torch.cat([fft_mag, pico_max, energia, eta], dim=-1)
        return self.rede(features)


def extrair_metadados_remanescentes_sxs(lista_ids):
    """
    Extrai do catalogo SXS os dados reais do buraco negro formado apos a colisao:
    - remnant_mass (fracao da massa inicial total M)
    - remnant_spin_z (spin adimensional final)
    - v_kick_kms (velocidade de recuo / chute relativistico em km/s)
    """
    import sxs
    dados_remanescentes = []
    print("[INFO] Extraindo metadados reais pos-colisao (Remanescente & Kick) do catalogo SXS...")

    for sxs_id in lista_ids:
        sim = sxs.load(sxs_id, ignore_deprecation=True)
        meta = sim.metadata

        q = float(meta.reference_mass_ratio)
        m_rem = float(meta.remnant_mass)
        spin_rem_vec = np.asarray(meta.remnant_dimensionless_spin, dtype=np.float64)
        spin_rem = float(np.linalg.norm(spin_rem_vec))
        vel_vec = np.asarray(meta.remnant_velocity, dtype=np.float64)
        v_kick_kms = float(np.linalg.norm(vel_vec) * VELOCIDADE_LUZ_KMS)
        energia_irradiada_pct = (1.0 - m_rem) * 100.0

        dados_remanescentes.append([m_rem, spin_rem, v_kick_kms, energia_irradiada_pct])
        print(
            f"  -> {sxs_id} (q={q:.2f}): Massa Final = {m_rem:.4f}M "
            f"(Energia Irradiada = {energia_irradiada_pct:.2f}%), "
            f"Spin Final = {spin_rem:.4f}, Chute (Kick) = {v_kick_kms:.1f} km/s"
        )

    return torch.tensor(dados_remanescentes, dtype=torch.float32)


def treinar_previsor_pre_merger_e_remanescente():
    device = torch.device("cpu")
    pasta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_sxs = os.path.join(pasta_base, "data", "raw", "dataset_sxs")
    pasta_modelos = os.path.join(pasta_base, "modelos_salvos")
    os.makedirs(pasta_modelos, exist_ok=True)

    simulacoes_alvo = [
        "SXS:BBH:0305",  # q=1.22
        "SXS:BBH:0115",  # q=1.07
        "SXS:BBH:0114",  # q=8.00
        "SXS:BBH:0113",  # q=5.00
        "SXS:BBH:0167",  # q=4.00
    ]

    # 1. Extrai e salva metadados reais do buraco negro final
    rem_tensor = extrair_metadados_remanescentes_sxs(simulacoes_alvo)
    torch.save(rem_tensor, os.path.join(pasta_sxs, "dataset_remanescente_sxs.pt"))

    # 2. Carrega ondas com fase alinhada (t in [-330M, +90M])
    t_fase = torch.load(os.path.join(pasta_sxs, "dataset_t_fase_alinhada.pt"), map_location=device)
    h_fase = torch.load(os.path.join(pasta_sxs, "dataset_h22_fase_alinhada.pt"), map_location=device)
    p_fase = torch.load(os.path.join(pasta_sxs, "dataset_params_fase_alinhada.pt"), map_location=device)

    num_sims, num_pontos = t_fase.shape
    t_unico = t_fase[0]

    # =================================================================
    # PARTE A: TREINAMENTO DO FNO PRE-MERGER (EARLY WARNING FORECASTER)
    # =================================================================
    # A rede recebe nos canais de entrada:
    # Canal 0: Onda parcial observada APENAS antes do corte (t < -50M), com t >= -50M ZERADO!
    # Canal 1: Mascara binaria indicando regiao observada (1.0) vs regiao futura oculta (0.0)
    # Canal 2: Coordenada temporal normalizada + razao de massa estimada na espiral
    t_corte = -50.0
    mascara_inspiral = (t_unico <= t_corte).float()  # 1 no passado, 0 no futuro (Merger e Ringdown ocultos)

    onda_parcial = h_fase * mascara_inspiral.unsqueeze(0)
    q_canal = p_fase[:, 0].unsqueeze(1).repeat(1, num_pontos)
    t_norm = (t_fase - t_fase.min()) / (t_fase.max() - t_fase.min())

    # Entrada [Batch, Pontos, 3]: (onda_inspiral_observada, q_estimado, tempo_norm)
    X_pre = torch.stack([onda_parcial, q_canal, t_norm], dim=-1)
    Y_completo = h_fase.unsqueeze(-1)

    print("\n[INFO] Treinando CosmicFNO Pre-Merger Forecaster (Prevendo a colisao antes do impacto)...")
    torch.manual_seed(42)
    fno_forecaster = CosmicFNO(num_modos=32, largura=64).to(device)
    opt_fno = optim.Adam(fno_forecaster.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()

    for epoca in range(250):
        opt_fno.zero_grad()
        pred = fno_forecaster(X_pre)
        loss = loss_fn(pred, Y_completo)
        loss.backward()
        opt_fno.step()

    fno_forecaster.eval()
    torch.save(fno_forecaster.state_dict(), os.path.join(pasta_modelos, "fno_pre_merger_forecaster.pth"))

    with torch.no_grad():
        pred_futuro = fno_forecaster(X_pre).squeeze().numpy()
        ref_np = h_fase.numpy()
        mask_futuro = (t_unico.numpy() > t_corte)
        ov_fut, mis_fut = met_mod.calcular_overlap_e_mismatch(ref_np[4, mask_futuro], pred_futuro[4, mask_futuro])
        print(f"[OK] Previsao da Colisao Futura (t > {t_corte}M) em SXS:BBH:0167 | Overlap no Futuro Oculto: {ov_fut*100:.2f}%")

    # =================================================================
    # PARTE B: TREINAMENTO DO PREDITOR DO BURACO NEGRO REMANESCENTE
    # =================================================================
    print("\n[INFO] Treinando Rede Espectral de Predicao do Buraco Negro Remanescente e Kick Velocity...")
    torch.manual_seed(42)
    rem_net = RemnantPredictorSpectral(num_modos_fft=64).to(device)
    opt_rem = optim.Adam(rem_net.parameters(), lr=0.005)

    # Normaliza o chute em centenas de km/s para estabilidade numerica no treino
    alvos_rem = torch.stack([
        rem_tensor[:, 0],          # m_rem (~0.95)
        rem_tensor[:, 1],          # spin_rem (~0.68)
        rem_tensor[:, 2] / 1000.0  # v_kick em milhares de km/s
    ], dim=-1)

    q_vec = p_fase[:, 0:1]
    for _ in range(400):
        opt_rem.zero_grad()
        saida_rem = rem_net(h_fase, q_vec)
        perda_rem = loss_fn(saida_rem, alvos_rem)
        perda_rem.backward()
        opt_rem.step()

    rem_net.eval()
    torch.save(rem_net.state_dict(), os.path.join(pasta_modelos, "remnant_predictor.pth"))
    print("[SUCESSO] Modelos Preditivos (Pre-Merger Forecaster & Remnant Kick Predictor) salvos!")


if __name__ == "__main__":
    treinar_previsor_pre_merger_e_remanescente()

