"""
CosmicFNO - Módulo 08: Métricas Físicas e Avaliação de Operadores Neurais
Autora: Shayene Figueredo (BSI/UFU)

Descrição:
    Substitui a avaliação simplista baseada apenas em MSE (Mean Squared Error)
    pelas métricas padrão da literatura de Scientific Machine Learning (SciML)
    e Astrofísica de Ondas Gravitacionais (LIGO/Virgo/SXS):

    1. Erro Relativo L2 (Relative L2 Error - Li et al., 2020):
       Mede o erro adimensional normalizado pela energia do sinal de referência.
       Evita a falsa impressão de erro baixo causada pela pequena escala numérica do strain.

    2. Overlap (Fidelidade / Produto Interno Normalizado):
       Mede a correlação morfológica e de fase entre a forma de onda de referência
       e a forma de onda emulada no espaço de Hilbert L2.

    3. Mismatch (1 - Overlap):
       Métrica padrão em Relatividade Numérica para modelos substitutos (Surrogate Models).
       Mismatch = 0 indica concordância de fase e morfologia perfeita.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importando CosmicFNO (Módulo 02)
fno_spec = importlib.util.spec_from_file_location(
    "02_modelo_fno", os.path.join(os.path.dirname(__file__), "02_modelo_fno.py")
)
fno_mod = importlib.util.module_from_spec(fno_spec)
fno_spec.loader.exec_module(fno_mod)
CosmicFNO = fno_mod.CosmicFNO

# Importando BaselineCNN1D (Módulo 04)
cnn_spec = importlib.util.spec_from_file_location(
    "04_modelo_cnn_baseline", os.path.join(os.path.dirname(__file__), "04_modelo_cnn_baseline.py")
)
cnn_mod = importlib.util.module_from_spec(cnn_spec)
cnn_spec.loader.exec_module(cnn_mod)
BaselineCNN1D = cnn_mod.BaselineCNN1D


# =====================================================================
# 1. DEFINIÇÃO MATEMÁTICA DAS MÉTRICAS FÍSICAS E DE SCIML
# =====================================================================

def calcular_mse(h_ref: np.ndarray, h_pred: np.ndarray) -> float:
    """Erro Quadrático Médio (MSE) clássico."""
    return float(np.mean((h_ref - h_pred) ** 2))


def calcular_erro_relativo_l2(h_ref: np.ndarray, h_pred: np.ndarray) -> float:
    """
    Erro Relativo na norma L2 (padrão em artigos de Fourier Neural Operators):
        E_L2 = ||h_ref - h_pred||_2 / ||h_ref||_2
    Retorna a fração (ex: 0.05 = 5% de erro relativo).
    """
    norma_diferenca = np.linalg.norm(h_ref - h_pred, ord=2)
    norma_referencia = np.linalg.norm(h_ref, ord=2)
    return float(norma_diferenca / (norma_referencia + 1e-12))


def calcular_overlap_e_mismatch(h_ref: np.ndarray, h_pred: np.ndarray) -> tuple[float, float]:
    """
    Produto interno normalizado (Overlap) sob ruído branco plano e Mismatch:
        <h1 | h2> = integral h1(t) * h2(t) dt
        Overlap = <h_ref | h_pred> / sqrt(<h_ref|h_ref> * <h_pred|h_pred>)
        Mismatch = 1 - Overlap
    """
    prod_interno = np.sum(h_ref * h_pred)
    norma_ref = np.sqrt(np.sum(h_ref ** 2))
    norma_pred = np.sqrt(np.sum(h_pred ** 2))
    
    overlap = float(prod_interno / (norma_ref * norma_pred + 1e-12))
    # Limita numericamente a [-1, 1] por precisão de ponto flutuante
    overlap = max(-1.0, min(1.0, overlap))
    mismatch = 1.0 - overlap
    return overlap, mismatch


def calcular_mismatch_otimizado_tempo(h_ref: np.ndarray, h_pred: np.ndarray) -> tuple[float, float]:
    """
    Calcula o Overlap e Mismatch maximizados sobre deslocamentos de tempo (time-shift delta_t)
    usando o Teorema da Correlação Cruzada via FFT:
        max_{delta_t} <h_ref(t) | h_pred(t + delta_t)>
    Isso avalia se a morfologia da onda foi aprendida corretamente mesmo que haja
    um pequeno atraso temporal na predição.
    """
    f_ref = np.fft.rfft(h_ref)
    f_pred = np.fft.rfft(h_pred)
    corr_cruzada = np.fft.irfft(f_ref * np.conj(f_pred), n=len(h_ref))
    
    norma_ref = np.sqrt(np.sum(h_ref ** 2))
    norma_pred = np.sqrt(np.sum(h_pred ** 2))
    
    overlap_max = float(np.max(np.abs(corr_cruzada)) / (norma_ref * norma_pred + 1e-12))
    overlap_max = max(0.0, min(1.0, overlap_max))
    mismatch_min = 1.0 - overlap_max
    return overlap_max, mismatch_min


def relatorio_metricas(nome_experimento: str, h_ref: np.ndarray, h_pred: np.ndarray) -> dict:
    """Calcula e imprime todas as métricas para um par (referência, predição)."""
    mse = calcular_mse(h_ref, h_pred)
    rel_l2 = calcular_erro_relativo_l2(h_ref, h_pred)
    overlap, mismatch = calcular_overlap_e_mismatch(h_ref, h_pred)
    overlap_opt, mismatch_opt = calcular_mismatch_otimizado_tempo(h_ref, h_pred)
    
    print(f"\n--- {nome_experimento} ---")
    print(f"  MSE                     : {mse:.6f}")
    print(f"  Erro Relativo L2        : {rel_l2:.4f} ({rel_l2*100:.2f}%)")
    print(f"  Overlap (Alinhado)      : {overlap:.6f}")
    print(f"  Mismatch (Alinhado)     : {mismatch:.6f}")
    print(f"  Mismatch (Otim. Tempo)  : {mismatch_opt:.6f} (Overlap Max: {overlap_opt:.6f})")
    
    return {
        "nome": nome_experimento,
        "mse": mse,
        "rel_l2": rel_l2,
        "overlap": overlap,
        "mismatch": mismatch,
        "mismatch_opt": mismatch_opt,
    }


# =====================================================================
# 2. AVALIAÇÃO DOS EXPERIMENTOS PRELIMINARES SOB MÉTRICAS FÍSICAS
# =====================================================================

def avaliar_experimento_1_reconstrucao(device: torch.device, pasta_raw: str) -> tuple[dict, dict]:
    """
    Compara FNO vs CNN 1D na tarefa de reconstrução da simulação SXS:BBH:0305
    sob condições experimentais equivalentes (100 épocas, lr=0.005).
    """
    print("\n=====================================================================")
    print("AVALIAÇÃO 1: CAPACIDADE DE RECONSTRUÇÃO (FNO vs CNN 1D - SXS:BBH:0305)")
    print("=====================================================================")
    
    t_real = torch.load(os.path.join(pasta_raw, "SXS_BBH_0305_tempo.pt"))[-8192:]
    h_real = torch.load(os.path.join(pasta_raw, "SXS_BBH_0305_h22.pt"))[-8192:]
    num_pontos = len(t_real)
    
    q_tensor = torch.full((num_pontos,), 1.22)
    spin_tensor = torch.full((num_pontos,), 0.33)
    t_norm = (t_real - t_real.min()) / (t_real.max() - t_real.min())
    
    X_fno = torch.stack([q_tensor, spin_tensor, t_norm], dim=-1).unsqueeze(0).to(device)
    Y_fno = h_real.unsqueeze(0).unsqueeze(-1).to(device)
    
    X_cnn = X_fno.permute(0, 2, 1)
    Y_cnn = h_real.unsqueeze(0).unsqueeze(1).to(device)
    
    torch.manual_seed(42)
    modelo_fno = CosmicFNO(num_modos=32, largura=64).to(device)
    opt_fno = optim.Adam(modelo_fno.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    for _ in range(100):
        opt_fno.zero_grad()
        loss = loss_fn(modelo_fno(X_fno), Y_fno)
        loss.backward()
        opt_fno.step()
        
    torch.manual_seed(42)
    modelo_cnn = BaselineCNN1D().to(device)
    opt_cnn = optim.Adam(modelo_cnn.parameters(), lr=0.005)
    
    for _ in range(100):
        opt_cnn.zero_grad()
        loss = loss_fn(modelo_cnn(X_cnn), Y_cnn)
        loss.backward()
        opt_cnn.step()
        
    modelo_fno.eval()
    modelo_cnn.eval()
    with torch.no_grad():
        pred_fno = modelo_fno(X_fno).cpu().squeeze().numpy()
        pred_cnn = modelo_cnn(X_cnn).cpu().squeeze().numpy()
        h_ref = h_real.numpy()
        
    res_fno = relatorio_metricas("Exp 1 - Reconstrução FNO (SXS:BBH:0305)", h_ref, pred_fno)
    res_cnn = relatorio_metricas("Exp 1 - Reconstrução CNN 1D (SXS:BBH:0305)", h_ref, pred_cnn)
    return res_fno, res_cnn


def avaliar_experimento_2_holdout(device: torch.device, pasta_sxs: str) -> tuple[dict, dict]:
    """
    Avalia o Experimento Hold-Out (4 simulações de treino vs 1 simulação de teste inédita)
    revelando o comportamento das métricas físicas diante do overfitting.
    """
    print("\n=====================================================================")
    print("AVALIAÇÃO 2: GENERALIZAÇÃO HOLD-OUT (TREINO vs TESTE INÉDITO)")
    print("=====================================================================")
    
    t_raw = torch.load(os.path.join(pasta_sxs, "dataset_t_alinhado.pt"))
    h22_raw = torch.load(os.path.join(pasta_sxs, "dataset_h22_alinhado.pt"))
    params = torch.load(os.path.join(pasta_sxs, "dataset_params_alinhado.pt"))
    
    num_sims, num_pontos = t_raw.shape
    q = params[:, 0].unsqueeze(1).repeat(1, num_pontos)
    spin1 = params[:, 1].unsqueeze(1).repeat(1, num_pontos)
    
    t_min = t_raw.min(dim=1, keepdim=True).values
    t_max = t_raw.max(dim=1, keepdim=True).values
    t_norm = (t_raw - t_min) / (t_max - t_min)
    
    X = torch.stack([q, spin1, t_norm], dim=-1)
    Y = h22_raw.unsqueeze(-1)
    
    idx_teste = 3
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
        perda = loss_fn(modelo(X_treino), Y_treino)
        perda.backward()
        otimizador.step()
        
    modelo.eval()
    with torch.no_grad():
        pred_treino = modelo(X_treino).cpu().squeeze().numpy()
        ref_treino = Y_treino.cpu().squeeze().numpy()
        
        pred_teste = modelo(X_teste).cpu().squeeze().numpy()
        ref_teste = Y_teste.cpu().squeeze().numpy()
        
    # Média das métricas nas simulações de treino
    metricas_treino_lista = [
         {
             "mse": calcular_mse(ref_treino[i], pred_treino[i]),
             "rel_l2": calcular_erro_relativo_l2(ref_treino[i], pred_treino[i]),
             "overlap": calcular_overlap_e_mismatch(ref_treino[i], pred_treino[i])[0],
             "mismatch": calcular_overlap_e_mismatch(ref_treino[i], pred_treino[i])[1],
         }
         for i in range(len(indices_treino))
    ]
    
    media_treino = {
        "nome": "Exp 2 - Hold-Out (Média Treino - 4 sims)",
        "mse": float(np.mean([m["mse"] for m in metricas_treino_lista])),
        "rel_l2": float(np.mean([m["rel_l2"] for m in metricas_treino_lista])),
        "overlap": float(np.mean([m["overlap"] for m in metricas_treino_lista])),
        "mismatch": float(np.mean([m["mismatch"] for m in metricas_treino_lista])),
    }
    
    print(f"\n--- {media_treino['nome']} ---")
    print(f"  MSE                     : {media_treino['mse']:.6f}")
    print(f"  Erro Relativo L2        : {media_treino['rel_l2']:.4f} ({media_treino['rel_l2']*100:.2f}%)")
    print(f"  Overlap (Alinhado)      : {media_treino['overlap']:.6f}")
    print(f"  Mismatch (Alinhado)     : {media_treino['mismatch']:.6f}")
    
    res_teste = relatorio_metricas(
        f"Exp 2 - Hold-Out (Teste Inédito q={params[idx_teste, 0]:.2f})",
        ref_teste,
        pred_teste
    )
    
    # Gerar gráfico comparativo de Mismatch e Erro Relativo L2
    gerar_painel_metricas(pasta_sxs, ref_teste, pred_teste, t_raw[idx_teste].numpy(), params[idx_teste, 0].item(), res_teste)
    return media_treino, res_teste


def gerar_painel_metricas(pasta_saida: str, h_ref: np.ndarray, h_pred: np.ndarray, t_np: np.ndarray, q_val: float, metricas: dict):
    """Plota a onda de teste acompanhada do erro pontual e das métricas físicas no título."""
    idx_centro = len(t_np) // 2
    janela = slice(idx_centro - 1500, idx_centro + 1000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    ax1.plot(t_np[janela], h_ref[janela], label=f"Onda de Referência SXS (q={q_val:.2f})", color="#4b0082", linewidth=2)
    ax1.plot(t_np[janela], h_pred[janela], label="Emulação FNO (Hold-Out)", color="#ff8c00", linestyle="--", linewidth=1.8)
    ax1.set_title(
        f"Avaliação Física Hold-Out (q={q_val:.2f}) | "
        f"MSE: {metricas['mse']:.6f} | Erro Rel. L2: {metricas['rel_l2']*100:.1f}% | "
        f"Mismatch: {metricas['mismatch']:.4f} (Overlap: {metricas['overlap']:.4f})",
        fontsize=10.5,
        fontweight="bold"
    )
    ax1.set_ylabel("Strain (h_22)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    erro_abs = np.abs(h_ref[janela] - h_pred[janela])
    ax2.plot(t_np[janela], erro_abs, color="#d62728", linewidth=1.5, label="|h_ref(t) - h_pred(t)|")
    ax2.fill_between(t_np[janela], 0, erro_abs, color="#d62728", alpha=0.2)
    ax2.set_xlabel("Tempo Alinhado pelo Pico do Merger (t = 0)")
    ax2.set_ylabel("Erro Absoluto")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    caminho_fig = os.path.join(pasta_saida, "avaliacao_metricas_fisicas.png")
    plt.savefig(caminho_fig, dpi=150)
    plt.close()
    print(f"\n[OK] Painel de métricas físicas salvo em: {caminho_fig}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Executando avaliação física no dispositivo: {device}")
    
    pasta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_raw = os.path.join(pasta_base, "data", "raw")
    pasta_sxs = os.path.join(pasta_raw, "dataset_sxs")
    
    avaliar_experimento_1_reconstrucao(device, pasta_raw)
    avaliar_experimento_2_holdout(device, pasta_sxs)

