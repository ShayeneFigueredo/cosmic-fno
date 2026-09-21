<div align="center">

# 🌌 CosmicFNO
### Emulação Ultrarrápida de Ondas Gravitacionais e Análise Espectral de Buracos Negros Utilizando Fourier Neural Operators

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![AI for Science](https://img.shields.io/badge/AI%20for%20Science-Physics%20Informed-purple?style=for-the-badge)](https://github.com/)
[![SXS Catalog](https://img.shields.io/badge/Data-SXS%20Gravitational%20Wave-black?style=for-the-badge)](https://www.black-holes.org/waveforms/)
[![UFU FACOM](https://img.shields.io/badge/UFU-FACOM%20BSI-green?style=for-the-badge)](https://www.facom.ufu.br/)

<br>

**Trabalho de Conclusão de Curso (TCC)**  
*Bacharelado em Sistemas de Informação*  
**Faculdade de Computação (FACOM) — Universidade Federal de Uberlândia (UFU)**  
**Autora:** Shayene Figueredo

</div>

---

## 🎯 Sobre o Projeto

O **CosmicFNO** é uma abordagem de ponta em *AI for Science* que utiliza **Fourier Neural Operators (FNO)** para prever e emular ondas gravitacionais emitidas pela fusão de buracos negros binários (*Binary Black Holes - BBH*).

Enquanto simulações de Relatividade Numérica clássicas em supercomputadores (HPC) exigem **dias ou semanas** para resolver as equações de campo de Einstein para um único evento, o **CosmicFNO** realiza essa predição no domínio espectral em **milissegundos**, alcançando aceleração de ordens de magnitude com erro relativo inferior a 1%.

```mermaid
flowchart LR
    A["Parâmetros Físicos\n(Massas m1, m2, Spins)"] --> B["CosmicFNO\n(Camadas Espectrais 1D)"]
    B --> C["Forma de Onda h(t)\n(Inspiral + Merger + Ringdown)"]
    C --> D["Dashboard de Decisão\n(Análise em Tempo Real)"]
    
    style B fill:#7b2cbf,stroke:#3c096c,stroke-width:2px,color:#fff
    style A fill:#0077b6,stroke:#03045e,stroke-width:2px,color:#fff
    style C fill:#2a9d8f,stroke:#264653,stroke-width:2px,color:#fff
    style D fill:#e76f51,stroke:#d62828,stroke-width:2px,color:#fff
```

---

## ⚡ Por que Fourier Neural Operators (FNO)?

1. **Invariância de Resolução:** Ao contrário de CNNs tradicionais, o FNO aprende mapeamentos contínuos entre espaços funcionais, permitindo avaliar a onda em qualquer taxa de amostragem sem retreinamento.
2. **Operação no Domínio de Frequência:** Inspirado na visão de frequências e vibrações, o operador espectral captura naturalmente as frequências de ressonância dos **Modos Quasinormais (QNM)** no amortecimento do buraco negro (*Ringdown*).
3. **Substituto Ultrarrápido (*Surrogate Model*):** Viabiliza a inferência de parâmetros em tempo real para observatórios como LIGO, Virgo e KAGRA.

---

## 📂 Estrutura do Repositório

```text
cosmic-fno/
├── dados/               # Dados processados e metadados das formas de onda
├── docs/                # Monografia em LaTeX/Markdown, referências e propostas
├── experimentos/        # Scripts de modelagem, treinamento e benchmarks
├── wiki/                # Documentação didática e base de conhecimento passo a passo
│   ├── 00_TRILHA_DE_APRENDIZAGEM.md
│   ├── 01_CONCEITOS_FISICOS_ONDAS_GRAVITACIONAIS.md
│   └── ...
├── .gitignore           # Filtro para ignorar datasets pesados e checkpoints
└── README.md            # Apresentação principal do projeto
```

---

## 📖 Trilha de Aprendizagem & Documentação (Wiki)

Toda a teoria física, cálculos matemáticos e explicações de código linha a linha estão documentados na nossa Wiki interna:

- 🌌 **[Módulo 01: Conceitos Físicos e Ondas Gravitacionais](wiki/01_CONCEITOS_FISICOS_ONDAS_GRAVITACIONAIS.md)**
- 📐 **[Módulo 02: Matemática de Fourier & Frequências](wiki/02_MATEMATICA_FOURIER_DO_ZERO.md)**
- 🛠️ **[Módulo 03: Engenharia de Dados & Catálogo SXS](wiki/03_ENGENHARIA_DADOS_SXS.md)**
- 🧠 **[Módulo 04: Redes Neurais & Arquitetura FNO](wiki/04_IA_FOURIER_NEURAL_OPERATOR.md)**
- 📊 **[Módulo 05: Métricas Científicas e Benchmarks](wiki/05_METRICAS_E_EXPERIMENTOS.md)**
- 🖥️ **[Módulo 06: Dashboard e Sistemas de Informação](wiki/06_SISTEMA_DASHBOARD_FIGMA.md)**

Consulte o mapa completo em **[Trilha de Aprendizagem](wiki/00_TRILHA_DE_APRENDIZAGEM.md)**.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Deep Learning / AI for Science:** PyTorch, NeuralOperator
- **Processamento Espectral & Dados:** NumPy, SciPy, Pandas
- **Visualização Científica:** Matplotlib, Seaborn
- **Design & Interface:** Figma (Design System & Dashboard)
- **Base de Dados:** Catálogo SXS (*Simulating eXtreme Spacetimes*)

---

## 👩‍💻 Autora & Afiliação

* **Autora:** Shayene Figueredo
* **Curso:** Bacharelado em Sistemas de Informação (BSI)
* **Instituição:** Universidade Federal de Uberlândia (UFU) - Faculdade de Computação (FACOM)
* **Ano:** 2026 / 2028

---

<div align="center">
  <sub>"Se você quiser descobrir os segredos do Universo, pense em termos de energia, frequência e vibração." — Nikola Tesla</sub>
</div>
