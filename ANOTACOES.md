# 📓 Diário de Bordo e Anotações de Pesquisa: CosmicFNO

**Autora:** Shayene Figueredo  
**Projeto:** CosmicFNO (TCC / Iniciação Científica - Sistemas de Informação / UFU)  
**Objetivo deste Documento:** Registrar todas as etapas práticas, experimentos, decisões técnicas, tratamentos de dados e conceitos matemáticos aprendidos para servir de base direta na escrita da monografia final.

---

## 📌 Índice de Registros
* [Entrada 001 - Concepção do Projeto e Escolha dos FNOs](#entrada-001---concepção-do-projeto-e-escolha-dos-fnos)
* [Entrada 002 - Estrutura dos Dados SXS e Pipeline de Ciência de Dados](#entrada-002---estrutura-dos-dados-sxs-e-pipeline-de-ciência-de-dados)
* [Entrada 003 - Mapeamento de Fontes, Links Oficiais e Bibliotecas](#entrada-003---mapeamento-de-fontes-links-oficiais-e-bibliotecas)
* [Entrada 004 - Parte 1.1 Concluída: Inspeção Estrutural da Onda no Tempo](#entrada-004---parte-11-concluída-inspeção-estrutural-da-onda-no-tempo)
* [Entrada 005 - Parte 1.2 Concluída: Decomposição Espectral de Fourier e Espectrograma](#entrada-005---parte-12-concluída-decomposição-espectral-de-fourier-e-espectrograma)

---

## 📝 Entrada 001 - Concepção do Projeto e Escolha dos FNOs
* **Data:** 13/09/2026
* **Status:** Concepção e Planejamento Inicial

### 1. O Problema
Simulações numéricas de relatividade geral para fusões de buracos negros binários são computacionalmente caras (dias a semanas em supercomputadores). A detecção em tempo real e a caracterização astronômica precisam de modelos substitutos (*surrogate models*) capazes de inferir a onda em milissegundos.

### 2. A Escolha dos Fourier Neural Operators (FNO)
* Em vez de redes neurais convencionais (CNNs/MLPs) que operam em malhas discretas de pixels ou pontos fixos, os **FNOs aprendem o operador diferencial contínuo** diretamente no domínio das frequências.
* **Vantagem de Ciência de Dados:** Invariância de resolução (o modelo pode ser avaliado em qualquer taxa de amostragem no tempo sem precisar ser retreinado).

---

## 📝 Entrada 002 - Estrutura dos Dados SXS e Pipeline de Ciência de Dados
* **Fonte de Dados:** Catálogo *Simulating eXtreme Spacetimes* (SXS).
* **Parâmetros de Entrada ($X$):**
  * Massas dos buracos negros: $m_1, m_2$ (ou razão de massa $q = m_1 / m_2$).
  * Vetores de rotação (*spins* adimensionais): $\vec{\chi}_1, \vec{\chi}_2$ (alinhados com o eixo $z$: $\chi_{1z}, \chi_{2z}$).
* **Variáveis de Saída ($Y$):**
  * Deformações do espaço-tempo: Polarizações $h_+(t)$ e $h_\times(t)$.
* **Fases da Onda a serem tratadas:**
  1. *Inspiral:* Aproximação orbital em espiral (frequência e amplitude crescem).
  2. *Merger:* Colisão máxima (pico de amplitude).
  3. *Ringdown:* Vibração do buraco negro resultante com decaimento exponencial e frequências características (Modos Quasinormais).

---

## 📝 Entrada 003 - Mapeamento de Fontes, Links Oficiais e Bibliotecas
* **Data:** 13/09/2026
* **Status:** Configuração e Mapeamento de Recursos

### 🌐 1. Fontes de Dados e Catálogos Oficiais
* **Catálogo SXS (Simulating eXtreme Spacetimes):**
  * *Site Oficial:* [https://www.black-holes.org/](https://www.black-holes.org/)
  * *Catálogo Aberto de Formas de Onda:* [https://data.black-holes.org/waveforms/](https://data.black-holes.org/waveforms/)
  * *Repositório GitHub do pacote Python SXS:* [https://github.com/sxs-collaboration/sxs](https://github.com/sxs-collaboration/sxs)
* **GWOSC (Gravitational Wave Open Science Center - LIGO/Virgo):**
  * *Portal Aberto:* [https://gwosc.org/](https://gwosc.org/)
  * *Tutoriais de Análise de Sinais:* [https://gwosc.org/tutorials/](https://gwosc.org/tutorials/)

### 🛠️ 2. Bibliotecas de Machine Learning & Ciência de Dados
* **NeuralOperator (Caltech / NVIDIA):**
  * *Documentação Oficial:* [https://neuraloperator.github.io/neuraloperator/](https://neuraloperator.github.io/neuraloperator/)
  * *Repositório GitHub:* [https://github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
  * *Artigo Fundacional (ICLR 2021):* [Fourier Neural Operator for Parametric Partial Differential Equations (Li et al.)](https://arxiv.org/abs/2010.08895)
* **PyTorch (Framework Central):**
  * *Documentação Oficial:* [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
  * *Módulo de Fourier (`torch.fft`):* [https://pytorch.org/docs/stable/fft.html](https://pytorch.org/docs/stable/fft.html)
* **Processamento de Sinais & Visualização:**
  * `scipy.signal` (Transformada de Fourier, Welch, Wavelets)
  * `matplotlib` & `seaborn` (Gráficos científicos para a monografia)
  * `pandas` & `numpy` (Manipulação matricial e tabulação dos metadados)

---

## 📝 Entrada 004 - Parte 1.1 Concluída: Inspeção Estrutural da Onda no Tempo
* **Data:** 13/09/2026
* **Script Executado:** `c:\apps\estudos-astronomia\TCC\experimentos\01_inspecao_onda_sxs.py`
* **Status:** Sucesso (2048 amostras temporais geradas e validadas).

### 📊 1. Dados e Arquivos Gerados
1. **Dataset Bruto (CSV):** `c:\apps\estudos-astronomia\TCC\dados\onda_sxs_q1_m30_m30.csv`
   * *Colunas:* `tempo_segundos` ($t \in [-0.8s, +0.2s]$) e `deformacao_h_plus` ($h_+(t)$).
   * *Configuração Física:* Buracos negros de massas iguais $m_1 = 30 M_\odot, m_2 = 30 M_\odot$ ($q = 1.0$) a uma distância de $400 \text{ Mpc}$.
2. **Gráfico Científico Gerado:** `c:\apps\estudos-astronomia\TCC\experimentos\grafico_01_fases_da_onda.png`

### 🔬 2. Mapeamento das 3 Fases Físicas no Tempo
* **Fase 1: Inspiral ($t < -20\text{ ms}$):**
  * Os dois buracos negros giram um ao redor do outro.
  * A frequência orbital e a amplitude sobem progressivamente (o sinal clássico em "Chirp").
* **Fase 2: Merger ($-20\text{ ms} \le t \le +20\text{ ms}$):**
  * O choque cataclísmico e união dos horizontes de eventos.
  * O pico de amplitude da onda ocorre exatamente em $t = 0\text{ ms}$.
* **Fase 3: Ringdown ($t > +20\text{ ms}$):**
  * O buraco negro resultante vibra em seus *Modos Quasinormais*.
  * Ocorre um decaimento exponencial com uma frequência de ressonância dominante ($\approx 250\text{ Hz}$).

---

## 📝 Entrada 005 - Parte 1.2 Concluída: Decomposição Espectral de Fourier e Espectrograma
* **Data:** 13/09/2026
* **Script Executado:** `c:\apps\estudos-astronomia\TCC\experimentos\02_analise_fourier_fft.py`
* **Status:** Sucesso (Transformada Rápida de Fourier e Espectrograma calculados).

### 📊 1. Dados e Arquivos Gerados
1. **Dataset Espectral (CSV):** `c:\apps\estudos-astronomia\TCC\dados\espectro_fourier_m30_m30.csv`
   * *Colunas:* `frequencia_hz` e `amplitude_espectral` ($|\tilde{h}(f)|$).
2. **Gráfico Científico Editorial:** `c:\apps\estudos-astronomia\TCC\experimentos\grafico_02_espectro_fourier.png`

### 🔬 2. Descobertas Científicas e Princípio de Tesla
* **Pico de Ressonância Dominante (Ringdown):** Detectado exatamente em **$f_{QNM} \approx 247.9\text{ Hz}$**.
  * Esse pico representa a frequência natural de vibração do espaço-tempo para um buraco negro de $60 M_\odot$.
* **O "Chirp" no Espectrograma (Painel B):**
  * O gráfico tempo vs. frequência mostra a curva contínua de energia subindo de $\sim 30\text{ Hz}$ até o ápice do choque em $t = 0\text{ ms}$.
* **Conexão com os Fourier Neural Operators (FNO):**
  * A rede neural FNO vai operar calculando convoluções matemáticas diretamente sobre esse espectro de frequências, garantindo que o modelo aprenda as ressonâncias físicas sem depender da resolução da malha temporal.

---

*(Próxima etapa: Parte 1.3 - Pipeline de Tratamento e Normalização para Entrada da Rede Neural.)*
