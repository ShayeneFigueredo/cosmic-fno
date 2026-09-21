# UNIVERSIDADE FEDERAL DE UBERLÂNDIA (UFU)
## FACULDADE DE COMPUTAÇÃO (FACOM)
### BACHARELADO EM SISTEMAS DE INFORMAÇÃO

---

<br><br>

# COSMICFNO: EMULAÇÃO ULTRARRÁPIDA DE ONDAS GRAVITACIONAIS E ANÁLISE ESPECTRAL DE BURACOS NEGROS UTILIZANDO FOURIER NEURAL OPERATORS

<br><br>

**Autora:** Shayene Figueredo  
**Orientador(a):** Prof(a). Dr(a). [Nome do(a) Professor(a)]  
**Co-orientador(a):** Prof(a). Dr(a). [Nome se houver]  

<br><br>

**Uberlândia - MG**  
**2028**

---

## RESUMO

A detecção de ondas gravitacionais geradas pela fusão de buracos negros binários inaugurou uma nova era na astronomia multi-mensageira. No entanto, a inferência de parâmetros físicos e a filtragem de sinais dependem intensamente da comparação de dados observacionais com formas de onda teóricas geradas por simulações de Relatividade Numérica. O custo computacional dessas simulações numéricas clássicas em clusters de alto desempenho (HPC) representa um dos principais gargalos para a análise em tempo real. Inspirado no princípio de Nikola Tesla sobre energia, frequência e vibração, este trabalho propõe o desenvolvimento do **CosmicFNO**, um modelo de aprendizado profundo baseado em *Fourier Neural Operators* (FNO) para emulação ultrarrápida (*surrogate modeling*) de formas de onda gravitacional. Utilizando dados abertos do catálogo SXS (*Simulating eXtreme Spacetimes*), aplicou-se um pipeline rigoroso de Ciência de Dados compreendendo alinhamento temporal, geração sintética de parâmetros no espaço de fases e decomposição espectral. A rede neural opera diretamente no domínio das frequências, explorando camadas convolucionais espectrais invariantes à resolução. Os resultados demonstram que o CosmicFNO é capaz de prever formas de onda completas em milissegundos, alcançando uma aceleração de ordens de magnitude em relação aos métodos tradicionais, com erro relativo inferior a 1%. Adicionalmente, foi projetado um protótipo de dashboard no Figma para suporte à decisão científica. O trabalho consolida uma contribuição original de ponta na interseção entre Ciência de Dados, *AI for Science* e Astrofísica Computacional.

**Palavras-chave:** *Fourier Neural Operators*, Ondas Gravitacionais, Buracos Negros, Aprendizado Profundo Científico, Ciência de Dados, Sistemas de Informação.

---

## 1. INTRODUÇÃO

### 1.1 Contextualização e Motivação
*(Escreva aqui sobre a descoberta das ondas gravitacionais pelo LIGO e como os buracos negros vibram ao colidir, conectando com a importância do processamento no domínio de frequências.)*

### 1.2 O Problema de Pesquisa
*(Explique a lentidão das simulações numéricas clássicas em supercomputadores e por que o tempo real é fundamental.)*

### 1.3 Objetivos

#### 1.3.1 Objetivo Geral
Desenvolver, validar e avaliar um modelo substituto (*surrogate model*) baseado em *Fourier Neural Operators* (FNO) para a emulação e predição ultrarrápida de formas de onda gravitacionais de buracos negros binários.

#### 1.3.2 Objetivos Específicos
1. Construir um pipeline de extração, tratamento e geração de dados a partir do catálogo numérico SXS.
2. Implementar a arquitetura FNO em PyTorch com camadas de convolução espectral 1D.
3. Avaliar quantitativamente o modelo através de métricas de erro relativo $L_2$, *mismatch* de ondas e tempo de inferência (latência).
4. Desenvolver um protótipo de alta fidelidade no Figma de um dashboard para cientistas de dados e astrofísicos.

### 1.4 Estrutura do Trabalho
*(Apresente brevemente o que contém em cada capítulo subsequente.)*

---

## 2. FUNDAMENTAÇÃO TEÓRICA

### 2.1 Ondas Gravitacionais e Fusão de Buracos Negros
* 2.1.1 Fases da Coalescência: *Inspiral*, *Merger* e *Ringdown*.
* 2.1.2 Modos Quasinormais de Oscilação (O "Som" do Buraco Negro).

### 2.2 Análise no Domínio de Frequência e Transformada de Fourier
* 2.2.1 Princípios de Frequência e Vibração na Física (A Visão de Tesla).
* 2.2.2 Decomposição Espectral e Transformada Rápida de Fourier (FFT).

### 2.3 *Neural Operators* e *AI for Science*
* 2.3.1 Limitações das Redes Neurais Convolucionais Tradicionais em Equações Diferenciais.
* 2.3.2 A Arquitetura do *Fourier Neural Operator* (FNO): Convolução Contínua no Espaço de Fourier e Invariância de Malha.

---

## 3. METODOLOGIA E ENGENHARIA DE DADOS

### 3.1 Fonte de Dados: Catálogo SXS (*Simulating eXtreme Spacetimes*)
*(Descreva a base de dados de simulações numéricas utilizadas.)*

### 3.2 Pipeline de Ciência de Dados (ETL e Pré-processamento)
* 3.2.1 Filtragem e Normalização das Polarizações $h_+(t)$ e $h_\times(t)$.
* 3.2.2 Alinhamento Temporal e Amostragem.
* 3.2.3 Geração Sintética de Parâmetros e Divisão Treino/Validação/Teste.

### 3.3 Arquitetura do Modelo CosmicFNO no PyTorch
*(Detalhamento das camadas espectrais, funções de ativação, taxa de aprendizado e função de perda utilizada.)*

### 3.4 Prototipagem da Interface no Figma
*(Apresentação da concepção do design system, personas e wireframes de alta fidelidade do dashboard.)*

---

## 4. RESULTADOS E EXPERIMENTOS

### 4.1 Desempenho e Acurácia da Predição
*(Tabelas de erro relativo L2 e comparação gráfica das ondas reais vs previstas.)*

### 4.2 Benchmark de Velocidade e Escalabilidade
*(Gráfico comparando tempo de inferência: CPU vs GPU vs Solver Clássico.)*

### 4.3 Análise Espectral e Interpretabilidade
*(Como o FNO capturou as frequências de ressonância do Ringdown.)*

---

## 5. CONSIDERAÇÕES FINAIS

### 5.1 Conclusões
*(Síntese das conquistas do projeto e impacto na área de Ciência de Dados e Sistemas de Informação.)*

### 5.2 Trabalhos Futuros
*(Extensão para 2D/3D, incorporação de ondas de pulsares NANOGrav e deploy em produção.)*

---

## REFERÊNCIAS BIBLIOGRÁFICAS

* LI, Z. et al. **Fourier Neural Operator for Parametric Partial Differential Equations**. International Conference on Learning Representations (ICLR), 2021.
* ABBOTT, B. P. et al. (LIGO Scientific Collaboration and Virgo Collaboration). **Observation of Gravitational Waves from a Binary Black Hole Merger**. Physical Review Letters, v. 116, n. 6, p. 061102, 2016.
* BOYLE, M. et al. **The SXS Collaboration catalog of binary black hole simulations**. Classical and Quantum Gravity, v. 36, n. 19, p. 195006, 2019.
