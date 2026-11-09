# 🛠️ Módulo 03: Engenharia de Dados & O Catálogo SXS

Bem-vinda ao Módulo 03! Agora entramos na ponte entre a física teórica e a Ciência de Dados prática. Aqui vamos entender como os dados de simulações de buracos negros são estruturados, tratados e preparados para alimentar o nosso modelo de Inteligência Artificial (**CosmicFNO**).

---

## 1. O que é o Catálogo SXS?

O **SXS (*Simulating eXtreme Spacetimes*)** é a maior colaboração científica internacional (liderada por instituições como *Caltech* e *Cornell*) dedicada a resolver as equações da Relatividade Geral de Einstein em supercomputadores.

* **O que eles fazem:** Eles simulam no computador duas massas colidindo e medem a "vibração" que sai do sistema.
* **O que nós usamos:** Os dados públicos de formas de onda gravitacionais geradas por essas simulações.

---

## 2. A Estrutura dos Dados de uma Onda Gravitacional

Uma onda gravitacional não é apenas um número solto, ela é uma **série temporal** contendo:

1. **Tempo ($t$):** O instante de cada medição em segundos (ou em unidades geométricas $M$).
2. **Deformação $h_+(t)$ (*h-plus*):** A distorção do espaço no formato de cruz ($+$).
3. **Deformação $h_\times(t)$ (*h-cross*):** A distorção do espaço no formato diagonal ($\times$).
4. **Modo Harmônico $(l=2, m=2)$:** Como a onda se espalha em uma esfera 3D, ela é dividida em "harmônicos esféricos". O modo $(2, 2)$ é o mais importante e carrega mais de **95% de toda a energia do evento**!

---

## 3. As 4 Etapas do Pipeline de Engenharia de Dados (ETL)

Antes de entregar os dados para o CosmicFNO, precisamos aplicar um pipeline de tratamento rigoroso:

```
[Dados Brutos SXS]
       │
       ▼
1. Extração e Seleção do Modo (2, 2)
       │
       ▼
2. Alinhamento Temporal (Pico do Merger em t = 0)
       │
       ▼
3. Amostragem Regular e Interpolação (Malha Fixa N = 1000)
       │
       ▼
4. Decomposição Espectral (FFT para Domínio de Frequência)
       │
       ▼
[Dataset Pronto para o FNO]
```

### Por que cada etapa é necessária?

* **Alinhamento Temporal:** Cada colisão dura um tempo ligeiramente diferente dependendo da massa. Para a IA comparar "maçãs com maçãs", colocamos o instante da colisão máxima (*Merger*) exatamente no tempo $t = 0$.
* **Amostragem Regular ($N=1000$ pontos):** Redes neurais e operadores de Fourier trabalham melhor quando os dados têm uma grade dimensional bem definida.
* **Cálculo da FFT:** Já extraímos as frequências e amplitudes para que o FNO possa treinar tanto no tempo quanto no espaço espectral.

---

## 4. O Código Prático: Linha por Linha

Nosso primeiro script de engenharia de dados está localizado em [`experimentos/01_extracao_e_analise_sxs.py`](../experimentos/01_extracao_e_analise_sxs.py).

Ele executa:
1. Geração/carregamento da forma de onda física padrão de colisão ($m_1 = 30 M_\odot$, $m_2 = 30 M_\odot$).
2. Separação visual das 3 fases: *Inspiral*, *Merger* e *Ringdown*.
3. Cálculo da Transformada Rápida de Fourier (FFT).
4. Exportação dos dados limpos em formato tabular CSV para a pasta `dados/`.
5. Geração de um gráfico comparativo em alta resolução.

---

## 📝 Resumo Rápido para Fixação
- [x] O catálogo SXS fornece as curvas teóricas mais precisas do mundo para ondas gravitacionais.
- [x] O modo $(2, 2)$ contém quase toda a informação necessária da colisão.
- [x] Alinhar o tempo no pico do *Merger* ($t=0$) é fundamental para treinar modelos de Machine Learning.
- [x] A FFT converte o sinal no tempo em um espectro de frequências que alimentará o CosmicFNO.
