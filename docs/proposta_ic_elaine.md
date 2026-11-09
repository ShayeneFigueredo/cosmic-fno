# 🌌 CosmicFNO: Proposta de Iniciação Científica (Draft)

**Aluna:** Shayene Figueredo (BSI - FACOM/UFU)
**Possível Orientadora:** Prof.ª Dr.ª Elaine Faria

---

## 📌 1. Resumo do Projeto
O CosmicFNO é um projeto de Iniciação Científica focado na intersecção entre *Machine Learning* e Ciência de Dados Científicos (*AI for Science*). O objetivo é investigar a viabilidade e a eficiência computacional de **Fourier Neural Operators (FNO)** em problemas de regressão e emulação de séries temporais complexas, utilizando como estudo de caso sinais astronômicos (Ondas Gravitacionais).

## 🎯 2. O Problema (Abordagem de Computação)
Em física computacional, prever séries temporais de alta complexidade frequentemente exige simulações numéricas (ex: resolver PDEs passo a passo no domínio do tempo), o que é computacionalmente proibitivo. Redes Neurais tradicionais (CNNs 1D, LSTMs) sofrem com dependência de resolução e dificuldade em aprender o mapeamento global do sinal.

O FNO surge como alternativa promissora ao aprender o mapeamento não-linear no domínio da frequência (via FFT), promovendo inferência rápida e invariância de resolução. A proposta desta IC é realizar um benchmark rigoroso de ML testando essa premissa.

## 🛠️ 3. Escopo Delimitado e Entregáveis (Plano de 12 Meses)
Para garantir a execução realista, a física teórica é abstraída e o foco recai exclusivamente no processamento da série temporal do sinal dominante.

* **Fase 1 (Meses 1-4): Engenharia de Dados e Pipeline Base**
  * Extração do modo harmônico dominante ($h_{22}$) do catálogo público *Simulating eXtreme Spacetimes (SXS)*.
  * Formatação em tensores PyTorch e criação de splits rigorosos (Treino/Validação/Teste) com base na razão de massas ($q$), para evitar *data leakage*.
* **Fase 2 (Meses 5-9): Implementação e Benchmark Experimental**
  * Treinamento do modelo FNO 1D.
  * Treinamento de um *baseline* tradicional (ex: 1D-CNN ou MLP).
  * Análise comparativa focada em: Erro MSE/L2, tempo de inferência por amostra e consumo de memória.
* **Fase 3 (Meses 10-12): Consolidação e Escrita**
  * Empacotamento do repositório no GitHub (código limpo, reprodutível e documentado).
  * Escrita de artigo (formato SBC/IEEE) reportando os achados metodológicos (sejam eles positivos ou apontando limitações do FNO), para submissão no evento de IC da UFU ou em workshops correlatos (BRACIS, CSBC).

---

## 📩 Rascunho de E-mail de Abordagem (Tom informal/próximo)

**Assunto:** Ideia de IC em Machine Learning (Séries Temporais / AI for Science)

Oi, professora Elaine! Tudo bem?

Queria te apresentar uma ideia de projeto de Iniciação Científica que estou estruturando e saber se você teria interesse em me orientar. 

Estou estudando bastante uma arquitetura chamada **Fourier Neural Operators (FNO)**, que lida com predição de séries temporais não no domínio do tempo, mas sim aprendendo no domínio da frequência (via Transformada de Fourier). 

A minha ideia de projeto não é focar na física teórica, mas sim fazer um projeto de Machine Learning empírico: pegar uma base de dados pública de ondas gravitacionais (o catálogo SXS) e extrair apenas o sinal principal (o modo $h_{22}$). O objetivo seria comparar o tempo de inferência, o custo de memória e a capacidade de generalização desse modelo FNO contra uma *baseline* comum, como uma CNN 1D. 

Como envolve muito tratamento de dados, split rigoroso pra evitar *data leakage* e análise de performance (áreas que eu sei que você domina), achei que teria tudo a ver com a sua linha de pesquisa e o trabalho que faz na FACOM.

Eu já tenho um protótipo inicial do código rodando em PyTorch para mostrar. Você teria uns 15 minutinhos em alguma das suas próximas idas à faculdade (ou online mesmo) pra gente bater um papo rápido sobre se esse escopo faz sentido pra uma IC com você?

Obrigada desde já, prof!

Abraço,
**Shayene**
