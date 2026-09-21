# 🎓 Proposta de Iniciação Científica (IC): Roteiro de Apresentação para a Professora de Ciência de Dados I (UFU)

Este documento foi preparado para você apresentar à sua professora de Ciência de Dados I. Ele contém o **roteiro de abordagem (e-mail/conversa)** e a **proposta estruturada de projeto de IC**.

---

## 💬 Roteiro de Abordagem (Como falar com a Professora)

> *"Professora Elaine, tudo bem?*
> 
> *Eu adorei a matéria de Ciência de Dados I e estou muito motivada para aprofundar na área. Estudei as novas tendências de **Deep Learning Científico (AI for Science)** e gostaria de propor um projeto de Iniciação Científica com você focado em **Fourier Neural Operators (FNO)** aplicados à emulação de séries temporais físicas (especificamente ondas gravitacionais e sinais de buracos negros).*
> 
> *O projeto envolve toda a esteira prática de Ciência de Dados: coleta de dados abertos (catálogo SXS), geração de dados sintéticos para preenchimento do espaço de fases, análise espectral com FFT, treinamento de FNO em PyTorch e análise de erro/desempenho.*
> 
> *Estruturei uma proposta de IC de 1 ano que servirá como base direta para o meu TCC em Sistemas de Informação. Posso te apresentar um resumo de 5 minutos ou enviar a proposta para a sua avaliação?"*

---

# 📄 Resumo da Proposta de Iniciação Científica (1 Página)

### **Título do Projeto:**
**Aplicação de Fourier Neural Operators (FNO) e Ciência de Dados para Emulação de Alta Performance em Sinais Físicos Não-Lineares**

* **Aluna Proponente:** Shayene Figueredo (Graduanda em Sistemas de Informação - UFU)
* **Orientadora:** Profª. Drª. Elaine Ribeiro de Faria (FACOM / UFU)
* **Colaboradora Matemática (Proposta):** Profª. M.Sc. Lílian Amorim (IFNMG - Campus Januária)
* **Grande Área:** Ciência da Computação / Ciência de Dados / Inteligência Artificial

---

### 1. Justificativa e Relevância
A computação científica enfrenta um desafio clássico: simular equações diferenciais parciais e sistemas dinâmicos complexos (como o comportamento do espaço-tempo em colisões de corpos massivos) exige tempo inviável de supercomputadores. 

Recentemente, a área de *AI for Science* introduziu os **Neural Operators**, com destaque para os **Fourier Neural Operators (FNO)**, que aprendem mapeamentos funcionais no domínio de Fourier, permitindo predições ordens de magnitude mais rápidas sem perda substancial de fidelidade matemática. 

A aplicação de técnicas de Ciência de Dados para tratar, gerar e modelar esses sinais representa uma oportunidade fértil de pesquisa de ponta com alto potencial de publicação acadêmica para a UFU.

---

### 2. Plano de Trabalho da Aluna (12 Meses)

| Mês | Atividade Principal | Entregável |
| :---: | :--- | :--- |
| **M1-M3** | Revisão bibliográfica sobre FNOs e exploração do catálogo aberto SXS (*Simulating eXtreme Spacetimes*). | Relatório de Análise Exploratória de Dados (EDA). |
| **M4-M6** | Construção do pipeline de dados: amostragem, geração de dados sintéticos e transformadas espectrais. | Dataset estruturado e scripts de ETL em Python/PyTorch. |
| **M7-M9** | Implementação e treinamento do modelo FNO 1D; ajuste de hiperparâmetros e funções de perda espectrais. | Modelo treinado e curvas de convergência. |
| **M10-M12** | Benchmarks de latência e erro $L_2$ vs métodos numéricos; redação de artigo para congresso e relatório final da IC. | Artigo submetido e base consolidada para o TCC. |

---

### 3. Vantagens Estratégicas para a Professora / Orientadora
1. **Tema Quente e Internacional:** FNOs e *AI for Science* estão no centro das principais conferências (NeurIPS, ICLR, IEEE).
2. **Dados Públicos e Estruturados:** O projeto não depende de experimentos físicos caros em laboratório; utiliza bases públicas abertas de alto padrão (SXS / LIGO).
3. **Alto Potencial de Publicação:** O trabalho gera facilmente artigos para eventos nacionais (SBC/CSBC/SBBD) e periódicos de Ciência de Dados.
