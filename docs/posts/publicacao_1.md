# Publicação 1: A Origem e a Modelagem Matemática

**Sugestão de Imagem:** Foto de um caderno/tablet com as anotações abaixo, e o notebook no fundo rodando o VSCode.

---

### O que escrever no caderno/tablet para a foto:
**Título:** Modelagem Matemática: Coalescência de Buracos Negros (Fase Ringdown)
**Desenho:** Plano cartesiano com uma onda amortecida (começa forte, decai para a direita).
**Fórmula:** $h(t) = A \cdot e^{-t/\tau} \cdot \cos(2\pi \cdot f_{QNM} \cdot t + \phi)$
**Anotações:**
- $h(t)$: Strain (Deformação do espaço-tempo). O que o LIGO mede.
- $e^{-t/\tau}$: Amortecimento exponencial (energia dissipada).
- $f_{QNM}$: Frequência do modo quasinormal.
- *Hipótese:* Substituir simulações $O(N)$ lentas por Fourier Neural Operators (FNO) em $O(1)$.

---

### Texto do Post para o LinkedIn:

A aviação e a exploração espacial sempre pareceram mundos distintos, mas na engenharia de dados, as equações muitas vezes falam a mesma língua. 

Trabalhar diariamente com o ecossistema da aviação no meu estágio me fez perceber a beleza por trás de sistemas complexos e otimização. Essa vivência reacendeu minha antiga paixão pela astrofísica e me levou a iniciar uma pesquisa independente focada na intersecção entre Inteligência Artificial e a Relatividade Geral. 

O meu objetivo? Acelerar a simulação de Ondas Gravitacionais.

Atualmente, quando os observatórios (como o LIGO) detectam o sinal de dois buracos negros colidindo, eles precisam comparar esse sinal com catálogos gerados por supercomputadores rodando simulações de Relatividade Numérica. O problema é que essas simulações numéricas (resolvendo as Equações de Campo de Einstein) são brutalmente custosas e lentas.

Estou iniciando a modelagem de um pipeline usando **Fourier Neural Operators (FNOs)** construído em PyTorch. A ideia é fugir do tradicional processamento "ponto a ponto" no domínio do tempo, e treinar a rede para aprender o mapeamento global das ondas no domínio da frequência, utilizando a Transformada Rápida de Fourier (FFT).

Na foto, um rascunho da modelagem analítica da fase *Ringdown* (quando os buracos negros recém-fundidos "vibram" antes de estabilizar). 

Se conseguirmos ensinar uma IA a mapear essas condições iniciais com invariância de resolução espacial, o ganho de performance na geração de *templates* pode ser na casa de ordens de grandeza.

Ainda tenho uma longa curva de aprendizado em física pela frente, mas a beleza da tecnologia é justamente construir as pontes que nos permitem enxergar mais longe. Animada para compartilhar as próximas etapas (e os prováveis bugs no caminho) com vocês! 🚀✨

#DataScience #MachineLearning #Astrophysics #PyTorch #Aviation #AI

