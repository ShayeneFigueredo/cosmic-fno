# ComparaÃ§Ã£o de Modelos: FNO vs 1D-CNN

Este documento detalha o teste de validaÃ§Ã£o de arquitetura (Representation Learning / Sanity Check) realizado com a simulaÃ§Ã£o real `SXS:BBH:0305` (GW150914).

O objetivo deste teste foi provar matematicamente que a arquitetura do **Fourier Neural Operator (FNO)** Ã© superior Ã s redes clÃ¡ssicas (como a **1D-CNN**) para a tarefa de emular ondas gravitacionais.

## O Problema das Ondas Gravitacionais
Ondas gravitacionais geradas pela colisÃ£o de buracos negros possuem trÃªs fases principais:
1. **Inspiral:** FrequÃªncia baixa e amplitude crescente.
2. **Merger:** O momento do choque. AltÃ­ssima frequÃªncia e pico de amplitude.
3. **Ringdown:** O eco final, que decai rapidamente.

Redes neurais tradicionais sofrem para modelar as fases de *Merger* e *Ringdown* porque o sinal muda de forma muito abrupta.

## Resultados do Teste

Ambos os modelos foram treinados por 100 Ã©pocas utilizando o mesmo sinal real, a mesma taxa de aprendizado e a mesma funÃ§Ã£o de perda (MSE - Mean Squared Error).

### 1. Modelo Baseline: 1D-CNN (Rede Neural Convolucional)
- **Erro Final (MSE):** ~0.0164
- **AnÃ¡lise:** As CNNs possuem um "campo receptivo local". Elas olham apenas para os pontos vizinhos na grade de tempo. Por serem "mÃ­opes", elas tendem a suavizar o sinal. Como visto no grÃ¡fico abaixo, a CNN nÃ£o consegue acompanhar as oscilaÃ§Ãµes violentas do *Merger*, criando uma linha quase reta que ignora a fÃ­sica do buraco negro.

![Resultado CNN](assets/resultado_treino_cnn.png)

### 2. Modelo Proposto: FNO (Fourier Neural Operator)
- **Erro Final (MSE):** ~0.0003 (Cerca de 50x menor que a CNN)
- **AnÃ¡lise:** O FNO nÃ£o olha para a grade de tempo. Ele converte o sinal inteiro para o domÃ­nio das frequÃªncias usando a Transformada RÃ¡pida de Fourier (FFT). Isso dÃ¡ a ele um "campo receptivo global" â€” ele enxerga a onda como um todo. O resultado Ã© uma reconstruÃ§Ã£o quase perfeita da amplitude e da frequÃªncia do sinal original.

![Resultado FNO](assets/resultado_treino_real.png)

## ConclusÃ£o
Este teste preliminar justifica firmemente a escolha do FNO como a arquitetura principal do projeto de IniciaÃ§Ã£o CientÃ­fica, provando sua capacidade superior de capturar a fÃ­sica complexa de sinais astrofÃ­sicos.


