# Publicação 2: Engenharia de Dados Sintéticos e Tensores

**Sugestão de Imagem:** Um print bem recortado do terminal rodando o script `01`, mostrando a mensagem "Dataset concluído! Formato de X: torch.Size([1000, 2])". Opcionalmente, um trecho de código ao lado.

---

### Texto do Post para o LinkedIn:

Não existe Inteligência Artificial avançada sem Engenharia de Dados bem feita. 

No meu último post, comentei sobre a pesquisa que estou desenvolvendo para simular ondas gravitacionais usando Fourier Neural Operators. Mas antes de construir o "cérebro" da rede, eu precisava de dados. Muitos dados.

Na astrofísica, os dados reais captados por detectores como o LIGO são incrivelmente ruidosos (interferências sísmicas, ruído quântico). Para a IA aprender a "física pura" de uma colisão de buracos negros, o pipeline padrão dita que devemos treiná-la em *dados sintéticos* — um ambiente isolado onde a matemática é perfeita.

Foi o que fiz essa semana. Construí um gerador em Python que calcula a equação da fase Ringdown de buracos negros com massas variadas e gera milhares de amostras.

Mas aqui entra a sacada arquitetural de Sistemas de Informação: **como armazenar isso?**
Salvar 1.000 (ou no futuro, 100.000) séries temporais densas em CSV seria um pesadelo computacional de I/O na hora do treinamento. 

A solução foi serializar os arrays diretamente em **Tensores do PyTorch (.pt)**. 
Ao manter os dados no formato nativo, o carregamento do disco para a memória RAM (e futuramente para a VRAM da GPU) acontece em bloco binário, eliminando o *overhead* de parsing de texto do CSV. A diferença no *dataloader* é absurda.

Os alicerces de dados estão prontos. Agora, a matemática vai ficar pesada. Próxima parada: implementar a arquitetura FNO e a Transformada Rápida de Fourier. 

Alguém aí já trocou pipelines de dados tubulares por Tensores PyTorch em cenários de alta densidade? Como foi o ganho de performance?

#DataEngineering #PyTorch #Python #DataScience #Astrophysics #DeepLearning
