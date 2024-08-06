# Instruções Para Deploy Na Nuvem

Este guia busca orientar sobre o processo de implantação do modelo de segmentação ResUNet no Azure. O processo envolve várias etapas, incluindo a criação de uma conta no Azure, a configuração de um workspace, o registro do modelo, a criação de um script de pontuação, a criação de um ambiente, a implantação do modelo e a realização de testes.

## O que é Implantação de Modelo em Aprendizado de Máquina?

A implantação de um modelo de aprendizado de máquina é o processo de integrar um modelo de aprendizado de máquina em um ambiente de produção onde ele pode receber uma entrada e retornar uma saída. O objetivo é tornar as previsões do seu modelo de aprendizado de máquina treinado disponíveis para outros, sejam usuários, gerentes ou outros sistemas.

### Critérios para Implantação de Modelo

Antes de implantar um modelo, é necessário atender a alguns critérios:

- **Portabilidade**: A capacidade do software ser transferido de uma máquina ou sistema para outro. Um modelo portátil tem um tempo de resposta relativamente baixo e pode ser reescrito com esforço mínimo.
- **Escalabilidade**: Refere-se a quão grande o modelo pode escalar. Um modelo escalável é aquele que não precisa ser redesenhado para manter seu desempenho.

### Arquitetura de Sistema de Aprendizado de Máquina para Implantação de Modelo

Em um nível alto, existem quatro partes principais em um sistema de aprendizado de máquina:

- **Camada de dados**: Fornece acesso a todas as fontes de dados que o modelo exigirá.
- **Camada de características**: Responsável por gerar dados de características de maneira transparente, escalável e utilizável.
- **Camada de pontuação**: Transforma características em previsões. Scikit-Learn é comumente usado e é o padrão da indústria para pontuação.
- **Camada de avaliação**: Verifica a equivalência de dois modelos e pode ser usada para monitorar modelos de produção. É usada para monitorar e comparar como as previsões de treinamento correspondem às previsões no tráfego ao vivo.

### Métodos de Implantação de Modelos

Existem três maneiras gerais de implantar um modelo de aprendizado de máquina: one-off, batch e em tempo real.

- **One-off**: Às vezes, um modelo só é necessário uma vez ou periodicamente. Nesse caso, o modelo pode ser treinado ad-hoc quando necessário e colocado em produção até que se deteriore o suficiente para exigir correção.
- **Batch**: Permite ter uma versão constantemente atualizada do modelo, eliminando a necessidade de usar o conjunto de dados completo para cada atualização. É útil se o modelo for usado de forma consistente, mas não necessariamente precisar das previsões em tempo real.
- **Em tempo real**: Em alguns casos, é necessária uma previsão em tempo real, como determinar se uma transação é fraudulenta. Isso é possível usando modelos de aprendizado de máquina online, como regressão linear usando gradiente descendente estocástico.

## Guia Passo a Passo

### 1. Configurar um Workspace no Azure Machine Learning

Um workspace no Azure Machine Learning é um recurso fundamental na nuvem utilizado para experimentar, treinar e implantar modelos de aprendizado de máquina.

1. Faça login no portal do Azure.
2. Clique em "Criar um recurso".
3. Pesquise por "Machine Learning" e selecione.
4. Clique em "Criar" e preencha os detalhes necessários, como nome do workspace, assinatura, grupo de recursos e localização.
5. Clique em "Revisar + Criar" e depois em "Criar".

### 2. Registrar o Modelo

O registro do modelo permite armazenar e versionar modelos no Azure Machine Learning.

1. No seu workspace, vá para "Modelos" e clique em "Registrar Modelo".
2. Forneça um nome para o modelo e faça o upload do arquivo do modelo treinado (`model_res_u_net.pth`). Este modelo pode ser obtido no nosso Jupyter Notebook ResUNet usando o comando `torch.save(model, '/content/drive/My Drive/model_res_u_net.pth')`.
3. Preencha outros detalhes conforme necessário e clique em "Registrar".

### 3. Criar um Script de Pontuação

O script de pontuação recebe entradas de dados em um pedido POST, pontua a entrada com o modelo e retorna os resultados.

1. Crie um script Python que carregue o modelo do workspace e defina duas funções: `init()` e `run(raw_data)`.
2. A função `init()` é usada para carregar o modelo. É executada apenas uma vez quando o contêiner Docker para o serviço web é iniciado.
3. A função `run(raw_data)` usa o modelo para prever a entrada de dados.

### 4. Definir uma Configuração de Inferência

Uma configuração de inferência descreve como configurar o serviço web que contém o modelo.

1. Defina uma configuração de inferência que especifique o runtime a ser usado, o script de pontuação e outras configurações.

### 5. Definir uma Configuração de Implantação

Uma configuração de implantação especifica a quantidade de recursos de computação a serem alocados para hospedar o modelo.

1. Crie uma configuração de implantação. É possível especificar o número de núcleos e a quantidade de memória.

### 6. Implantar o Modelo

1. No seu workspace, vá para "Endpoints" e clique em "Novo Endpoint".
2. Selecione o modelo que deseja implantar, a configuração de inferência e a configuração de implantação.
3. Clique em "Implantar".

### 7. Testar o Modelo

1. Após a implantação do modelo, é possível testá-lo enviando uma solicitação POST com dados de entrada para a URL do serviço web.
2. O serviço web retornará a previsão do modelo.

## Conclusão

A implantação de um modelo de aprendizado de máquina no Azure envolve várias etapas, mas uma vez que o processo é compreendido, ele pode ser facilmente replicado para outros modelos. A chave é entender os diferentes componentes envolvidos e como eles interagem entre si.