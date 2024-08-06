<table>
<tr>
<td>
<a href= "https://adeagro.com.br/"><img src="./docs/Imagens/adeagro.png" alt="A de Agro" border="0" width="100%"></a>
</td>
<td><a href= "https://www.inteli.edu.br/"><img src="./docs/Imagens/logo-inteli.png" alt="Inteli - Instituto de Tecnologia e Liderança" border="15" width="20%"></a>
</td>
</tr>
</table>

# Segmentação de talhões por meio de visão computacional

## Descrição da empresa
> A de Agro é uma empresa focada em se tornar a principal fonte de informações de safras do Brasil, oferecendo análises de inteligência de mercado, monitoramento e previsões para uma ampla gama de setores, incluindo bancos, empresas de crédito rural, seguradoras e outros players do agronegócio. Orientada por valores fundamentais, a empresa prioriza dados concretos sobre suposições, autonomia sobre hierarquia, entrega de valor para o cliente e trabalho em equipe para alcançar resultados.

### Grupo Mockingjay

#### Integrantes:

* [Allan Casado](allan.casado@sou.inteli.edu.br)
* [Cristiane Coutinho](cristiane.coutinho@sou.inteli.edu.br)
* [Elias Biondo](elias.biondo@sou.inteli.edu.br)
* [Gábrio Silva](gabrio.silva@sou.inteli.edu.br)
* [Giovana Thomé](giovana.thome@sou.inteli.edu.br)
* [Rafael Cabral](rafael.cabral@sou.inteli.edu.br)
* [Thomas Barton](thomas.barton@sou.inteli.edu.br)

## Descrição

O objetivo do projeto é desenvolver um modelo de visão computacional capaz de identificar os talhões produtivos de fazendas da Região Sul numa determinada safra. Com este projeto, é esperado que a identificação de talhões da Região Sul seja mais precisa, aumentando a confiança interna dos clientes atuais e dos potenciais clientes sobre os resultados para esta região.

## Documentação

Os arquivos da documentação deste projeto estão na pasta [/docs](/docs), incluindo:

- Documentação de Definições Básicas
- Documentação da métrica interna de avaliação CovR e script para cálculo da métrica
- Documentação sobre rotulação de imagens
- Links importantes:
  - [Sentinel-hub EO-Browser](https://apps.sentinel-hub.com/eo-browser/)
  - [QGIS](https://www.qgis.org/pt_BR/site/about/index.html)

## Artigo

Os arquivos do artigo estão na pasta [/artigo](/artigo). O conteúdo deste artigo foi elaborado como parte das atividades de aprendizado dos alunos.


## Instruções de Execução

Esta seção fornece instruções detalhadas para a configuração e execução do backend da aplicação e para o uso dos notebooks de treinamento e inferência.

### Configuração e Execução do Backend

Siga os passos abaixo para configurar e executar o backend da aplicação:

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/InteliProjects/2023M7T1-Inteli-Grupo-1
   ```

2. **Acesse a pasta do projeto:**
   ```bash
   cd 2023M7T1-Inteli-Grupo-1
   ```

3. **Acesse a pasta do backend:**
   ```bash
   cd Backend
   ```

4. **Crie um ambiente virtual Python (versão 3.10.14):**
   ```bash
   python -m venv venv
   ```

5. **Ative o ambiente virtual:**
   - No Windows:
     ```bash
     venv\Scripts\activate
     ```
   - No Linux ou macOS:
     ```bash
     source venv/bin/activate
     ```

6. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

7. **Execute o backend:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

8. **Acesse a documentação da API:**
   Abra o navegador e visite:
   ```
   http://localhost:8000/api/schema
   ```

### Uso dos Notebooks de Treinamento e Inferência

Para utilizar os notebooks de treinamento e inferência, siga os passos:

1. **Acesse a pasta do projeto:**
   ```bash
   cd 2023M7T1-Inteli-Grupo-1
   ```

2. **Acesse a pasta de Modelos:**
   ```bash
   cd Modelos
   ```

3. **Escolha um dos Modelos disponíveis e acesse sua pasta:**
   Por exemplo, para o modelo ResUnet:
   ```bash
   cd ResUnet
   ```

4. **Abra o notebook de treinamento no Google Colab.**

5. **Faça o upload das imagens de treinamento e validação para o ambiente de execução ou no Google Drive.**

6. **Faça o upload da pasta `Modulos` para o ambiente de execução ou no Google Drive.**

7. **Ajuste os caminhos das imagens de treinamento e validação no notebook.**

8. **Execute o notebook para iniciar o processo de treinamento e inferência.**

# Tags
## SPRINT-1
### Adições e Definições Iniciais
- Resumo e pontos chave do artigo
- Definição do campo e problema
- Value Proposition Canvas
- Descrição da fonte de dados e justificativa

### Desenvolvimento de Notebooks e Código
- Notebooks de image enhancement e análise exploratória
- Testes de pré-processamento
- Código de normalização de imagem
- Banco de imagens
- Random crop e data augmentations
- Pipeline de processamento

### Frontend e Organização do Projeto
- Adição do frontend
- Dockerfile
- Desenvolvimento de artigo

## SPRINT-2
### Refatorações e Organização do Projeto
- Refatoração da pipeline
- Organização de pastas
- Movimentação de notebooks
- Correções gramaticais e atualizações de caminho

### Desenvolvimento de Notebooks e Código
- Adição de notebooks (CNN U-Net Xception, CNN U-Net, principal)
- Experimento com arquitetura CNN
- Nova máscara QGIS
- Rascunho do modelo base
- Transfer learning

### Artigo
- Descrição do método de transfer learning
- Pesquisa e traduções
- Seção de data augmentations
- Explicação de binary-cross-entropy

## SPRINT-3
### Desenvolvimento de Modelos e Pipeline
- Adição de novos modelos (ResUNet, DeepLabV3 descontinuado)
- Cobertura do ResUNet
- Correções e atualizações da pipeline
- Refatoração de crop aleatório para treino
- Transfer learning ResNet50

### Artigo
- Ilustrações e diagramas
- Descrição e documentação da arquitetura do modelo
- Definição de parâmetros da rede

## SPRINT-4
### Atualizações e Finalizações de Pipeline e Treinamento
- Atualização de PDF
- Finalização da pipeline e testes
- Treinamento com imagens com e sem augmentation

### Desenvolvimento de Código e Organização
- Módulos Python
- Backend - API do modelo
- Organização do repositório
- Colab para validação de 1000 imagens

### Documentação e Artigo
- Metodologia do artigo e modificações gerais

## SPRINT-5
### Treinamento e Ajustes Técnicos
- Cross training
- Coeficiente Dice para transfer learning
- Treino com imagens sintéticas

### Correções e Refatorações
- Correção de pasta Experimentos
- Refatoração de endpoints da API e nomes dos modelos

### Documentação e Apresentação
- Apresentação final
- Atualização do README

# Licença

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
