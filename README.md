# Aprendizado por Reforço no Ambiente Taxi-v3

Este projeto apresenta uma visualização interativa do aprendizado por reforço utilizando o ambiente Taxi-v3 da biblioteca Gymnasium (anteriormente OpenAI Gym). O objetivo é demonstrar de forma didática o processo de aprendizado de um agente, comparando um agente sem treinamento, o progresso das recompensas durante o treinamento e o desempenho do agente treinado. Tudo isso é exibido em uma única interface gráfica utilizando Pygame.

(Gif abaixo do agente sem treinamento, gráfico da curva de aprendizagem do agente e por final o agente treinado)
![Aprendizado-por-Reforco-no-Ambiente-Taxi-v3/Taxi.gif](https://github.com/Matheussbzp/Aprendizado-por-Reforco-no-Ambiente-Taxi-v3/blob/bc7ccf570fcb6414b67846c85c383b92115be20b/Taxi.gif)
                                                       

## Objetivos do Projeto

1. **Demonstrar o Aprendizado por Reforço**: Mostrar o processo de aprendizado de um agente no ambiente Taxi-v3.
2. **Comparar Desempenhos**: Visualizar a diferença entre um agente sem treinamento e um agente treinado.
3. **Gráfico de Recompensas**: Exibir o progresso das recompensas ao longo do treinamento.
4. **Interface Interativa**: Proporcionar uma visualização clara e interativa usando Pygame.

## Funcionalidades

- Visualização do agente sem treinamento.
- Gráfico de recompensas durante o treinamento.
- Visualização do agente treinado.
- Interface responsiva.

## Requisitos

- Python 3.x
- Gymnasium
- Matplotlib
- NumPy
- Pygame

## Instalação

1. Clone este repositório:

    ```git clone https://github.com/Matheussbzp/Aprendizado-por-Reforco-no-Ambiente-Taxi-v3.git```
   
2. Navegue até o diretório do projeto:

   ```cd Aprendizado-por-Reforco-no-Ambiente-Taxi-v3```

3. Crie um ambiente virtual:

   ```python -m venv venv```
   
4. Ative o ambiente virtual:

   ```.\venv\Scripts\activate```
   
 4.1 No macOS e Linux:

    source venv/bin/activate
    
5. Instale as dependências:

   ```pip install -r requirements.txt```
   
## **Execução**

1. Treine o agente:

   ```python taxi_q.py --train```
   
2. Visualize o agente sem treinamento:

   ```python taxi_q.py --visualize --no-training```

3. Visualize o agente treinado:

   ```python taxi_q.py --visualize --trained```
   
## Agradecimentos

Gostaria de expressar minha imensa gratidão ao Victor Dias, do canal Universo Programado, por indiretamente inspirar este sonho de aprender Machine Learning através dos seus conteúdos. Se não fosse por ele, eu não teria descoberto este universo tão cedo.
https://www.universoprogramado.com/

O Matheus do passado se sentiria orgulhoso.

