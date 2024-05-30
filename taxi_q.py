import pickle  # Para salvar e carregar a tabela Q

import gymnasium as gym  # Biblioteca para simular o ambiente de aprendizado por reforço
import matplotlib.pyplot as plt  # Para criar gráficos
import numpy as np  # Para cálculos numéricos
import pygame  # Para criar a interface gráfica
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # Para desenhar gráficos no Pygame


# Função que treina o agente
def train_agent(episodes, learning_rate_a=0.9, discount_factor_g=0.9, epsilon=1, epsilon_decay_rate=0.0001,
                save_interval=1000):
    env = gym.make('Taxi-v3')  # Inicializa o ambiente Taxi-v3
    q = np.zeros((env.observation_space.n, env.action_space.n))  # Cria a tabela Q com zeros
    rng = np.random.default_rng()  # Gerador de números aleatórios
    rewards_per_episode = np.zeros(episodes)  # Guarda as recompensas por episódio

    # Loop principal para treinar o agente
    for i in range(episodes):
        state = env.reset()[0]  # Reseta o ambiente e pega o estado inicial
        terminated = False  # Flag para verificar se o episódio terminou
        truncated = False  # Flag para verificar se o episódio foi truncado

        while not terminated and not truncated:
            if rng.random() < epsilon:  # Exploração: escolhe uma ação aleatória
                action = env.action_space.sample()
            else:  # Exploração: escolhe a melhor ação com base na tabela Q
                action = np.argmax(q[state, :])

            # Executa a ação no ambiente
            new_state, reward, terminated, truncated, info = env.step(action)
            # Atualiza a tabela Q usando a fórmula de Q-Learning
            q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )
            state = new_state  # Atualiza o estado atual

        # Decaimento do epsilon para reduzir a exploração ao longo do tempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001  # Diminui a taxa de aprendizado quando a exploração termina

        if reward == 20:  # Recompensa máxima no Taxi-v3
            rewards_per_episode[i] = 1

        # Salva a tabela Q a cada intervalo definido
        if i % save_interval == 0:
            with open("taxi.pkl", "wb") as f:
                pickle.dump(q, f)

    env.close()  # Fecha o ambiente
    # Salva a tabela Q ao final do treinamento
    with open("taxi.pkl", "wb") as f:
        pickle.dump(q, f)

    return rewards_per_episode, q  # Retorna as recompensas por episódio e a tabela Q


# Função que executa o agente no ambiente
def run_agent(env, q, delay=100, random_actions=False):
    state = env.reset()[0]  # Reseta o ambiente e pega o estado inicial
    terminated = False  # Flag para verificar se o episódio terminou
    truncated = False  # Flag para verificar se o episódio foi truncado
    total_reward = 0  # Inicializa a recompensa total

    # Loop principal para executar o agente
    while not terminated and not truncated:
        if random_actions:  # Se escolher ações aleatórias
            action = env.action_space.sample()
        else:  # Se escolher a melhor ação com base na tabela Q
            action = np.argmax(q[state, :])
        # Executa a ação no ambiente
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state  # Atualiza o estado atual
        total_reward += reward  # Acumula a recompensa total
        raw_image = env.render()  # Renderiza o ambiente
        surf = pygame.surfarray.make_surface(np.transpose(raw_image, (1, 0, 2)))  # Converte a imagem para o Pygame
        yield surf, total_reward  # Retorna a superfície e a recompensa total
        pygame.time.wait(delay)  # Adiciona um atraso entre as ações para visualização

    yield None, total_reward  # Finaliza o episódio


# Função para plotar as recompensas
def plot_rewards(rewards, ax):
    sum_rewards = np.zeros_like(rewards)  # Inicializa o array de recompensas acumuladas
    for t in range(len(rewards)):
        sum_rewards[t] = np.sum(
            rewards[max(0, t - 100):(t + 1)])  # Calcula as recompensas acumuladas a cada 100 episódios
    ax.clear()  # Limpa o gráfico
    ax.plot(sum_rewards, label='Recompensas por Episódio')  # Plota as recompensas acumuladas
    ax.set_xlabel('Episódios')  # Define o rótulo do eixo X
    ax.set_ylabel('Recompensas Acumuladas')  # Define o rótulo do eixo Y
    ax.set_title('Recompensas por Episódio - Q-Learning em Taxi-v3')  # Define o título do gráfico
    ax.legend()  # Adiciona a legenda


# Função para desenhar texto na tela do Pygame
def draw_text(screen, text, position, font_size=30, color=(0, 0, 0)):
    font = pygame.font.Font(None, font_size)  # Define a fonte e o tamanho do texto
    text_surface = font.render(text, True, color)  # Renderiza o texto
    screen.blit(text_surface, position)  # Desenha o texto na tela na posição especificada


# Função principal que roda todo o processo
def run_all():
    pygame.init()  # Inicializa o Pygame
    window_height = 620  # Define a altura da janela
    window_width = 1920  # Define a largura da janela
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)  # Cria a janela redimensionável
    pygame.display.set_caption('Visualização Q-Learning Taxi-v3')  # Define o título da janela

    fig, ax = plt.subplots(figsize=(6, 4))  # Cria uma figura e um eixo para o gráfico
    canvas = FigureCanvas(fig)  # Cria um canvas para desenhar o gráfico

    # Inicializa os ambientes
    env1 = gym.make('Taxi-v3', render_mode='rgb_array')
    env2 = gym.make('Taxi-v3', render_mode='rgb_array')

    # Treina o agente e salva a tabela Q
    rewards_during_training, q_trained = train_agent(5000)  # Treina o agente por 5000 episódios

    running = True  # Flag para manter a janela aberta
    clock = pygame.time.Clock()  # Relógio para controlar a taxa de atualização

    # Cria geradores para executar os agentes
    agent1_generator = run_agent(env1, np.zeros((env1.observation_space.n, env1.action_space.n)), delay=100,
                                 random_actions=True)
    agent2_generator = run_agent(env2, q_trained, delay=100)

    while running:  # Loop principal do Pygame
        screen.fill((255, 255, 255))  # Preenche a tela com branco
        window_width, _ = screen.get_size()  # Obtém o tamanho atual da janela
        section_width = window_width // 3  # Divide a largura da janela em três seções
        section_height = window_height - 100  # Define a altura das seções

        # Desenha os títulos centralizados
        draw_text(screen, "Sem Treinamento", (section_width // 2 - 100, 20), 40)
        draw_text(screen, "Recompensas Durante Treinamento", (section_width + section_width // 2 - 200, 20), 40)
        draw_text(screen, "Treinado", (2 * section_width + section_width // 2 - 60, 20), 40)

        # Atualiza a visualização do agente sem treinamento
        try:
            surf1, _ = next(agent1_generator)
            if surf1:
                screen.blit(pygame.transform.scale(surf1, (section_width, section_height)), (0, 100))
            else:
                agent1_generator = run_agent(env1, np.zeros((env1.observation_space.n, env1.action_space.n)), delay=100,
                                             random_actions=True)
        except StopIteration:
            agent1_generator = run_agent(env1, np.zeros((env1.observation_space.n, env1.action_space.n)), delay=100,
                                         random_actions=True)

        # Plota as recompensas durante o treinamento
        plot_rewards(rewards_during_training, ax)
        canvas.draw()
        raw_data = canvas.buffer_rgba()
        plot_surf = pygame.image.frombuffer(raw_data, canvas.get_width_height(), "RGBA")
        plot_surf = pygame.transform.scale(plot_surf, (section_width, section_height))
        screen.blit(plot_surf, (section_width, 100))

        # Atualiza a visualização do agente treinado
        try:
            surf2, _ = next(agent2_generator)
            if surf2:
                screen.blit(pygame.transform.scale(surf2, (section_width, section_height)), (2 * section_width, 100))
            else:
                agent2_generator = run_agent(env2, q_trained, delay=100)
        except StopIteration:
            agent2_generator = run_agent(env2, q_trained, delay=100)

        pygame.display.flip()  # Atualiza a tela do Pygame
        clock.tick(30)  # Ajusta a taxa de atualização para 30 FPS

        # Processa eventos do Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Sai do loop principal
            elif event.type == pygame.VIDEORESIZE:
                window_width, window_height = event.size
                window_height = 620  # Mantém a altura ajustada
                screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)

    env1.close()  # Fecha o primeiro ambiente
    env2.close()  # Fecha o segundo ambiente
    pygame.quit()  # Encerra o Pygame


if __name__ == '__main__':
    run_all()  # Executa a função principal
