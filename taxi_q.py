import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def train_agent(episodes, learning_rate_a=0.9, discount_factor_g=0.9, epsilon=1, epsilon_decay_rate=0.0001,
                save_interval=1000):
    env = gym.make('Taxi-v3')
    q = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)
            q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward == 20:
            rewards_per_episode[i] = 1

        if i % save_interval == 0:
            with open("taxi.pkl", "wb") as f:
                pickle.dump(q, f)

    env.close()
    with open("taxi.pkl", "wb") as f:
        pickle.dump(q, f)

    return rewards_per_episode, q


def run_agent(env, q, delay=100, random_actions=False):
    state = env.reset()[0]
    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        if random_actions:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state, :])
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
        total_reward += reward
        raw_image = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(raw_image, (1, 0, 2)))
        yield surf, total_reward
        pygame.time.wait(delay)

    yield None, total_reward


def plot_rewards(rewards, ax):
    sum_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        sum_rewards[t] = np.sum(rewards[max(0, t - 100):(t + 1)])
    ax.clear()
    ax.plot(sum_rewards, label='Recompensas por Episódio')
    ax.set_xlabel('Episódios')
    ax.set_ylabel('Recompensas Acumuladas')
    ax.set_title('Recompensas por Episódio - Q-Learning em Taxi-v3')
    ax.legend()


def draw_text(screen, text, position, font_size=30, color=(0, 0, 0)):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def run_all():
    pygame.init()
    window_height = 620  # Altura ajustada
    window_width = 1920  # Largura ajustada para 1920 pixels
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption('Visualização Q-Learning Taxi-v3')

    fig, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvas(fig)

    # Inicializar ambientes
    env1 = gym.make('Taxi-v3', render_mode='rgb_array')
    env2 = gym.make('Taxi-v3', render_mode='rgb_array')

    # Treinar o agente para gerar o arquivo taxi.pkl
    rewards_during_training, q_trained = train_agent(5000)  # Reduzido para 5000 episódios

    running = True
    clock = pygame.time.Clock()

    agent1_generator = run_agent(env1, np.zeros((env1.observation_space.n, env1.action_space.n)), delay=100,
                                 random_actions=True)
    agent2_generator = run_agent(env2, q_trained, delay=100)

    while running:
        screen.fill((255, 255, 255))
        window_width, _ = screen.get_size()
        section_width = window_width // 3
        section_height = window_height - 100

        # Desenhar títulos centralizados
        draw_text(screen, "Sem Treinamento", (section_width // 2 - 100, 20), 40)
        draw_text(screen, "Recompensas Durante Treinamento", (section_width + section_width // 2 - 200, 20), 40)
        draw_text(screen, "Treinado", (2 * section_width + section_width // 2 - 60, 20), 40)

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

        # Gráfico de recompensas durante o treinamento
        plot_rewards(rewards_during_training, ax)
        canvas.draw()
        raw_data = canvas.buffer_rgba()
        plot_surf = pygame.image.frombuffer(raw_data, canvas.get_width_height(), "RGBA")
        plot_surf = pygame.transform.scale(plot_surf, (section_width, section_height))
        screen.blit(plot_surf, (section_width, 100))

        try:
            surf2, _ = next(agent2_generator)
            if surf2:
                screen.blit(pygame.transform.scale(surf2, (section_width, section_height)), (2 * section_width, 100))
            else:
                agent2_generator = run_agent(env2, q_trained, delay=100)
        except StopIteration:
            agent2_generator = run_agent(env2, q_trained, delay=100)

        pygame.display.flip()
        clock.tick(30)  # Ajuste para garantir uma atualização de tela mais suave

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                window_width, window_height = event.size
                window_height = 620  # Manter altura ajustada
                screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)

    env1.close()
    env2.close()
    pygame.quit()


if __name__ == '__main__':
    run_all()
