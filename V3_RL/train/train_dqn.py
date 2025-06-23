import torch
import numpy as np
from V3_RL.env.depot_env import DepotEnv
import matplotlib.pyplot as plt
from V3_RL.agent.dqn_agent import AttentionDQNAgent


def train_dqn(
        episodes=5000,
        target_update_freq=20,
        stack_height=6,
        num_orders=30,
        hidden_dim=128,
        n_heads=4,
        lr=1e-3
):
    env = DepotEnv(stack_height=stack_height, num_orders=num_orders)
    stack_input_dim = 6
    order_input_dim = 6

    agent = AttentionDQNAgent(
        stack_input_dim=stack_input_dim,
        order_input_dim=order_input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        lr=lr
    )

    reward_history = []
    best_reward = float('-inf')
    best_episode = -1

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            order_vec = state[:order_input_dim]
            num_stacks = env.num_stacks
            stack_vecs = state[order_input_dim:].reshape(num_stacks, stack_input_dim)
            valid_actions_mask = env.get_valid_action_mask()

            action = agent.act(order_vec, stack_vecs, valid_actions_mask)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_order_vec = next_state[:order_input_dim]
                next_stack_vecs = next_state[order_input_dim:].reshape(num_stacks, stack_input_dim)
                next_valid_actions_mask = env.get_valid_action_mask()
            else:
                next_order_vec = np.zeros(order_input_dim)
                next_stack_vecs = np.zeros((num_stacks, stack_input_dim))
                next_valid_actions_mask = np.zeros(num_stacks + 1, dtype=bool)

            agent.store(order_vec, stack_vecs, action, reward, next_order_vec, next_stack_vecs, done,
                        next_valid_actions_mask)
            agent.train_step()
            state = next_state

        reward_history.append(total_reward)
        agent.decay_epsilon()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode: {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode + 1  # 1-based counting
            torch.save(agent.policy_net.state_dict(), "trained_dqn_best.pth")
            print(f"🌟 Best model saved at episode {best_episode}, reward={best_reward:.2f}")

    torch.save(agent.policy_net.state_dict(), "trained_dqn.pth")
    print("Training completed and model saved.")

    plt.plot(reward_history, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Trend (Attention DQN)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return reward_history, agent, best_episode, best_reward
