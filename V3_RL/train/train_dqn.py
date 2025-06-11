import torch
import numpy as np
from V3_RL.env.depot_env import DepotEnv
from V3_RL.agent.dqn_agent import AttentionDQNAgent
from V3_RL.env.depot_env_wait import DepotEnvWithWait
import matplotlib.pyplot as plt

def train_dqn(
    episodes=5000,
    target_update_freq=20,
    stack_height=6,
    num_orders=30,
    hidden_dim=128,
    n_heads=4,
    lr=1e-3
):
    # 不指定num_stacks，环境内部自由变化
    env = DepotEnv(stack_height=stack_height, num_orders=num_orders)

    stack_input_dim = 6  # 单个stack的特征维度
    order_input_dim = 6  # 单个订单特征维度

    agent = AttentionDQNAgent(
        stack_input_dim=stack_input_dim,
        order_input_dim=order_input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        lr=lr
    )

    reward_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        reward_history.append(total_reward)

        while not done:
            order_vec = state[:order_input_dim]
            num_stacks = env.num_stacks  # 动态获取堆栈数量
            stack_vecs = state[order_input_dim:].reshape(num_stacks, stack_input_dim)

            valid_actions_mask = np.array([
                (len(stack.containers) < env.stack_height if env.orders[env.current_step].is_loading
                 else stack.top_container() and stack.top_container().size == env.orders[env.current_step].size)
                for stack in env.depot.stacks
            ])

            action = agent.act(order_vec, stack_vecs, valid_actions_mask)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_order_vec = next_state[:order_input_dim]
                next_stack_vecs = next_state[order_input_dim:].reshape(env.num_stacks, stack_input_dim)

                next_valid_actions_mask = np.array([
                    (len(stack.containers) < env.stack_height if env.orders[env.current_step].is_loading
                     else stack.top_container() and stack.top_container().size == env.orders[env.current_step].size)
                    for stack in env.depot.stacks
                ])
            else:
                next_order_vec = np.zeros(order_input_dim)
                next_stack_vecs = np.zeros((env.num_stacks, stack_input_dim))
                next_valid_actions_mask = np.zeros(env.num_stacks, dtype=bool)

            agent.store(order_vec, stack_vecs, action, reward, next_order_vec, next_stack_vecs, done, next_valid_actions_mask)
            agent.train_step()

            state = next_state

        reward_history.append(total_reward)
        agent.decay_epsilon()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode: {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), "trained_dqn_best.pth")

    torch.save(agent.policy_net.state_dict(), "trained_dqn.pth")
    print("Training completed and model saved.")

    plt.plot(reward_history, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Trend (Attention DQN)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return reward_history, agent


def train_dqn_with_wait(
    episodes=5000,
    update_target_every=20,
    num_stacks=8,
    stack_height=4,
    hidden_dim=128,
    n_heads=4,
    lr=1e-3
):
    from V3_RL.env.depot_env_wait import DepotEnvWithWait
    from V3_RL.agent.dqn_agent import AttentionDQNAgent

    env = DepotEnvWithWait(num_stacks=num_stacks, stack_height=stack_height)

    stack_input_dim = 6
    order_input_dim = 6

    agent = AttentionDQNAgent(
        stack_input_dim=stack_input_dim,
        order_input_dim=order_input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        lr=lr
    )

    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False

        while not done:
            order_vec = state[:order_input_dim]
            stack_vecs = state[order_input_dim:].reshape(env.num_stacks, stack_input_dim)

            valid_actions_mask = np.array([
                True if stack.top_container() else False
                for stack in env.depot.stacks
            ] + [True])  # 包括 wait 动作

            action = agent.act(order_vec, stack_vecs, valid_actions_mask)

            next_state, reward, done, _ = env.step(action)

            if not done:
                next_order_vec = next_state[:order_input_dim]
                next_stack_vecs = next_state[order_input_dim:].reshape(env.num_stacks, stack_input_dim)

                next_valid_actions_mask = np.array([
                    True if stack.top_container() else False
                    for stack in env.depot.stacks
                ] + [True])
            else:
                next_order_vec = np.zeros(order_input_dim)
                next_stack_vecs = np.zeros((env.num_stacks, stack_input_dim))
                next_valid_actions_mask = np.zeros(env.num_stacks + 1, dtype=bool)

            agent.store(order_vec, stack_vecs, action, reward, next_order_vec, next_stack_vecs, done, next_valid_actions_mask)
            agent.train_step()

            state = next_state
            ep_reward += reward

        agent.decay_epsilon()
        if ep % update_target_every == 0:
            agent.update_target_network()

        total_rewards.append(ep_reward)
        if ep % 100 == 0:
            print(f"Episode {ep}: Reward = {ep_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    return total_rewards, agent

