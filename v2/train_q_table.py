import numpy as np
import matplotlib.pyplot as plt
from RL.greedy_env import GreedyEnv
import json

class QTrainer:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.q_table = np.zeros(num_actions)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q_table))
        else:
            return np.argmax(self.q_table)

    def update_q(self, action, reward):
        current_q = self.q_table[action]
        new_q = current_q + self.lr * (reward - current_q)
        self.q_table[action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    env = GreedyEnv()
    trainer = QTrainer(num_actions=len(env.actions))

    episodes = 200
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        action = trainer.choose_action()
        next_state, reward, done, info = env.step(action)

        trainer.update_q(action, reward)
        trainer.decay_epsilon()
        rewards.append(reward)

        print(f"Episode {episode+1}: Action {action}, Reward {reward:.2f}, Epsilon {trainer.epsilon:.3f}, Alpha {info['alpha']}, Beta {info['beta']}")

    best_action = np.argmax(trainer.q_table)
    best_alpha, best_beta = env.actions[best_action]
    print("\n✅ Best Alpha/Beta combo:", best_alpha, best_beta)
    print("Q-values:", trainer.q_table)

    # Save best alpha/beta as JSON
    with open("best_params.json", "w") as f:
        json.dump({"alpha": best_alpha, "beta": best_beta}, f)

    # Optional: save Q-table for further analysis
    np.save("q_table.npy", trainer.q_table)

    plt.plot(rewards)
    plt.title("Training Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()
