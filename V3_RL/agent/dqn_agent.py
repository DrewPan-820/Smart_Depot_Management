import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from V3_RL.env.AttentionQNetwork import AttentionQNetwork

class AttentionDQNAgent:
    def __init__(self, stack_input_dim, order_input_dim, hidden_dim, n_heads,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, buffer_size=10000, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.policy_net = AttentionQNetwork(stack_input_dim, order_input_dim, hidden_dim=hidden_dim, n_heads=n_heads)
        self.target_net = AttentionQNetwork(stack_input_dim, order_input_dim, hidden_dim=hidden_dim, n_heads=n_heads)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_size)

    def act(self, order_vec, stack_vecs, valid_actions_mask):
        if random.random() < self.epsilon:
            # 随机选择有效动作
            valid_actions = np.where(valid_actions_mask)[0]
            return random.choice(valid_actions)

        self.policy_net.eval()
        with torch.no_grad():
            scores = self.policy_net(torch.FloatTensor(order_vec).unsqueeze(0),
                                     torch.FloatTensor(stack_vecs).unsqueeze(0))  # (1, n_stack)
            scores = scores.squeeze(0).cpu().numpy()
            scores[~valid_actions_mask] = -np.inf  # mask invalid actions
            action = np.argmax(scores)
        return action

    def store(self, order_vec, stack_vecs, action, reward, next_order_vec, next_stack_vecs, done, next_valid_actions_mask):
        self.replay_buffer.append((order_vec, stack_vecs, action, reward, next_order_vec, next_stack_vecs, done, next_valid_actions_mask))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        order_vecs, stack_vecs, actions, rewards, next_order_vecs, next_stack_vecs, dones, _ = zip(*batch)  # 忽略 mask

        order_vecs = torch.FloatTensor(order_vecs)
        stack_vecs = torch.FloatTensor(stack_vecs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_order_vecs = torch.FloatTensor(next_order_vecs)
        next_stack_vecs = torch.FloatTensor(next_stack_vecs)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 当前 Q
        current_q = self.policy_net(order_vecs, stack_vecs).gather(1, actions)

        # 目标 Q
        with torch.no_grad():
            next_q = self.target_net(next_order_vecs, next_stack_vecs)
            next_q_max = next_q.max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_max

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
