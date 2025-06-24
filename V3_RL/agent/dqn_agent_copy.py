import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from V3_RL.env.AttentionQNetwork import AttentionQNetwork

class AttentionDQNAgent:
    def __init__(self, stack_input_dim, order_input_dim, hidden_dim, n_heads,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01, buffer_size=10000, batch_size=64):
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
        num_stacks = len(stack_vecs)
        valid_actions = np.where(valid_actions_mask)[0].tolist()

        if len(valid_actions) == 0:
            return num_stacks  # wait

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        self.policy_net.eval()
        with torch.no_grad():
            scores = self.policy_net(
                torch.FloatTensor(order_vec).unsqueeze(0),
                torch.FloatTensor(stack_vecs).unsqueeze(0)
            )
            scores = scores.squeeze(0).cpu().numpy()
            # 扩展 Q 值数组为 n_stack + 1（增加 wait 动作）
            scores = np.append(scores, -np.inf)
            full_mask = np.append(valid_actions_mask.astype(bool), True)
            scores[~full_mask] = -np.inf
            action = int(np.argmax(scores))
            # 强制保证返回的动作合法
            assert full_mask[action], f"Selected invalid action: {action}"
        return action

    def store(self, order_vec, stack_vecs, action, reward, next_order_vec, next_stack_vecs, done, next_valid_actions_mask):
        self.replay_buffer.append((order_vec, stack_vecs, action, reward, next_order_vec, next_stack_vecs, done, next_valid_actions_mask))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        order_vecs, stack_vecs, actions, rewards, next_order_vecs, next_stack_vecs, dones, next_valid_actions_mask = zip(*batch)

        order_vecs = torch.FloatTensor(order_vecs)
        stack_vecs = torch.FloatTensor(stack_vecs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_order_vecs = torch.FloatTensor(next_order_vecs)
        next_stack_vecs = torch.FloatTensor(next_stack_vecs)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 当前 Q
        current_q = self.policy_net(order_vecs, stack_vecs).gather(1, actions)

        # === 优化点：为target Q值批量掩码非法动作 ===
        with torch.no_grad():
            next_q = self.target_net(next_order_vecs, next_stack_vecs)
            # mask非法动作：next_valid_actions_mask是(batch_size, n_action)的布尔数组
            # next_valid_actions_mask 需要转成 np.ndarray
            next_valid_mask = np.stack(next_valid_actions_mask)
            next_q[~torch.tensor(next_valid_mask)] = -float('inf')
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
