import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from V3_RL.env.AttentionQNetwork import AttentionQNetwork
from V3_RL.agent.prioritized_replay_buffer import PrioritizedReplayBuffer

class AttentionDQNAgent:
    def __init__(self, stack_input_dim, order_input_dim, hidden_dim, n_heads,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01, buffer_size=10000, batch_size=64, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.frame_idx = 0

        # 现在 order_input_dim=6, future_order_vecs=18, 所以 order_all_dim=24
        self.order_all_dim = order_input_dim + 3 * order_input_dim  # 24

        self.policy_net = AttentionQNetwork(stack_input_dim, order_input_dim, hidden_dim=hidden_dim, n_heads=n_heads)
        self.target_net = AttentionQNetwork(stack_input_dim, order_input_dim, hidden_dim=hidden_dim, n_heads=n_heads)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha)

    def act(self, order_vec, future_order_vecs, stack_vecs, valid_actions_mask):
        num_stacks = len(stack_vecs)
        valid_actions = np.where(valid_actions_mask)[0].tolist()

        if len(valid_actions) == 0:
            return num_stacks  # wait

        # 拼接当前订单 + 未来订单
        full_order_vec = np.concatenate([order_vec, future_order_vecs])  # 24维
        # test code
        # print("[DEBUG][agent.act] full_order_vec.shape:", full_order_vec.shape,
        #       "stack_vecs.shape:", stack_vecs.shape)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        self.policy_net.eval()
        with torch.no_grad():
            scores = self.policy_net(
                torch.FloatTensor(full_order_vec).unsqueeze(0),  # (1, 24)
                torch.FloatTensor(stack_vecs).unsqueeze(0)       # (1, n_stack, 6)
            )
            scores = scores.squeeze(0).cpu().numpy()
            scores = np.append(scores, -np.inf)
            full_mask = np.append(valid_actions_mask.astype(bool), True)
            scores[~full_mask] = -np.inf
            action = int(np.argmax(scores))
            assert full_mask[action]
        return action

    def store(self, order_vec, future_order_vecs, stack_vecs, action, reward,
              next_order_vec, next_future_order_vecs, next_stack_vecs, done, next_valid_actions_mask):
        # 存储当前和下一个“全订单信息”
        self.replay_buffer.add((order_vec, future_order_vecs, stack_vecs, action, reward,
                                next_order_vec, next_future_order_vecs, next_stack_vecs, done, next_valid_actions_mask))

    def train_step(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        self.frame_idx += 1
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=beta)
        if not batch:
            return

        (order_vecs, future_order_vecs, stack_vecs, actions, rewards,
         next_order_vecs, next_future_order_vecs, next_stack_vecs, dones, next_valid_actions_mask) = zip(*batch)

        order_vecs = np.array(order_vecs)
        future_order_vecs = np.array(future_order_vecs)
        full_order_vecs = np.concatenate([order_vecs, future_order_vecs], axis=1)  # (B, 24)

        next_order_vecs = np.array(next_order_vecs)
        next_future_order_vecs = np.array(next_future_order_vecs)
        next_full_order_vecs = np.concatenate([next_order_vecs, next_future_order_vecs], axis=1)  # (B, 24)
        # test code
        # print("[DEBUG][train_step] full_order_vecs.shape:", full_order_vecs.shape)
        # print("[DEBUG][train_step] next_full_order_vecs.shape:", next_full_order_vecs.shape)
        # print("[DEBUG][train_step] policy_net.order_proj.weight.shape:", self.policy_net.order_proj.weight.shape)
        stack_vecs = torch.FloatTensor(stack_vecs)
        next_stack_vecs = torch.FloatTensor(next_stack_vecs)
        full_order_vecs = torch.FloatTensor(full_order_vecs)
        next_full_order_vecs = torch.FloatTensor(next_full_order_vecs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        # 当前 Q
        current_q = self.policy_net(full_order_vecs, stack_vecs).gather(1, actions)

        # print("full_order_vecs shape:", full_order_vecs.shape)
        # print("order_proj weight shape:", self.policy_net.order_proj.weight.shape)

        # ===== Double DQN 核心部分 =====
        with torch.no_grad():
            next_q_policy = self.policy_net(next_full_order_vecs, next_stack_vecs)
            next_valid_mask = np.stack(next_valid_actions_mask)
            next_q_policy[~torch.tensor(next_valid_mask)] = -float('inf')
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)

            next_q_target = self.target_net(next_full_order_vecs, next_stack_vecs)
            next_q_target[~torch.tensor(next_valid_mask)] = -float('inf')
            next_q_target = next_q_target.gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_target

        td_errors = (current_q - target_q).detach().cpu().numpy()
        loss = self.criterion(current_q, target_q)
        loss = loss * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, np.abs(td_errors).flatten())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
