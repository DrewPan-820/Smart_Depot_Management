import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha

    def add(self, experience, td_error=1.0):
        # 保证优先级正数且非nan
        init_prio = abs(td_error) if np.isfinite(td_error) and td_error > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(init_prio)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = init_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        scaled_prios = np.array(self.priorities) ** self.alpha
        total_prio = scaled_prios.sum()
        # === 修正点1: 全为零/全nan时均匀采样 ===
        if not np.isfinite(total_prio) or total_prio == 0:
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probs = scaled_prios / total_prio

        # === 修正点2: 若批量过大，clip掉 ===
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        # 保证所有优先级为有限正数
        for idx, err in zip(indices, td_errors):
            prio = abs(err)
            if not np.isfinite(prio) or prio <= 0:
                prio = 1.0
            self.priorities[idx] = prio
