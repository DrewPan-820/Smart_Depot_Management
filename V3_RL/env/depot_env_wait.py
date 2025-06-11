from gym import Env, spaces
import numpy as np
from V3_RL.sim.depot import Depot
from V3_RL.data.order_simulator import generate_mock_orders
from V3_RL.sim.container import Container

class DepotEnvWithWait(Env):
    def __init__(self, num_stacks=8, stack_height=4, num_orders=30, depot=None, orders=None):
        super().__init__()
        self.external_depot = depot
        self.external_orders = orders
        self.default_num_stacks = num_stacks
        self.default_stack_height = stack_height
        self.num_orders = num_orders

        self.depot = None
        self.orders = []
        self.current_step = 0

        self.reset()

    def reset(self):
        self.depot = self.external_depot if self.external_depot else Depot(self.default_num_stacks, self.default_stack_height)
        self.orders = self.external_orders if self.external_orders else generate_mock_orders(self.num_orders)
        self.current_step = 0

        self.num_stacks = len(self.depot.stacks)
        self.stack_height = len(self.depot.stacks[0].containers) if self.depot.stacks else self.default_stack_height

        self.action_space = spaces.Discrete(self.num_stacks + 1)  # +1 表示 "wait" 动作
        state_dim = self._compute_state_dim()
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)

        return self._get_state()

    def step(self, action):
        order = self.orders[self.current_step]
        reward = 0

        valid_actions_mask = self.get_valid_action_mask()
        is_valid = (action < self.num_stacks) and valid_actions_mask[action]

        if action == self.num_stacks:
            # 等待行为：轻微惩罚
            reward = -0.2
        elif is_valid:
            stack = self.depot.stacks[action]
            if order.is_loading:
                container = Container(id=-1, size=order.size, grace_period=24)
                stack.add_container(container)
                reward = 1.0
            else:
                container = stack.remove_top_container()
                ratio = container.idle_time / container.grace_period
                reward = 1.0 - np.tanh(2.0 * abs(1.0 - ratio))
        else:
            # 非法动作：强烈惩罚
            reward = -2.0

        self.current_step += 1
        self.depot.increment_idle_times()
        done = self.current_step >= len(self.orders)

        next_state = self._get_state() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_state, reward, done, {}

    def _get_state(self):
        order = self.orders[self.current_step]
        order_size_onehot = [int(order.size == s) for s in ['20ft', '40ft', '60ft', '80ft']]
        order_vec = order_size_onehot + [order.priority / 3, int(order.is_loading)]

        stack_vecs = []
        for stack in self.depot.stacks:
            top = stack.top_container()
            if top:
                s = [top.idle_time / top.grace_period] + [int(top.size == sz) for sz in ['20ft', '40ft', '60ft', '80ft']]
            else:
                s = [0, 0, 0, 0, 0]
            s.append(len(stack.containers) / self.stack_height)
            stack_vecs.extend(s)

        return np.array(order_vec + stack_vecs, dtype=np.float32)

    def get_valid_action_mask(self):
        """
        生成当前订单下的合法 stack 掩码 + wait 动作合法性
        返回 shape = [num_stacks + 1] 的 bool 数组
        """
        order = self.orders[self.current_step]
        mask = []

        for stack in self.depot.stacks:
            if order.is_loading:
                is_valid = len(stack.containers) < self.stack_height and \
                           (not stack.containers or stack.top_container().size == order.size)
            else:
                is_valid = stack.top_container() and stack.top_container().size == order.size
            mask.append(is_valid)

        mask.append(True)  # wait 永远是合法动作
        return np.array(mask, dtype=bool)

    def _compute_state_dim(self):
        order_dim = 6  # 4 one-hot + 1 priority + 1 is_loading
        stack_dim = 6  # idle_ratio + 4 one-hot + height_ratio
        return order_dim + self.default_num_stacks * stack_dim

    def render(self, mode="human"):
        print("==== DEPOT STATE ====")
        print(self.depot)
