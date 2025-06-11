import gym
import numpy as np
from gym import spaces
from V3_RL.sim.depot import Depot
from V3_RL.data.order_simulator import generate_mock_orders
from V3_RL.sim.container import Container
from V3_RL.env.reward_scheme import compute_reward

class DepotEnv(gym.Env):
    def __init__(self, num_stacks=8, stack_height=4, num_orders=30, depot=None, orders=None):
        super(DepotEnv, self).__init__()

        self.external_depot = depot
        self.external_orders = orders
        self.default_num_stacks = num_stacks
        self.default_stack_height = stack_height
        self.num_orders = num_orders

        self.depot = None
        self.orders = []
        self.current_step = 0
        self.current_time = 0

        # 动态初始化
        self.action_space = None
        self.observation_space = None

        self.reset()

    def reset(self):
        self.depot = self.external_depot if self.external_depot else Depot(self.default_num_stacks, self.default_stack_height)
        self.orders = self.external_orders if self.external_orders else generate_mock_orders(self.num_orders)
        self.current_step = 0
        self.current_time = 0

        self.num_stacks = len(self.depot.stacks)
        self.stack_height = len(self.depot.stacks[0].containers) if self.depot.stacks else self.default_stack_height

        self.action_space = spaces.Discrete(self.num_stacks)
        state_dim = self._compute_state_dim()
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)

        return self._get_state()

    def step(self, action):
        order = self.orders[self.current_step]
        stack = self.depot.stacks[action]
        success = False
        removed_container = None

        if order.is_loading:
            if len(stack.containers) < self.stack_height and \
                    (not stack.containers or stack.top_container().size == order.size):
                container = Container(id=-1, size=order.size, grace_period=24)
                stack.add_container(container)
                success = True
        else:
            if stack.top_container() and stack.top_container().size == order.size:
                removed_container = stack.remove_top_container()
                success = True

        reward = compute_reward(order, stack, success, removed_container)

        self.current_step += 1
        self.current_time += 1  # 模拟时间流逝 (例如每步1分钟)
        self.depot.increment_idle_times()

        done = self.current_step >= len(self.orders)

        state = self._get_state() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return state, reward, done, {"success": success}

    def _get_state(self):
        order = self.orders[self.current_step]
        order_size_onehot = [int(order.size == s) for s in ['20ft', '40ft', '60ft', '80ft']]
        order_vec = order_size_onehot + [order.priority / 3, int(order.is_loading)]

        stack_vecs = []
        for stack in self.depot.stacks:
            top = stack.top_container()
            if top:
                s = [top.idle_time / top.grace_period] + \
                    [int(top.size == sz) for sz in ['20ft', '40ft', '60ft', '80ft']]
            else:
                s = [0, 0, 0, 0, 0]
            s.append(len(stack.containers) / self.stack_height)
            stack_vecs.extend(s)

        return np.array(order_vec + stack_vecs, dtype=np.float32)

    def _compute_state_dim(self):
        order_dim = 4 + 1 + 1
        stack_dim = 6
        return order_dim + len(self.depot.stacks) * stack_dim

    def render(self, mode="human"):
        print(f"==== DEPOT STATE at step {self.current_step} ====")
        print(self.depot)
