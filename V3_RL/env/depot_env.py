import gym
import numpy as np
from gym import spaces
from V3_RL.sim.depot import Depot
from V3_RL.data.order_simulator import generate_mock_orders
from V3_RL.sim.container import Container

class DepotEnv(gym.Env):
    def __init__(self, num_stacks=8, stack_height=4, num_orders=30, depot=None, orders=None):
        super(DepotEnv, self).__init__()
        self.external_depot = depot
        self.external_orders = orders
        self.default_num_stacks = num_stacks
        self.default_stack_height = stack_height
        self.num_orders = num_orders

        # 初始化环境状态变量
        self.depot = None
        self.orders = []
        self.current_order_idx = 0
        self.current_order = None
        self.current_time = 0

        # 动作空间和状态空间将在 reset 中根据实际堆场大小设置
        self.action_space = None
        self.observation_space = None

        # 重置环境以初始化状态
        self.reset()

    def reset(self):
        """重置环境状态。如果提供了外部 depot 和 orders，则使用它们，否则随机生成新的。"""
        # 设置堆场和订单列表（优先使用传入的外部对象）
        self.depot = self.external_depot if self.external_depot else Depot(self.default_num_stacks, self.default_stack_height)
        self.orders = self.external_orders if self.external_orders else generate_mock_orders(self.num_orders)

        # 重置当前订单索引和时间
        self.current_order_idx = 0
        self.current_order = self.orders[0]
        self.current_time = 0

        # 更新堆栈数量和堆高容量
        self.num_stacks = len(self.depot.stacks)
        self.stack_height = self.default_stack_height  # 假设所有 stack 容量一致

        # 定义动作空间（堆栈数 + 1个等待动作），定义状态空间（Box，维度为状态向量长度）
        self.action_space = spaces.Discrete(self.num_stacks + 1)
        state_dim = self._compute_state_dim()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)

        return self._get_state()

    def step(self, action):
        order = self.current_order
        success = False
        removed_container = None

        if action == self.num_stacks:
            # 跳过动作 (等待)
            valid_exists = False
            if order.is_loading:
                # 检查是否有可放置的位置
                for stack in self.depot.stacks:
                    if len(stack.containers) < self.stack_height and (
                            not stack.containers or stack.top_container().size == order.size):
                        valid_exists = True
                        break
            else:
                # 检查是否有对应尺寸的集装箱可卸载
                for stack in self.depot.stacks:
                    top_container = stack.top_container()
                    if top_container and top_container.size == order.size:
                        valid_exists = True
                        break
            # 根据是否存在可行动作来决定跳过惩罚力度
            reward = -2.0 if valid_exists else 0.0
        else:
            stack = self.depot.stacks[action]
            if order.is_loading:
                # 尝试执行装载动作
                if len(stack.containers) < self.stack_height and (
                        not stack.containers or stack.top_container().size == order.size):
                    # 可以装载：创建并加入 Container
                    container = Container(id=-1, size=order.size, grace_period=24)
                    stack.add_container(container)
                    success = True
            else:
                # 尝试执行卸载动作
                top_container = stack.top_container()
                if top_container and top_container.size == order.size:
                    removed_container = stack.remove_top_container()
                    success = True

            # 奖励计算
            if not success:
                # 无效动作惩罚
                reward = -1.0
            elif order.is_loading:
                # 装载成功奖励
                reward = 1.0
            else:
                # 卸载成功奖励
                if removed_container.is_expired():
                    # 集装箱已过期，惩罚
                    reward = -1.0
                else:
                    # 集装箱未过期，根据idle_time占宽限期比例计算奖励
                    ratio = removed_container.idle_time / removed_container.grace_period
                    if ratio > 1.0:
                        ratio = 1.0  # 理论上未过期时ratio<=1
                    reward = 1.0 + 2.0 * ratio

        # 如果动作有效或选择了等待，则推进到下一个订单
        if success or action == self.num_stacks:
            self.current_order_idx += 1
            if self.current_order_idx < len(self.orders):
                self.current_order = self.orders[self.current_order_idx]
            else:
                self.current_order = None

        # 增加时间步，并更新所有堆场中集装箱的idle time
        self.current_time += 1
        self.depot.increment_idle_times()

        done = (self.current_order_idx >= len(self.orders))
        state = self._get_state() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "success": success,
            "skipped": (action == self.num_stacks),
            "expired": int(removed_container is not None and removed_container.is_expired())
        }
        return state, reward, done, info

    def _get_state(self):
        """构造当前环境状态的扁平向量表示，加入未来3个订单信息。"""
        order = self.current_order
        # 当前订单特征：尺寸独热编码4维，优先级（除以3归一化），类型（装载=1或卸载=0）
        order_size_onehot = [int(order.size == s) for s in ['20ft', '40ft', '60ft', '80ft']]
        order_vec = order_size_onehot + [order.priority / 3.0, int(order.is_loading)]

        # === 新增：未来3个订单信息 ===
        future_order_vecs = []
        for offset in range(1, 4):  # 只看接下来的3单
            idx = self.current_order_idx + offset
            if idx < len(self.orders):
                fo = self.orders[idx]
                onehot = [int(fo.size == s) for s in ['20ft', '40ft', '60ft', '80ft']]
                vec = onehot + [fo.priority / 3.0, int(fo.is_loading)]
            else:
                vec = [0, 0, 0, 0, 0.0, 0]  # 长度与order_vec一致，全部补零
            future_order_vecs.extend(vec)

        # 堆场状态特征：每个堆栈的顶部箱闲置比例、顶部箱尺寸独热、堆栈当前高度比例
        stack_vecs = []
        for stack in self.depot.stacks:
            top = stack.top_container()
            if top:
                s = [top.idle_time / top.grace_period] + [int(top.size == sz) for sz in
                                                          ['20ft', '40ft', '60ft', '80ft']]
            else:
                s = [0, 0, 0, 0, 0]
            s.append(len(stack.containers) / self.stack_height)
            stack_vecs.extend(s)
        # test code can delete
        # print("[DEBUG][env] state.shape:", np.array(order_vec + future_order_vecs + stack_vecs, dtype=np.float32).shape,
        #       "len(order_vec):", len(order_vec),
        #       "len(future_order_vecs):", len(future_order_vecs),
        #       "len(stack_vecs):", len(stack_vecs))
        # 拼接所有state
        return np.array(order_vec + future_order_vecs + stack_vecs, dtype=np.float32)

    def get_valid_action_mask(self):
        order = self.current_order
        mask = []
        for stack in self.depot.stacks:
            if order.is_loading:
                valid = (len(stack.containers) < self.stack_height) and (
                            not stack.containers or stack.top_container().size == order.size)
            else:
                top_container = stack.top_container()
                valid = (top_container is not None and top_container.size == order.size)
            mask.append(valid)
        # 只有所有动作都不能选时，wait 才为 True，否则为 False
        if any(mask):
            mask.append(False)
        else:
            mask.append(True)
        return np.array(mask, dtype=bool)

    def _compute_state_dim(self):
        # 订单向量维度：4（尺寸）+ 1（优先级）+ 1（类型） = 6
        order_dim = 6
        # 每个堆栈向量维度：1（闲置比率）+ 4（尺寸独热）+ 1（高度比率） = 6
        stack_dim = 6
        # 总状态维度 = 订单部分 + 所有堆栈部分
        return order_dim + self.num_stacks * stack_dim

    def render(self, mode="human"):
        """打印当前堆场状态（用于调试）。"""
        print(f"==== DEPOT STATE at order {self.current_order_idx} ====")
        print(self.depot)
