import sys
import os
import torch
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta

from V3_RL.env.depot_env import DepotEnv
from V3_RL.sim.depot import Depot
from V3_RL.sim.container import Container
from V3_RL.data.order_simulator import Order
from v1_greedy.baseline.greedy_allocator import GreedyAllocator
from V3_RL.evaluation.metrics import EvaluationMetrics
from V3_RL.agent.dqn_agent import AttentionDQNAgent
from V3_RL.env.AttentionQNetwork import AttentionQNetwork

# === 设置初始堆场状态 ===
init_depot = Depot(num_stacks=5, stack_height=4)
init_depot.stacks[0].containers = [Container(0, '40ft', idle_time=12, grace_period=24)]
init_depot.stacks[1].containers = [Container(1, '20ft', idle_time=20, grace_period=24),
                                   Container(2, '20ft', idle_time=14, grace_period=24)]
init_depot.stacks[2].containers = [Container(3, '60ft', idle_time=4, grace_period=24)]
init_depot.stacks[3].containers = []
init_depot.stacks[4].containers = [Container(4, '40ft', idle_time=22, grace_period=24),
                                   Container(5, '40ft', idle_time=19, grace_period=24),
                                   Container(6, '40ft', idle_time=5, grace_period=24)]

# === 构造订单列表 ===
start = datetime(2025, 1, 1, 9, 0)
orders = [
    Order(0, '20ft', start, start + timedelta(minutes=12), 2, False),
    Order(1, '40ft', start, start + timedelta(minutes=6), 3, False),
    Order(2, '60ft', start, start, 1, True),
    Order(3, '20ft', start, start + timedelta(minutes=4), 3, False),
    Order(4, '40ft', start, start, 1, True),
    Order(5, '20ft', start, start, 2, True),
    Order(6, '40ft', start, start + timedelta(minutes=10), 2, False),
    Order(7, '20ft', start, start + timedelta(minutes=2), 3, False),
    Order(8, '20ft', start, start, 1, True),
    Order(9, '40ft', start, start + timedelta(minutes=25), 2, False),
]

optimal_map = {
    0: 'stack 1', 1: 'stack 0', 2: 'stack 2', 3: 'wait', 4: 'stack 4',
    5: 'stack 3', 6: 'stack 4', 7: 'wait', 8: 'stack 3', 9: 'stack 4'
}

# === 打印订单 DataFrame ===
df = pd.DataFrame([{
    'Order ID': o.order_id,
    'Size': o.size,
    'Priority': o.priority,
    'Type': 'Loading' if o.is_loading else 'Unloading',
    'Deadline(min)': int(o.deadline_timestamp - o.created_timestamp / 60)
} for o in orders])
print("\n🧾 Order Overview:")
print(df)

# === Greedy 测试函数 ===
def run_greedy():
    print("\n[GREEDY EVALUATION]")
    depot = deepcopy(init_depot)
    allocator = GreedyAllocator(depot, alpha=0.6, beta=0.4)

    for order in orders:
        if order.is_loading:
            container = Container(id=-1, size=order.size, grace_period=24)
            allocator.place_container(container)
            print(f"Order {order.order_id} LOAD → placed [AUTO]")
        else:
            selected = allocator.select_stack_for_order(order)
            actual = f"stack {selected.stack_id}" if selected else "None"
            expected = optimal_map[order.order_id]
            print(f"Order {order.order_id} UNLOAD → selected: {actual}, expected: {expected}")
            if selected:
                selected.remove_top_container()
        depot.increment_idle_times()

# === RL 测试函数 ===
def run_rl(model_path):
    print("\n[RL EVALUATION]")
    env = DepotEnv(depot=deepcopy(init_depot), orders=deepcopy(orders))
    order_dim, stack_dim, n_heads, hidden_dim = 6, 6, 4, 128

    agent = AttentionDQNAgent(stack_input_dim=stack_dim, order_input_dim=order_dim,
                              hidden_dim=hidden_dim, n_heads=n_heads)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()

    for order in orders:
        state = env._get_state()
        order_vec = state[:order_dim]
        stack_vecs = state[order_dim:].reshape(env.num_stacks, stack_dim)

        valid_actions_mask = np.array([
            (len(stack.containers) < env.stack_height if order.is_loading
             else stack.top_container() and stack.top_container().size == order.size)
            for stack in env.depot.stacks
        ])

        action = agent.act(order_vec, stack_vecs, valid_actions_mask)
        actual = f"stack {action}" if action < env.num_stacks else "wait"
        expected = optimal_map[order.order_id]
        print(f"Order {order.order_id} {'LOAD' if order.is_loading else 'UNLOAD'} → selected: {actual}, expected: {expected}")
        env.step(action)

# === 主入口 ===
if __name__ == "__main__":
    run_greedy()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'V3_RL', 'trained_dqn.pth')
    run_rl(model_path)
