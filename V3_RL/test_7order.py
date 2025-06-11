import sys
import os
import torch
from copy import deepcopy
from V3_RL.env.depot_env import DepotEnv
from V3_RL.sim.depot import Depot
from V3_RL.sim.container import Container
# from V3_RL.agent.dqn_agent import DQNAgent
from V3_RL.data.order_simulator import Order
from v1_greedy.baseline.greedy_allocator import GreedyAllocator
from v1_greedy.env.depot_simulator import Depot as GreedyDepot
from V3_RL.evaluation.metrics import EvaluationMetrics
from datetime import datetime, timedelta
import numpy as np

# 设定固定 depot 状态
init_depot = Depot(num_stacks=5, stack_height=4)

# 手动设定每个 stack 内容，加入 grace_period
init_depot.stacks[0].containers = [Container(0, '40ft', idle_time=10, grace_period=24), Container(1, '40ft', idle_time=6, grace_period=24)]
init_depot.stacks[1].containers = [Container(2, '20ft', idle_time=4, grace_period=24)]
init_depot.stacks[2].containers = [Container(3, '60ft', idle_time=23, grace_period=24)]
init_depot.stacks[3].containers = []
init_depot.stacks[4].containers = [Container(4, '20ft', idle_time=21, grace_period=24), Container(5, '20ft', idle_time=18, grace_period=24)]

# 模拟一组订单
start = datetime(2025, 1, 1, 9, 0)
orders = [
    Order(0, '20ft', start, start + timedelta(minutes=12), 2, False),
    Order(1, '40ft', start, start + timedelta(minutes=30), 1, False),
    Order(2, '60ft', start, start, 2, True),
    Order(3, '20ft', start, start + timedelta(minutes=8), 3, False),
    Order(4, '40ft', start, start, 1, True),
    Order(5, '20ft', start, start, 2, True),
    Order(6, '20ft', start, start + timedelta(minutes=6), 3, False),
]

# 最优策略参考（手动推导）
optimal_map = {
    0: 'stack 1',
    1: 'stack 0',
    2: 'stack 2',
    3: 'stack 4',
    4: 'stack 0',
    5: 'stack 3',
    6: 'stack 4',
}

# Greedy 策略测试
def run_greedy(init_depot):
    print("\n[GREEDY EVALUATION]")
    depot = deepcopy(init_depot)
    allocator = GreedyAllocator(depot, alpha=0.6, beta=0.4)
    for order in orders:
        if order.is_loading:
            container = Container(id=-1, size=order.size, grace_period=24)
            allocator.place_container(container)
            print(f"Order {order.order_id} LOADING → placed to [AUTO]")
        else:
            selected = allocator.select_stack_for_order(order)
            actual = f"stack {selected.stack_id}" if selected else "None"
            expected = optimal_map[order.order_id]
            print(f"Order {order.order_id} UNLOADING → selected: {actual}, expected: {expected}")
            if selected:
                selected.remove_top_container()
        depot.increment_idle_times()

# RL 策略测试
from V3_RL.agent.dqn_agent import AttentionDQNAgent  # 替换 DQNAgent
from V3_RL.env.AttentionQNetwork import AttentionQNetwork  # 如需直接使用模型结构
def run_rl(model_path):
    print("\n[RL EVALUATION]")
    env = DepotEnv(depot=deepcopy(init_depot), orders=deepcopy(orders))

    order_input_dim = 6
    stack_input_dim = 6
    n_heads = 4
    hidden_dim = 128

    agent = AttentionDQNAgent(
        stack_input_dim=stack_input_dim,
        order_input_dim=order_input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads
    )

    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()

    for order in orders:
        state = env._get_state()
        order_vec = state[:order_input_dim]
        stack_vecs = state[order_input_dim:].reshape(env.num_stacks, stack_input_dim)

        valid_actions_mask = np.array([
            (len(stack.containers) < env.stack_height if order.is_loading
             else stack.top_container() and stack.top_container().size == order.size)
            for stack in env.depot.stacks
        ])

        action = agent.act(order_vec, stack_vecs, valid_actions_mask)
        actual = f"stack {action}"
        expected = optimal_map[order.order_id]
        print(
            f"Order {order.order_id} {'LOAD' if order.is_loading else 'UNLOAD'} → selected: {actual}, expected: {expected}")
        env.step(action)


def evaluate_greedy(orders, depot):
    allocator = GreedyAllocator(depot, alpha=0.6, beta=0.4)
    metrics = EvaluationMetrics()

    for order in orders:
        if order.is_loading:
            container = Container(id=-1, size=order.size, grace_period=24)
            allocator.place_container(container)
        else:
            stack = allocator.select_stack_for_order(order)
            if stack:
                container = stack.remove_top_container()
                metrics.record_order_fulfillment(order, fulfill_time=0)
                metrics.record_used_container(container)
            else:
                metrics.record_no_match_order()
                if order.priority == 3:
                    metrics.record_high_priority_miss(order)
        depot.increment_idle_times()

    metrics.record_expired_containers(depot)
    return metrics

if __name__ == "__main__":
    # g_metrics = evaluate_greedy(deepcopy(orders), deepcopy(init_depot))
    # g_metrics.report()
    run_greedy(init_depot)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'V3_RL', 'trained_dqn.pth')
    run_rl(model_path=model_path)
