import os
import torch
import pandas as pd
from datetime import datetime, timedelta

from V3_RL.sim.depot import Depot
from V3_RL.sim.container import Container
from V3_RL.data.order_simulator import Order
from V3_RL.agent.dqn_agent import AttentionDQNAgent
from V3_RL.evaluation.test_DRL_Greedy import run_full_evaluation  # ✅ 新封装的入口函数

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

# === 手动指定的最优调度方案 ===
optimal_map = {
    0: 'stack 1',
    1: 'stack 0',
    2: 'stack 0',
    3: 'stack 1',
    4: 'stack 1',
    5: 'stack 3',
    6: 'stack 4',
    7: 'stack 3',
    8: 'stack 3',
    9: 'stack 4'
}

# === 打印订单表格 ===
def print_depot_order(orders):
    df = pd.DataFrame([{
        'Order ID': o.order_id,
        'Size': o.size,
        'Priority': o.priority,
        'Type': 'Loading' if o.is_loading else 'Unloading'
    } for o in orders])
    print("\n🧾 Order Overview:")
    print(df)

# === 打印初始堆场状态 ===
def print_depot_state(depot):
    print(f"\n🏭 Initial Depot State ({len(depot.stacks)} stacks):")
    for idx, stack in enumerate(depot.stacks):
        containers_desc = ", ".join(
            f"[id={c.id}, size={c.size}, idle={c.idle_time}/{c.grace_period}]"
            for c in stack.containers
        ) or "EMPTY"
        print(f" Stack {idx}: {containers_desc}")

# === 主函数入口 ===
if __name__ == "__main__":
    print_depot_state(init_depot)
    print_depot_order(orders)

    model_path = os.path.join(os.path.dirname(__file__), 'trained_dqn_best.pth')
    run_full_evaluation(init_depot, orders, optimal_map, model_path)
