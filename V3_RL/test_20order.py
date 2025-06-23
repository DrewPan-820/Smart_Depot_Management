import os
import torch
import pandas as pd
from datetime import datetime, timedelta

from V3_RL.sim.depot import Depot
from V3_RL.sim.container import Container
from V3_RL.data.order_simulator import Order
from V3_RL.evaluation.test_DRL_Greedy import run_full_evaluation  # 你的入口

# === 初始堆场状态 ===
init_depot = Depot(num_stacks=5, stack_height=4)
init_depot.stacks[0].containers = [
    Container(0, '40ft', idle_time=12, grace_period=24)
]
init_depot.stacks[1].containers = [
    Container(1, '20ft', idle_time=20, grace_period=24),
    Container(2, '20ft', idle_time=14, grace_period=24)
]
init_depot.stacks[2].containers = [
    Container(3, '60ft', idle_time=4, grace_period=24)
]
init_depot.stacks[3].containers = []
init_depot.stacks[4].containers = [
    Container(4, '40ft', idle_time=22, grace_period=24),
    Container(5, '40ft', idle_time=19, grace_period=24),
    Container(6, '40ft', idle_time=5, grace_period=24)
]

# === 订单列表（含2次必须wait）===
start = datetime(2025, 1, 1, 9, 0)
orders = [
    Order(0, '20ft', start, start + timedelta(minutes=12), 2, False),   # stack 1
    Order(1, '40ft', start, start + timedelta(minutes=6), 3, False),    # stack 0
    Order(2, '60ft', start, start, 1, True),                            # stack 0
    Order(3, '20ft', start, start + timedelta(minutes=4), 3, False),    # stack 1
    # ====== 必须wait（此时无20ft可操作，假设设计这样） ======
    Order(4, '20ft', start, start + timedelta(minutes=5), 2, False),    # wait
    Order(5, '40ft', start, start, 1, True),                            # stack 1
    Order(6, '40ft', start, start + timedelta(minutes=8), 2, False),    # stack 4
    Order(7, '20ft', start, start, 2, True),                            # stack 3
    Order(8, '20ft', start, start + timedelta(minutes=3), 2, False),    # stack 3
    Order(9, '60ft', start, start + timedelta(minutes=18), 1, False),   # stack 2
    Order(10, '40ft', start, start + timedelta(minutes=9), 2, False),   # stack 4
    Order(11, '20ft', start, start + timedelta(minutes=7), 1, True),    # stack 3
    Order(12, '20ft', start, start + timedelta(minutes=11), 2, False),  # stack 3
    # ====== 必须wait（此时无20ft可操作，假设设计这样） ======
    Order(13, '20ft', start, start + timedelta(minutes=8), 3, False),   # wait
    Order(14, '40ft', start, start, 2, True),                           # stack 4
    Order(15, '20ft', start, start, 1, True),                           # stack 3
    Order(16, '20ft', start, start + timedelta(minutes=15), 2, False),  # stack 3
    Order(17, '40ft', start, start + timedelta(minutes=25), 3, False),  # stack 4
    Order(18, '20ft', start, start, 1, True),                           # stack 3
    Order(19, '40ft', start, start, 2, True),                           # stack 4
]

# === 严谨的最优解（人工根据堆场和订单状态一一推理） ===
optimal_map = {
    0:  'stack 1',    # UNLOAD
    1:  'stack 0',    # UNLOAD
    2:  'stack 0',    # LOAD
    3:  'stack 1',    # UNLOAD
    4:  'wait',       # 必须wait
    5:  'stack 1',    # LOAD
    6:  'stack 4',    # UNLOAD
    7:  'stack 3',    # LOAD
    8:  'stack 3',    # UNLOAD
    9:  'stack 2',    # UNLOAD
    10: 'stack 4',    # UNLOAD
    11: 'stack 3',    # LOAD
    12: 'stack 3',    # UNLOAD
    13: 'wait',       # 必须wait
    14: 'stack 4',    # LOAD
    15: 'stack 3',    # LOAD
    16: 'stack 3',    # UNLOAD
    17: 'stack 4',    # UNLOAD
    18: 'stack 3',    # LOAD
    19: 'stack 4',    # LOAD
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

if __name__ == "__main__":
    print_depot_state(init_depot)
    print_depot_order(orders)

    model_path = os.path.join(os.path.dirname(__file__), 'trained_dqn_best.pth')
    run_full_evaluation(init_depot, orders, optimal_map, model_path)
