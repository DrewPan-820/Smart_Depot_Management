import sys
import os
import torch
from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt

# === 添加模块路径（允许通过包方式导入） ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# === 包方式导入 ===
from v1_greedy.baseline.greedy_allocator import GreedyAllocator
from v1_greedy.env.depot_simulator import Depot as OldDepot
from V3_RL.env.depot_env import DepotEnv
from V3_RL.sim.depot import Depot
from V3_RL.sim.container import Container
from V3_RL.agent.dqn_agent import DQNAgent
from V3_RL.data.order_simulator import generate_mock_orders
from V3_RL.evaluation.metrics import EvaluationMetrics


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


def evaluate_dqn(orders, depot, model_path):
    env = DepotEnv(depot=depot, orders=deepcopy(orders))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()

    metrics = EvaluationMetrics()
    env.current_step = 0

    for order in orders:
        state = env._get_state()
        action = agent.act(state)

        if order.is_loading:
            stack = env.depot.stacks[action]
            if len(stack.containers) < env.stack_height and \
                    (not stack.containers or stack.top_container().size == order.size):
                container = Container(id=-1, size=order.size, grace_period=24)
                stack.add_container(container)
        else:
            stack = env.depot.stacks[action]
            if stack.top_container() and stack.top_container().size == order.size:
                container = stack.remove_top_container()
                metrics.record_order_fulfillment(order, fulfill_time=0)
                metrics.record_used_container(container)
            else:
                metrics.record_no_match_order()
                if order.priority == 3:
                    metrics.record_high_priority_miss(order)

        env.depot.increment_idle_times()

    metrics.record_expired_containers(env.depot)
    return metrics


if __name__ == "__main__":
    print("\n📦 Generating fixed test orders...")
    orders = generate_mock_orders(13, start_datetime=datetime(2025, 1, 1, 8, 0))
    initial_depot = Depot(num_stacks=8, stack_height=4)

    print("\n=== 🟢 Evaluating Greedy ===")
    g_metrics = evaluate_greedy(deepcopy(orders), deepcopy(initial_depot))
    g_metrics.report()

    print("\n=== 🤖 Evaluating RL Agent ===")
    model_path = os.path.join(project_root, "V3_RL", "trained_dqn.pth")
    rl_metrics = evaluate_dqn(deepcopy(orders), deepcopy(initial_depot), model_path=model_path)
    rl_metrics.report()

    # 📊 可视化对比
    labels = ["on-time", "expired", "priority missed", "efficiency"]
    greedy_vals = [
        g_metrics.fulfilled_on_time,
        g_metrics.expired_containers,
        g_metrics.ignored_high_priority,
        g_metrics.compute_container_efficiency()
    ]
    rl_vals = [
        rl_metrics.fulfilled_on_time,
        rl_metrics.expired_containers,
        rl_metrics.ignored_high_priority,
        rl_metrics.compute_container_efficiency()
    ]

    x = range(len(labels))
    plt.bar(x, greedy_vals, width=0.4, label="Greedy", align="center")
    plt.bar([i + 0.4 for i in x], rl_vals, width=0.4, label="RL (DQN)", align="center")
    plt.xticks([i + 0.2 for i in x], labels)
    plt.title("Greedy vs RL Strategy Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
