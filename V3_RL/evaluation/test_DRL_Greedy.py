import time
import numpy as np
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
from V3_RL.sim.container import Container
from V3_RL.env.depot_env import DepotEnv
from v1_greedy.baseline.greedy_allocator import GreedyAllocator
from V3_RL.agent.dqn_agent import AttentionDQNAgent


def run_greedy_eval(init_depot, orders, optimal_map):
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
    return depot


def run_rl_eval(init_depot, orders, model_path, optimal_map):
    print("\n[RL EVALUATION]")
    depot = deepcopy(init_depot)
    env = DepotEnv(depot=depot, orders=deepcopy(orders))
    order_dim, stack_dim, n_heads, hidden_dim = 6, 6, 4, 128

    agent = AttentionDQNAgent(stack_input_dim=stack_dim, order_input_dim=order_dim,
                              hidden_dim=hidden_dim, n_heads=n_heads)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()

    for order in orders:
        state = env._get_state()
        order_vec = state[:order_dim]
        stack_vecs = state[order_dim:].reshape(env.num_stacks, stack_dim)

        valid_actions_mask = env.get_valid_action_mask()
        action = agent.act(order_vec, stack_vecs, valid_actions_mask)

        actual = f"stack {action}" if action < env.num_stacks else "wait"
        expected = optimal_map[order.order_id]
        print(f"Order {order.order_id} {'LOAD' if order.is_loading else 'UNLOAD'} → selected: {actual}, expected: {expected}")
        env.step(action)

    return env.depot


def evaluate_metrics(depot, algorithm_func, orders):
    depot = deepcopy(depot)
    start_time = time.time()
    algorithm_func(depot, orders)
    runtime = time.time() - start_time

    total_containers = sum(len(stack.containers) for stack in depot.stacks)
    overdue_containers = sum(
        1 for stack in depot.stacks for c in stack.containers if c.idle_time > c.grace_period
    )
    avg_idle_ratio = np.mean([
        c.idle_time / c.grace_period for stack in depot.stacks for c in stack.containers
    ]) if total_containers > 0 else 0

    metrics_result = {
        'overdue_containers': round(overdue_containers, 5),
        'avg_idle_ratio': round(avg_idle_ratio, 5),
        'runtime_sec': round(runtime, 5)
    }

    return metrics_result


def plot_metrics(greedy_metrics, rl_metrics):
    labels = ['Overdue Containers', 'Average Idle Ratio', 'Runtime (sec)']
    greedy_values = [
        greedy_metrics['overdue_containers'],
        greedy_metrics['avg_idle_ratio'],
        greedy_metrics['runtime_sec']
    ]
    rl_values = [
        rl_metrics['overdue_containers'],
        rl_metrics['avg_idle_ratio'],
        rl_metrics['runtime_sec']
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, greedy_values, width, label='Greedy', color='skyblue')
    bars2 = ax.bar(x + width/2, rl_values, width, label='RL', color='lightgreen')

    ax.set_ylabel('Metric Values')
    ax.set_title('Comparison of Greedy vs RL Algorithms')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.5f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def run_full_evaluation(init_depot, orders, optimal_map, model_path):
    greedy_depot = run_greedy_eval(init_depot, orders, optimal_map)
    rl_depot = run_rl_eval(init_depot, orders, model_path, optimal_map)

    greedy_metrics = evaluate_metrics(greedy_depot, lambda d, o: None, orders)
    rl_metrics = evaluate_metrics(rl_depot, lambda d, o: None, orders)

    print("\n📊 Greedy Metrics:")
    for k, v in greedy_metrics.items():
        print(f"{k}: {v:.5f}")

    print("\n📊 RL Metrics:")
    for k, v in rl_metrics.items():
        print(f"{k}: {v:.5f}")

    plot_metrics(greedy_metrics, rl_metrics)
