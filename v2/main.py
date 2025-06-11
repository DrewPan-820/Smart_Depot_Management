import json
import matplotlib.pyplot as plt
from copy import deepcopy

from env.depot_simulator import Depot
from data.order_simulator import generate_mock_orders
from baseline.greedy_allocator import GreedyAllocator
from evaluation.metrics import EvaluationMetrics
from env.containers_simulator import Container

def run_greedy_simulation(alpha=0.5, beta=0.5, current_time=0, orders=None, show=True):
    depot = Depot(num_stacks=12, stack_height=6)
    if orders is None:
        orders = generate_mock_orders(30)
    metrics = EvaluationMetrics()

    if show:
        print("\n============ Initial Orders ============")
        print(orders)

    allocator = GreedyAllocator(depot, alpha=alpha, beta=beta)

    if show:
        print("\n============ Initial Depot ============")
        print(depot)

    for order in orders:
        if order.is_loading:
            container = Container(id=-1, size=order.size, grace_period=24, idle_time=0)
            allocator.place_container(container)
        else:
            stack = allocator.select_stack_for_order(order)
            if stack:
                container = stack.remove_top_container()
                metrics.record_order_fulfillment(order, fulfill_time=current_time)
                metrics.record_used_container(container)
            else:
                metrics.record_no_match_order()
                if order.priority == 3:
                    metrics.record_high_priority_miss(order)
        depot.increment_idle_times()

    metrics.record_expired_containers(depot)
    if show:
        print("\n============ Final Depot ============")
        print(depot)
        print("\n============ Metrics Report ============")
        metrics.report()

    return metrics

if __name__ == "__main__":
    # 读取 RL 调优得到的最优 alpha/beta
    with open("./RL/best_params.json", "r") as f:
        best = json.load(f)
        rl_alpha = best["alpha"]
        rl_beta = best["beta"]

    # 用相同订单做对比
    orders = generate_mock_orders(60)

    print("\n=== Baseline (0.5 / 0.5) ===")
    m1 = run_greedy_simulation(alpha=0.5, beta=0.5, orders=deepcopy(orders), show=False)

    print("\n=== RL Tuned ({:.2f} / {:.2f}) ===".format(rl_alpha, rl_beta))
    m2 = run_greedy_simulation(alpha=rl_alpha, beta=rl_beta, orders=deepcopy(orders), show=False)

    # 输出指标对比
    print("\n>>> Baseline Metrics:")
    m1.report()
    print("\n>>> RL Tuned Metrics:")
    m2.report()

    # 可视化对比
    labels = ["on-time", "expired", "priority missed", "efficiency"]
    baseline = [
        m1.fulfilled_on_time,
        m1.total_expired,
        m1.ignored_high_priority,
        m1.compute_container_efficiency()
    ]
    rl_tuned = [
        m2.fulfilled_on_time,
        m2.total_expired,
        m2.ignored_high_priority,
        m2.compute_container_efficiency()
    ]

    x = range(len(labels))
    plt.bar(x, baseline, width=0.4, label="Baseline", align="center")
    plt.bar([i + 0.4 for i in x], rl_tuned, width=0.4, label="RL Tuned", align="center")
    plt.xticks([i + 0.2 for i in x], labels)
    plt.title("Baseline vs RL Tuned Strategy Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
