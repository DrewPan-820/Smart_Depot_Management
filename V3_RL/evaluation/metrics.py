import time
from copy import deepcopy
import numpy as np

def evaluate_metrics(depot, algorithm_func, orders):
    depot = deepcopy(depot)  # 防止修改原始状态
    start_time = time.time()

    # 执行算法
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
        'overdue_containers': overdue_containers,
        'avg_idle_ratio': avg_idle_ratio,
        'runtime_sec': runtime
    }

    return metrics_result
