import random
from env.depot_simulator import Depot
from data.order_simulator import generate_mock_orders
from baseline.greedy_allocator import GreedyAllocator
from evaluation.metrics import EvaluationMetrics
from env.containers_simulator import Container

class GreedyEnv:
    def __init__(self, num_stacks=8, stack_height=4, num_orders=30):
        self.num_stacks = num_stacks
        self.stack_height = stack_height
        self.num_orders = num_orders

        # Define action space: (alpha, beta) pairs
        self.actions = [
            (1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.4, 0.6), (0.2, 0.8),
            (0.0, 1.0), (0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (0.9, 0.1)
        ]

    def reset(self):
        self.depot = Depot(self.num_stacks, self.stack_height)
        self.orders = generate_mock_orders(self.num_orders)
        self.future_order_stats = self._extract_future_stats(self.orders)
        return self.future_order_stats

    def step(self, action_id):
        alpha, beta = self.actions[action_id]
        allocator = GreedyAllocator(self.depot, alpha=alpha, beta=beta)
        metrics = EvaluationMetrics()

        for order in self.orders:
            if order.is_loading:
                container = Container(id=-1, size=order.size, grace_period=24, idle_time=0)
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
            self.depot.increment_idle_times()

        # Calculate reward from metrics
        metrics.record_expired_containers(self.depot)
        reward = self._compute_reward(metrics)

        # Return new state (we can re-use the same order stats here), reward, done, info
        return self.future_order_stats, reward, True, {"alpha": alpha, "beta": beta, "metrics": metrics}

    def _extract_future_stats(self, orders):
        size_count = {s: 0 for s in ['20ft', '40ft', '60ft', '80ft']}
        priority_count = {1: 0, 2: 0, 3: 0}
        loading_count = 0

        for o in orders:
            size_count[o.size] += 1
            priority_count[o.priority] += 1
            if o.is_loading:
                loading_count += 1

        total = len(orders)
        return [
            size_count['20ft']/total,
            size_count['40ft']/total,
            size_count['60ft']/total,
            size_count['80ft']/total,
            priority_count[1]/total,
            priority_count[2]/total,
            priority_count[3]/total,
            loading_count/total
        ]

    def _compute_reward(self, metrics):
        # reward = +10 * avg_efficiency - 2 * expired - 3 * missed high-priority
        efficiency = metrics.compute_container_efficiency()
        expired = metrics.total_expired
        high_priority_missed = metrics.ignored_high_priority
        return 10 * efficiency - 2 * expired - 3 * high_priority_missed
