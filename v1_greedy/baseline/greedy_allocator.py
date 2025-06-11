from v1_greedy.env.depot_simulator import Depot
from v1_greedy.data.order_simulator import generate_mock_orders
from v1_greedy.env.containers_simulator import Container

class GreedyAllocator:
    def __init__(self, depot: Depot, alpha=0.5, beta=0.5):
        self.depot = depot
        self.alpha = alpha  # Weight for priority
        self.beta = beta    # Weight for deadline urgency

    @staticmethod
    def urgency_score(container):
        return container.idle_time / container.grace_period

    def order_urgency(self, order, current_time, max_priority=3, max_deadline_gap=100):
        p_norm = order.priority / max_priority
        remaining_time = order.deadline_timestamp - current_time
        r_norm = max(0, 1 - (remaining_time / max_deadline_gap))
        return self.alpha * p_norm + self.beta * r_norm

    def select_stack_for_order(self, order):
        candidate_stacks = [
            stack for stack in self.depot.stacks
            if stack.top_container() and stack.top_container().size == order.size
        ]

        if not candidate_stacks:
            print(f"No available stack for order {order.order_id} (size {order.size})")
            return None

        def sort_key(stack):
            return -self.urgency_score(stack.top_container())

        selected_stack = sorted(candidate_stacks, key=sort_key)[0]
        return selected_stack

    def place_container(self, container):
        candidate_stacks = [
            stack for stack in self.depot.stacks
            if (not stack.containers or stack.top_container().size == container.size)
            and len(stack.containers) < 4  # assume max height = 4
        ]

        if not candidate_stacks:
            print(f"No suitable stack found for container {container.id}")
            return False

        target_stack = min(candidate_stacks, key=lambda s: len(s.containers))
        target_stack.add_container(container)
        print(f"Container {container.id} (size={container.size}) placed into Stack {target_stack.stack_id}")
        return True

    def fulfill_orders(self, orders, current_time):
        fulfilled = []
        for order in sorted(orders, key=lambda o: -self.order_urgency(o, current_time)):
            if hasattr(order, 'is_loading') and order.is_loading:
                container = Container(id=-1, size=order.size, grace_period=24, idle_time=0)
                self.place_container(container)
                continue

            stack = self.select_stack_for_order(order)
            if stack:
                container = stack.remove_top_container()
                print(f"Order {order.order_id} fulfilled by Container {container.id} (size={container.size}) from Stack {stack.stack_id}")
                fulfilled.append(order)
            else:
                print(f"Order {order.order_id} could not be fulfilled.")
            self.depot.increment_idle_times()
        return fulfilled

if __name__ == "__main__":
    depot = Depot(num_stacks=8, stack_height=4)
    orders = generate_mock_orders(10)
    allocator = GreedyAllocator(depot, alpha=0.6, beta=0.4)

    print("============ Initial Depot ============")
    print(depot)

    print("\n============ Processing Orders ============")
    allocator.fulfill_orders(orders, current_time=0)

    print("\n============ Final Depot ============")
    print(depot)
