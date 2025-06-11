class EvaluationMetrics:
    def __init__(self):
        self.fulfilled_on_time = 0
        self.fulfilled_late = 0
        self.ignored_high_priority = 0
        self.orders_with_no_match = 0
        self.expired_containers = 0
        self.total_idle_time_used = 0
        self.used_container_count = 0

    def record_order_fulfillment(self, order, fulfill_time):
        if fulfill_time <= order.deadline_timestamp:
            self.fulfilled_on_time += 1
        else:
            self.fulfilled_late += 1

    def record_high_priority_miss(self, order):
        self.ignored_high_priority += 1

    def record_no_match_order(self):
        self.orders_with_no_match += 1

    def record_used_container(self, container):
        self.total_idle_time_used += container.idle_time
        self.used_container_count += 1

    def record_expired_containers(self, depot):
        count = 0
        for stack in depot.stacks:
            for c in stack.containers:
                if c.is_expired():
                    count += 1
        self.expired_containers = count

    def compute_container_efficiency(self):
        if self.used_container_count == 0:
            return 0.0
        avg_idle = self.total_idle_time_used / self.used_container_count
        return max(0.0, 1.0 - avg_idle)

    def report(self):
        print("============ Evaluation Report ============")
        print(f"Fulfilled on-time orders     : {self.fulfilled_on_time}")
        print(f"Fulfilled late orders        : {self.fulfilled_late}")
        print(f"Ignored high-priority orders : {self.ignored_high_priority}")
        print(f"Orders with no match         : {self.orders_with_no_match}")
        print(f"Expired containers           : {self.expired_containers}")
        if self.used_container_count > 0:
            avg_idle = self.total_idle_time_used / self.used_container_count
            print(f"Avg idle time (used only)    : {avg_idle:.2f}")
        else:
            print("Avg idle time (used only)    : N/A")
