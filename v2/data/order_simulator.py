import random
from datetime import datetime, timedelta


class Order:
    def __init__(self, order_id, size, created_time, deadline_time, priority, is_loading=False):
        self.order_id = order_id
        self.size = size
        self.created_time = created_time.strftime('%Y-%m-%d %H:%M')
        self.deadline = deadline_time.strftime('%Y-%m-%d %H:%M')
        self.created_timestamp = int(created_time.timestamp() // 60)
        self.deadline_timestamp = int(deadline_time.timestamp() // 60)
        self.priority = priority
        self.is_loading = is_loading

    def __repr__(self):
        type_str = "LOAD" if self.is_loading else "UNLOAD"
        return (f"Order(id={self.order_id}, type={type_str}, size={self.size}, created={self.created_time}, "
                f"deadline={self.deadline}, priority={self.priority})")


def generate_mock_orders(num_orders, start_datetime=None):
    if start_datetime is None:
        start_datetime = datetime(2025, 1, 1, 8, 0)

    orders = []
    for i in range(num_orders):
        size = random.choice(['20ft', '40ft', '60ft', '80ft'])
        created_delay = random.randint(0, 60)
        created_time = start_datetime + timedelta(minutes=created_delay)
        deadline_offset = random.randint(60, 180)
        deadline_time = created_time + timedelta(minutes=deadline_offset)
        priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        is_loading = random.random() < 0.1  # 20% of orders are loading
        order = Order(order_id=i, size=size, created_time=created_time,
                      deadline_time=deadline_time, priority=priority, is_loading=is_loading)
        orders.append(order)
    return orders


if __name__ == "__main__":
    orders = generate_mock_orders(30)
    orders.sort(key=lambda x: x.deadline)
    for order in orders:
        print(order)
