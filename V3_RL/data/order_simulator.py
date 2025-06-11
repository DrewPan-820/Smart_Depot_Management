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
        self.is_loading = is_loading  # True 表示放入箱子，False 表示出库订单

    def __repr__(self):
        task = "LOAD" if self.is_loading else "UNLOAD"
        return f"Order(id={self.order_id}, size={self.size}, priority={self.priority}, task={task})"

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
        is_loading = random.random() < 0.3  # 假设 30% 是 loading 任务
        order = Order(i, size, created_time, deadline_time, priority, is_loading)
        orders.append(order)

    return orders

def generate_valid_orders(depot, num_orders, start_datetime=None):
    orders = []
    current_time = start_datetime or datetime(2025, 1, 1, 8, 0)

    for i in range(num_orders):
        is_loading = random.random() < 0.5  # 50%入库，50%出库

        if is_loading:
            # 查找可以入库的尺寸（空位并顶部尺寸一致 or 空堆栈）
            stack_sizes = {
                stack.top_container().size
                for stack in depot.stacks
                if len(stack.containers) < depot.stack_height and
                   (not stack.containers or stack.top_container())
            }
            size = random.choice(list(stack_sizes)) if stack_sizes else random.choice(['20ft', '40ft', '60ft', '80ft'])
        else:
            # 出库时查找当前仓库顶部容器的尺寸
            available_sizes = {
                stack.top_container().size
                for stack in depot.stacks
                if stack.top_container() is not None
            }
            if not available_sizes:
                is_loading = True
                size = random.choice(['20ft', '40ft', '60ft', '80ft'])
            else:
                size = random.choice(list(available_sizes))

        created_time = current_time + timedelta(minutes=random.randint(0, 30))
        deadline_time = created_time + timedelta(minutes=random.randint(60, 180))
        priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]

        order = Order(order_id=i, size=size, created_time=created_time, deadline_time=deadline_time, priority=priority)
        order.is_loading = is_loading
        orders.append(order)

    return orders

