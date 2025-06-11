from env.containers_simulator import Container, Stack
import random

class Depot:
    def __init__(self, num_stacks, stack_height):
        self.stacks = [Stack(stack_id=i) for i in range(num_stacks)]
        self.init_stacks(stack_height)

    def init_stacks(self, stack_height):
        container_id = 0

        # 为每种尺寸定义默认 grace period
        size_to_grace = {
            '20ft': 2,
            '40ft': 4,
            '60ft': 6,
            '80ft': 8
        }

        for stack in self.stacks:
            # 每个stack固定一个尺寸
            stack_size = random.choice(list(size_to_grace.keys()))
            grace_period = size_to_grace[stack_size]

            for _ in range(stack_height):
                container = Container(id=container_id, size=stack_size, grace_period=grace_period)
                stack.add_container(container)
                container_id += 1

    def increment_idle_times(self):
        for stack in self.stacks:
            for container in stack.containers:
                container.idle_time += 1

    def __repr__(self):
        return '\n'.join(str(stack) for stack in self.stacks)

# 测试代码
if __name__ == "__main__":
    depot = Depot(num_stacks=8, stack_height=4)
    print(depot)
