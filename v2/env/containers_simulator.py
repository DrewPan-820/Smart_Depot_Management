import random
class Container:
    def __init__(self, id, size, grace_period=24, idle_time=None):
        self.id = id
        self.size = size
        self.grace_period = grace_period

        # Simulate realistic idle_time if not provided
        if idle_time is None:
            # Default: randomly assign idle_time in [0, grace_period * 0.8]
            self.idle_time = random.randint(0, int(grace_period * 0.8))
        else:
            # Use user-specified idle_time (e.g., during loading or test)
            self.idle_time = idle_time

    def is_expired(self):
        return self.idle_time > self.grace_period

    def __repr__(self):
        return f"Container(id={self.id}, size={self.size}, idle_time={self.idle_time}, grace_period={self.grace_period})"

class Stack:
    def __init__(self, stack_id):
        self.stack_id = stack_id
        # The stack is represented by a list, with the last element being the top
        self.containers = []

    # Simulate the loading operation of the depot
    def add_container(self, container):
        self.containers.append(container)

    # Simulate the uploading operation of the depot
    def remove_top_container(self):
        if self.containers:
            return self.containers.pop()
        return None

    def top_container(self):
        if self.containers:
            return self.containers[-1]
        return None

    def __repr__(self):
        return f"Stack(id={self.stack_id}, containers={self.containers})"
