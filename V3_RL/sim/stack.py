from sim.container import Container

class Stack:
    def __init__(self, stack_id):
        self.stack_id = stack_id
        self.containers = []

    def add_container(self, container: Container):
        self.containers.append(container)

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

