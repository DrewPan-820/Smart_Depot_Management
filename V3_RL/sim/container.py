class Container:
    def __init__(self, id, size, grace_period=24, idle_time=0):
        self.id = id
        self.size = size
        self.grace_period = grace_period
        self.idle_time = idle_time

    def is_expired(self):
        return self.idle_time > self.grace_period

    def __repr__(self):
        return f"Container(id={self.id}, size={self.size}, idle={self.idle_time}, grace={self.grace_period})"
