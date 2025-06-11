# reward_scheme.py

def compute_reward(order, stack, success, removed_container=None):
    """
    Compute reward based on order type, priority, and container condition.

    Args:
        order: the order object, with fields like is_loading, priority, deadline_timestamp
        stack: the stack object used
        success: whether the operation succeeded (bool)
        removed_container: if unloading, the container removed from the stack

    Returns:
        A scalar float reward
    """
    if not success:
        return -1.0  # harsh penalty for invalid action

    if order.is_loading:
        # Optional: bonus for placing on matching-size stack
        return 1.0
    else:
        # unloading
        if removed_container is None:
            return -1.0

        ratio = removed_container.idle_time / removed_container.grace_period
        # reward highest when ratio ~ 1 (perfectly timed), lower when early or late
        reward = 1.0 - abs(1.0 - ratio)  # can also use: 1 - tanh(...)
        return max(reward, -1.0)  # keep in safe bounds


