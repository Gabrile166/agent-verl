DEFAULT_SUCCESS_REWARD = 10.0


def compute_binary_success_reward(won, success_reward: float = DEFAULT_SUCCESS_REWARD) -> float:
    return float(success_reward) * float(won)
