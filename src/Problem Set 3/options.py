# This file contains the options that you should modify to solve Question 2

# Seek the near terminal state via the short dangerous path
def question2_1():
    # TODO: Choose options that would lead to the desired results
    # Since we want to proceed with the short dangerous path
    return {
        "noise": 0.2,  # Default noise for exploration
        "discount_factor": 1,  # Full consideration of future rewards rather than immediate ones
        # Negative living reward to encourage quick termination and avoid the safe path
        "living_reward": -3
    }


# Seek the near terminal state via the long safe path
def question2_2():
    # TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.2,  # Keep the same noise just for a degree of randomness
        # Lower discount factor to prioritize immediate rewards & move away from -10 state
        "discount_factor": 0.3,
        "living_reward": -0.2  # Slightly negative living reward to discourage unnecessary moves and prioritize the safe path over the dangerous one
    }


# Seek the far terminal state via the short dangerous path
def question2_3():
    # TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.2,
        "discount_factor": 1,  # This is to fully consider future rewards
        # Negative living reward to choose the dangerous path but bigger than in (1) because we're seeking the far state
        "living_reward": -2
    }


# Seek the far terminal state via the long safe path
def question2_4():
    # TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.2,
        # Long safe path means that we'll want to look for the future rewards over the immediate ones
        "discount_factor": 1,
        # Slightly negative to discourage unnecessary moves, still big to ensure safe path
        "living_reward": -0.2
    }


# Avoid any terminal state and keep the episode going on forever
def question2_5():
    # TODO: Choose options that would lead to the desired results
    return {
        # [Deterministic Environment since we want the agent to move as we like which in this case means to keep exploring due to the positive rewards on the way]
        "noise": 0,
        "discount_factor": 1,
        "living_reward": 0.1  # Positive living reward to encourage continuous exploration
    }


# Seek any terminal state (even ones with the -10 penalty) and try to end the episode quickly
def question2_6():
    # TODO: Choose options that would lead to the desired results
    return {
        # Remove randomness by setting noise to zero [Deterministic behavior since we want to finish as fast as possible]
        "noise": 0,
        "discount_factor": 1,
        "living_reward": -50  # Highly negative living reward to encourage quick termination
    }
