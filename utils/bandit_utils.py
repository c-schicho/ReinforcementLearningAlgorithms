from typing import Dict, Union

import numpy as np
from tqdm import tqdm


def run_experiment(
        agent_class,
        bandit_class,
        seed: int = 9999,
        n_runs: int = 2_000,
        n_steps: int = 1_000,
        agent_args: Union[Dict, None] = None,
        bandit_args: Union[Dict, None] = None
):
    if agent_args is None:
        agent_args = dict()

    if bandit_args is None:
        bandit_args = dict()

    agent_args_string = ','.join([f"{key}={val}" for key, val in agent_args.items()])
    print(f"Start running experiment {agent_class.__name__}({agent_args_string})")

    rewards = np.zeros((n_runs, n_steps))
    exp_seed_sequence = np.random.SeedSequence(seed)

    curr_run = 1

    for run in tqdm(range(1, n_runs + 1), total=n_runs, ncols=90):
        agent_seed, bandit_seed = exp_seed_sequence.spawn(2)
        agent_args["seed"] = agent_seed
        bandit_args["seed"] = bandit_seed

        agent = agent_class(**agent_args)
        bandit = bandit_class(**bandit_args)

        for step in range(n_steps):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.update_estimates(action, reward)
            rewards[run - 1, step] = reward

        curr_run += 1

    return rewards
