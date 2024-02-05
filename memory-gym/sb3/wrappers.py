import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.spaces import Discrete, MultiDiscrete


class ConvertMultiDiscreteToDiscrete(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assuming the multi discrete action space has n discrete spaces each with k actions
        if not isinstance(env.action_space, MultiDiscrete):
            raise ValueError("MultiDiscrete action space required.")
        self.n = env.action_space.nvec.size
        self.k = env.action_space.nvec[0]
        self.action_space = Discrete(self.n * self.k)

    def action(self, act):
        # Convert the discrete action to multi discrete action
        multi_discrete_action = [0] * self.n
        for i in range(self.n):
            if act >= self.k:
                act -= self.k
                multi_discrete_action[i] = 1
        return multi_discrete_action


class AdaptedDictObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Extract low and high bounds from each Box in the Dict space
        lows, highs = [], []
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("Dict observation space required.")
        for space in env.observation_space.spaces.values():
            if not isinstance(space, gym.spaces.Box):
                raise ValueError("Box observation space required.")
            lows.append(space.low.flatten())
            highs.append(space.high.flatten())

        # Determine new shape and bounds for the concatenated space
        total_size = sum(low.size for low in lows)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(lows).reshape(total_size),
            high=np.concatenate(highs).reshape(total_size),
            shape=(total_size,),
            dtype=np.float32,
        )

    def observation(self, observation):
        # Concatenate the observations from each Box space
        return np.concatenate([observation[key].flatten() for key in observation])


class DiscreteToBoxWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "The action space must be discrete"
        n = env.action_space.n
        # Create a Box space with the same number of discrete actions
        # Assuming each discrete action maps to a point in [0, 1] in the Box space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(n,), dtype=np.float32)

    def action(self, action):
        # Map the discrete action to a point in the Box space
        box_action = np.zeros(self.action_space.shape)
        box_action[action] = 1.0
        return box_action
