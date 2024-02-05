import gymnasium as gymnasium
import gym
import numpy as np

class GymFromGymnasium(gym.Env):
    def __init__(self, gymnasium_env):
        super(GymFromGymnasium, self).__init__()
        self.gymnasium_env = gymnasium_env

        # Convert Gymnasium spaces to OpenAI Gym spaces
        self.action_space = self.convert_space(gymnasium_env.action_space)
        self.observation_space = self.convert_space(gymnasium_env.observation_space)

    def convert_space(self, space):
        if isinstance(space, gymnasium.spaces.Discrete):
            return gym.spaces.Discrete(space.n)
        elif isinstance(space, gymnasium.spaces.Box):
            return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            return gym.spaces.MultiDiscrete(space.nvec)
        elif isinstance(space, gymnasium.spaces.Dict):
            return gym.spaces.Dict({key: self.convert_space(subspace) for key, subspace in space.spaces.items()})
        # Add more conversions as necessary for other space types
        else:
            raise NotImplementedError(f"Conversion for space type {type(space)} not implemented")



    def step(self, action):
        observation, reward, done, _, info = self.gymnasium_env.step(action)
        return observation, reward, done, info

    def reset(self, **kwargs):
        return self.gymnasium_env.reset(**kwargs)[0]

    def render(self, mode='human', **kwargs):
        return self.gymnasium_env.render(mode=mode, **kwargs)

    def close(self):
        return self.gymnasium_env.close()

    # def seed(self, seed=None):
    #     return self.gymnasium_env.seed(seed)


class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete), "Action space must be MultiDiscrete"

        # Calculate the size of the discrete action space
        self.action_space = gym.spaces.Discrete(np.prod(env.action_space.nvec))

        # Store the original nvec for encoding/decoding
        self.nvec = env.action_space.nvec

    def action(self, action):
        # Convert a discrete action into a MultiDiscrete action
        multi_discrete_action = np.empty(len(self.nvec), dtype=np.int32)

        for i in range(len(self.nvec)):
            multi_discrete_action[i] = action % self.nvec[i]
            action = action // self.nvec[i]

        return multi_discrete_action

    def reverse_action(self, multi_discrete_action):
        # Convert a MultiDiscrete action back into a discrete action
        discrete_action = 0
        for i in reversed(range(len(self.nvec))):
            discrete_action *= self.nvec[i]
            discrete_action += multi_discrete_action[i]

        return discrete_action