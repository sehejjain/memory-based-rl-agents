"""
The code defines a custom policy class for an actor-critic algorithm
using a GRU-based neural network as the feature extractor.
"""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GRUNetwork(BaseFeaturesExtractor):
    """The GRUNetwork class is a subclass of BaseFeaturesExtractor and represents a GRU-based neural network."""

    def __init__(self, observation_space, features_dim=512, *args, **kwargs):
        super(GRUNetwork, self).__init__(
            observation_space=observation_space, features_dim=features_dim
        )
        print(observation_space)
        self.visual_head = nn.Sequential(
            nn.Conv2d(observation_space["image"].shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.vector_head = nn.Sequential(
            nn.Linear(observation_space["target_color"].shape[0], 128),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(input_size=1024, hidden_size=512)
        self.dense = nn.Linear(512, 512)

    def forward(self, observations):
        """
        The forward function takes in observations, processes them through visual and vector heads,
        concatenates the features, passes them through a GRU layer, and returns the output.

        :param observations: The "observations" parameter is a dictionary that contains two keys:
        "visual_obs" and "vector_obs"
        :return: The output of the forward function is the hidden state of the GRU layer, denoted as
        'h'.
        """

        visual_obs = observations["image"].permute(0, 3, 1, 2)
        vector_obs = observations["target_color"]
        visual_features = self.visual_head(visual_obs)
        vector_features = self.vector_head(vector_obs)
        h = torch.cat([visual_features, vector_features], dim=1)
        batch_size = h.size(0)
        h = h.view(batch_size, -1, self.features_dim)  # Reshape to sequences
        h = self.gru(h)
        h = h.view(batch_size, -1)  # Reshape back to original batch shape

        h = self.dense(h)
        return h


class CustomGRUPolicy(ActorCriticPolicy):
    """The CustomPolicy class is a subclass of ActorCriticPolicy that uses a GRUNetwork as the features extractor and has a value and dense2 layer for policy and value estimation."""

    def __init__(self, lr_schedule, *args, **kwargs):
        """
        The `__init__` function initializes the `CustomPolicy` class and sets up the neural network
        architecture.
        """

        super(CustomGRUPolicy, self).__init__(
            *args,
            **kwargs,
            lr_schedule=lr_schedule,
            features_extractor_class=GRUNetwork,
            features_extractor_kwargs=dict(features_dim=1024)
        )
        self.base_network = GRUNetwork(self.observation_space)

        self.value = nn.Linear(512, 1)
        self.dense2 = nn.Linear(512, 3)
        self.activation = nn.ReLU()

    def forward(self, obs, *args, **kwargs):
        """
        The forward function takes an observation as input, extracts features from it, and returns the
        action distribution and value estimates.

        :param obs: The "obs" parameter represents the observation or input data that is passed to the
        forward method. It is used to extract features from the input data and generate the policy and
        value outputs
        :return: The forward method is returning the action distribution and value.
        """
        features = self.extract_features(obs)
        value = self.activation(self.dense1(features))
        policy = self.activation(self.dense2(features))
        return self._get_action_dist_from_latent(policy), value
