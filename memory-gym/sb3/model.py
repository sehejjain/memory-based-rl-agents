"""
The code defines a custom policy class for an actor-critic algorithm
using a GRU-based neural network as the feature extractor.
"""
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.distributions import Categorical
from transformers import TransfoXLConfig, TransfoXLModel


def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def xavier_uniform_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.0)
    return module


class GRUNetwork(BaseFeaturesExtractor):
    """The GRUNetwork class is a subclass of BaseFeaturesExtractor and represents a GRU-based neural network."""

    def __init__(self, observation_space, features_dim=512, *args, **kwargs):
        super(GRUNetwork, self).__init__(
            observation_space=observation_space, features_dim=features_dim
        )
        self.visual_head = nn.Sequential(
            nn.Conv2d(observation_space.shape[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros((1,) + observation_space.shape)
            sample_input = sample_input.permute(0, 3, 1, 2)  # NCHW
            gru_input_size = self.visual_head(sample_input).shape[1]

        self.gru = nn.GRUCell(input_size=gru_input_size, hidden_size=512)
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
        visual_obs = observations.permute(0, 3, 1, 2)  # NCHW
        h = self.visual_head(visual_obs)
        batch_size = h.size(0)
        actual_features_dim = h.size(1)

        # Now reshape h to [batch_size, -1, actual_features_dim]
        # h = h.view(batch_size, -1, actual_features_dim)
        h = self.gru(h)
        h = h.view(batch_size, -1)  # Reshape back to original batch shape
        # print("before dense", h.shape)
        h = self.dense(h)
        # print("after dense", h.shape)
        return h


class CustomGRUPolicy(ActorCriticPolicy):
    """The CustomPolicy class is a subclass of ActorCriticPolicy that uses a GRUNetwork as the features extractor and has a value and dense2 layer for policy and value estimation."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        """
        The `__init__` function initializes the `CustomPolicy` class and sets up the neural network
        architecture.
        """

        super(CustomGRUPolicy, self).__init__(
            action_space=action_space,
            observation_space=observation_space,
            lr_schedule=lr_schedule,
            features_extractor_class=GRUNetwork,
            features_extractor_kwargs=dict(features_dim=512),
            *args,
            **kwargs,
        )
        self.base_network = GRUNetwork(self.observation_space)

        self.action_space = action_space
        nvec = self.action_space.nvec
        # for multi discrete
        self.policy_out = nn.ModuleList([nn.Linear(512, n) for n in nvec])

        self.value = nn.Linear(512, 1)
        self.dense2 = nn.Linear(512, 64)
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

        value = self.activation(self.value(features))
        policy = self.activation(self.dense2(features))
        # Get action distribution from policy output
        action_dist = self._get_action_dist_from_latent(policy)

        # Sample actions and compute log probabilities
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)

        return actions, value, log_probs


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden, out_dim, n_hidden=0):
        super(DiscreteActor, self).__init__()
        self.modlist = [
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.ReLU(),
        ]
        if n_hidden > 0:
            self.modlist.extend(
                [
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden, elementwise_affine=False),
                    nn.ReLU(),
                ]
                * n_hidden
            )
        self.modlist.extend([nn.Linear(hidden, out_dim), nn.Softmax(dim=-1)])
        self.actor = nn.Sequential(*self.modlist).apply(orthogonal_init)

    def forward(self, states, deterministic=False):
        probs = self.actor(states)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs).squeeze()
        else:
            action = dist.sample().squeeze()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions.squeeze())
        entropy = dist.entropy()
        return log_prob, entropy


class MultiDiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden, action_dims, n_hidden=0):
        super(MultiDiscreteActor, self).__init__()
        self.action_dims = action_dims
        self.modlist = [
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.ReLU(),
        ]
        if n_hidden > 0:
            self.modlist.extend(
                [
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden, elementwise_affine=False),
                    nn.ReLU(),
                ]
                * n_hidden
            )
        self.actor_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
                for action_dim in self.action_dims
            ]
        )
        self.actor = nn.Sequential(*self.modlist).apply(orthogonal_init)

    def forward(self, states, deterministic=False):
        x = self.actor(states)
        actions, log_probs = [], []
        for actor_head in self.actor_heads:
            probs = actor_head(x)
            dist = Categorical(probs)

            if deterministic:
                action = torch.argmax(probs).squeeze()
            else:
                action = dist.sample().squeeze()

            log_prob = dist.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)

        return torch.stack(actions), torch.stack(log_probs)

    # def evaluate(self, states, actions):
    #     x = self.actor(states)
    #     log_probs, entropies = [], []
    #     for actor_head, action in zip(self.actor_heads, actions.unbind()):
    #         probs = actor_head(x)
    #         dist = Categorical(probs)
    #         log_prob = dist.log_prob(action.squeeze())
    #         entropy = dist.entropy()
    #         log_probs.append(log_prob)
    #         entropies.append(entropy)

    #     return torch.stack(log_probs), torch.stack(entropies)

    def evaluate(self, hidden, actions):
        x = self.actor(hidden)
        log_probs, entropies = [], []

        for i, actor_head in enumerate(self.actor_heads):
            probs = actor_head(x)
            dist = Categorical(probs)

            # Adjust the action shape to match the batch shape
            action = actions[:, i].unsqueeze(-1)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            log_probs.append(log_prob)
            entropies.append(entropy)

        return torch.stack(log_probs), torch.stack(entropies)


class SmallImpalaCNN(nn.Module):
    def __init__(self, observation_shape, channel_scale=1, hidden_dim=256):
        super(SmallImpalaCNN, self).__init__()
        self.obs_size = observation_shape
        self.in_channels = 3
        kernel1 = 8 if self.obs_size[1] > 9 else 4
        kernel2 = 4 if self.obs_size[2] > 9 else 2
        stride1 = 4 if self.obs_size[1] > 9 else 2
        stride2 = 2 if self.obs_size[2] > 9 else 1
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=16 * channel_scale,
                kernel_size=kernel1,
                stride=stride1,
            ),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16 * channel_scale,
                out_channels=32 * channel_scale,
                kernel_size=kernel2,
                stride=stride2,
            ),
            nn.ReLU(),
        )

        in_features = self._get_feature_size(self.obs_size)
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_dim)

        self.hidden_dim = hidden_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 3, 1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

    def _get_feature_size(self, shape):
        if shape[0] != 3:
            dummy_input = torch.zeros((shape[-1], *shape[:-1])).unsqueeze(0)
        else:
            dummy_input = torch.zeros((shape[0], *shape[1:])).unsqueeze(0)
        x = self.block2(self.block1(dummy_input))
        return np.prod(x.shape[1:])


class FrozenHopfield(nn.Module):
    def __init__(self, hidden_dim, input_dim, embeddings, beta):
        super(FrozenHopfield, self).__init__()
        self.rand_obs_proj = torch.nn.Parameter(
            torch.normal(
                mean=0.0, std=1 / np.sqrt(hidden_dim), size=(hidden_dim, input_dim)
            ),
            requires_grad=False,
        )
        self.word_embs = embeddings
        self.beta = beta

    def forward(self, observations):
        observations = self._preprocess_obs(observations)
        observations = observations @ self.rand_obs_proj.T
        similarities = (
            observations
            @ self.word_embs.T
            / (
                observations.norm(dim=-1).unsqueeze(1)
                @ self.word_embs.norm(dim=-1).unsqueeze(0)
                + 1e-8
            )
        )
        softm = torch.softmax(self.beta * similarities, dim=-1)
        state = softm @ self.word_embs
        return state

    def _preprocess_obs(self, obs):
        obs = obs.mean(1)
        obs = torch.stack([o.view(-1) for o in obs])
        return obs


class HELM(nn.Module):
    def __init__(
        self,
        action_space,
        input_dim,
        optimizer,
        learning_rate,
        epsilon=1e-8,
        mem_len=511,
        beta=1,
        device="cuda",
    ):
        super(HELM, self).__init__()
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len

        self.model = TransfoXLModel.from_pretrained("transfo-xl-wt103", config=config)
        n_tokens = self.model.word_emb.n_token
        word_embs = self.model.word_emb(torch.arange(n_tokens)).detach().to(device)
        hidden_dim = self.model.d_embed
        hopfield_input = np.prod(input_dim[1:])
        self.frozen_hopfield = FrozenHopfield(
            hidden_dim, hopfield_input, word_embs, beta=beta
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_encoder = SmallImpalaCNN(
            input_dim, channel_scale=4, hidden_dim=hidden_dim
        )
        self.out_dim = hidden_dim * 2
        # n_hidden = (
        #     action_space.nvec
        #     if action_space.__class__.__name__ == "MultiDiscrete"
        #     else action_space.n
        # )
        # print(n_hidden)

        if action_space.__class__.__name__ == "MultiDiscrete":
            self.actor = MultiDiscreteActor(self.out_dim, 128, action_space.nvec).apply(
                orthogonal_init
            )

        else:
            self.actor = DiscreteActor(self.out_dim, 128, action_space.n).apply(
                orthogonal_init
            )
        self.critic = nn.Sequential(
            nn.Linear(self.out_dim, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(512, 1),
        ).apply(orthogonal_init)
        try:
            self.optimizer = getattr(torch.optim, optimizer)(
                self.yield_trainable_params(), lr=learning_rate, eps=epsilon
            )
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        self.memory = None

    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if "model." in n:
                continue
            else:
                yield p

    def forward(self, observations):
        bs, *_ = observations.shape
        obs_query = self.query_encoder(observations)
        vocab_encoding = self.frozen_hopfield.forward(observations)
        out = self.model(
            inputs_embeds=vocab_encoding.unsqueeze(1),
            output_hidden_states=True,
            mems=self.memory,
        )  # type: ignore
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        hiddens = out.last_hidden_state[:, -1, :].cpu().numpy()

        hidden = torch.cat([hidden, obs_query], dim=-1)

        actions, log_probs = self.actor(hidden)
        # print("Unmodified", actions)
        actions = actions.view(1, -1)
        # print("Modified", actions)
        values = self.critic(hidden).squeeze()

        return (
            actions.cpu().numpy(),
            values.cpu().numpy(),
            log_probs.cpu().numpy().squeeze(),
            hiddens,
        )

    def evaluate_actions(self, hidden_states, actions, observations):
        queries = self.query_encoder(observations)
        hidden = torch.cat([hidden_states, queries], dim=-1)

        log_prob, entropy = self.actor.evaluate(hidden, actions)
        value = self.critic(hidden).squeeze()

        # If actions are a list (for MultiDiscrete), the log_prob and entropy will be tensors of shape [num_actions, batch_size]
        if isinstance(actions, list):
            log_prob = log_prob.sum(0)  # Sum log probabilities over actions
            entropy = entropy.mean(0)  # Mean entropy over actions

        return value, log_prob, entropy
