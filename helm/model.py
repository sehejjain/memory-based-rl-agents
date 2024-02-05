import torch.nn as nn
from torch.distributions import Categorical
from transformers import TransfoXLModel, TransfoXLConfig, TransfoXLTokenizer
import torch
import numpy as np
import clip
import os
from clip.simple_tokenizer import SimpleTokenizer


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden, out_dim, n_hidden=0):
        super(DiscreteActor, self).__init__()
        self.modlist = [nn.Linear(input_dim, hidden),
                        nn.LayerNorm(hidden, elementwise_affine=False),
                        nn.ReLU()]
        if n_hidden > 0:
            self.modlist.extend([nn.Linear(hidden, hidden),
                                 nn.LayerNorm(hidden, elementwise_affine=False),
                                 nn.ReLU()] * n_hidden)
        self.modlist.extend([nn.Linear(hidden, out_dim),
                            nn.Softmax(dim=-1)])
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


class SmallImpalaCNN(nn.Module):
    def __init__(self, observation_shape, channel_scale=1, hidden_dim=256, n_env=16):
        super(SmallImpalaCNN, self).__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.obs_size = observation_shape
        self.in_channels = 3
        kernel1 = 8 if self.obs_size[1] > 9 else 4
        kernel2 = 4 if self.obs_size[2] > 9 else 2
        stride1 = 4 if self.obs_size[1] > 9 else 2
        stride2 = 2 if self.obs_size[2] > 9 else 1
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=n_env*channel_scale, kernel_size=kernel1, stride=stride1),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=16*channel_scale, out_channels=32*channel_scale, kernel_size=kernel2, stride=stride2),
                                    nn.ReLU())

        in_features = self._get_feature_size(self.obs_size)
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_dim)

        self.hidden_dim = hidden_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        # print("impala input", x.shape)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 3, 1, 2)
            # print("permuted", x.shape)
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        # print("impala output", x.shape)
        return x

    def _get_feature_size(self, shape):
        if shape[0] != 3:
            dummy_input = torch.zeros((shape[-1], *shape[:-1])).unsqueeze(0)
            print(dummy_input.shape)
        else:
            dummy_input = torch.zeros((shape[0], *shape[1:])).unsqueeze(0)
        x = self.block2(self.block1(dummy_input))
        return np.prod(x.shape[1:])


class FrozenHopfield(nn.Module):
    def __init__(self, hidden_dim, input_dim, embeddings, beta):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        super(FrozenHopfield, self).__init__()
        self.rand_obs_proj = torch.nn.Parameter(torch.normal(mean=0.0, std=1 / np.sqrt(hidden_dim), size=(hidden_dim, input_dim)), requires_grad=False)
        self.word_embs = embeddings
        self.beta = beta

    def forward(self, observations):
        # print(observations.shape)
        observations = self._preprocess_obs(observations)
        observations = observations @ self.rand_obs_proj.T
        similarities = observations @ self.word_embs.T / (
                    observations.norm(dim=-1).unsqueeze(1) @ self.word_embs.norm(dim=-1).unsqueeze(0) + 1e-8)
        softm = torch.softmax(self.beta * similarities, dim=-1)
        state = softm @ self.word_embs
        return state

    def _preprocess_obs(self, obs):
        obs = obs.mean(1)
        obs = torch.stack([o.view(-1) for o in obs])
        return obs


class HELM(nn.Module):
    def __init__(self, action_space, input_dim, optimizer, learning_rate, epsilon=1e-8, mem_len=511, beta=1,
                 device='cuda'):
        super(HELM, self).__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len

        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103', config=config)
        n_tokens = self.model.word_emb.n_token
        word_embs = self.model.word_emb(torch.arange(n_tokens)).detach().to(device)
        hidden_dim = self.model.d_embed
        hopfield_input = np.prod(input_dim[1:])
        self.frozen_hopfield = FrozenHopfield(hidden_dim, hopfield_input, word_embs, beta=beta)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_encoder = SmallImpalaCNN(input_dim, channel_scale=4, hidden_dim=hidden_dim)
        self.out_dim = hidden_dim*2
        self.actor = DiscreteActor(self.out_dim, 128, action_space.n).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.out_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.yield_trainable_params(), lr=learning_rate,
                                                             eps=epsilon)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        self.memory = None

    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if 'model.' in n:
                continue
            else:
                yield p

    def forward(self, observations):
        bs, *_ = observations.shape
        obs_query = self.query_encoder(observations)
        vocab_encoding = self.frozen_hopfield.forward(observations)
        # print("vocab", vocab_encoding.shape,"vocab squeezed", vocab_encoding.unsqueeze(1).shape,  "memory", self.memory[0].shape)
        out = self.model(inputs_embeds=vocab_encoding.unsqueeze(1), output_hidden_states=True, mems=self.memory)
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        hiddens = out.last_hidden_state[:, -1, :].cpu().numpy()

        hidden = torch.cat([hidden, obs_query], dim=-1)

        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()

        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), hiddens

    def evaluate_actions(self, hidden_states, actions, observations):
        queries = self.query_encoder(observations)
        hidden = torch.cat([hidden_states, queries], dim=-1)

        log_prob, entropy = self.actor.evaluate(hidden, actions)
        value = self.critic(hidden).squeeze()

        return value, log_prob, entropy



def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def xavier_uniform_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.)
    return module
