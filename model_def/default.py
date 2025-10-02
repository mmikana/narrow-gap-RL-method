import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal


from ReplayMemory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')


class CriticNetwork(nn.Module):
    def __init__(self, lr_critic, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.to(device)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q


class ValueNetwork(nn.Module):
    def __init__(self, lr_critic, state_dim, fc1_dim, fc2_dim):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.fc1 = nn.Linear(self.state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.v = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)

        return v


class ActorNetwork(nn.Module):
    def __init__(self, lr_actor, state_dim, action_dim, fc1_dim, fc2_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.max_action = max_action

        self.fc1 = nn.Linear(self.state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.mu = nn.Linear(fc2_dim, action_dim)
        self.sigma = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.to(device)

        self.tiny_positive = 1e-6

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x)) * self.max_action  # [-max_action, max_action]
        sigma = F.softplus(self.sigma(x)) + self.tiny_positive
        sigma = torch.clamp(sigma, min=self.tiny_positive, max=1)

        return mu, sigma

    def get_action(self, state):
        mu, sigma = self.forward(state)
        probability = Normal(mu, sigma)
        action = probability.sample()
        tanh_action = torch.tanh(action)  # 使用确定性策略进行推理
        scaled_action = tanh_action * self.max_action
        return scaled_action

        # mu, sigma = self.forward(state)
        # actuion = torch.tanh(mu)
        # return actuion

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probability = Normal(mu, sigma)

        if reparameterize:
            raw_action = probability.rsample()
        else:
            raw_action = probability.sample()

        tanh_action = torch.tanh(raw_action)
        scaled_action = tanh_action * self.max_action

        # 计算log概率（包含tanh变换的修正）
        log_prob = probability.log_prob(raw_action)
        log_prob -= torch.log(1 - tanh_action.pow(2) + self.tiny_positive)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return scaled_action, log_prob
