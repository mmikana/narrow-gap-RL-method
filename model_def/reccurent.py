import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal


from ReplayMemory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')


class CriticNetwork(nn.Module):
    def __init__(self, lr_critic, state_dim, action_dim, gru_hidden_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gru_hidden_dim = gru_hidden_dim  # GRU隐藏层维度
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        # 输入层：将状态和动作拼接后映射到适合GRU的维度
        self.input_proj = nn.Linear(self.state_dim + self.action_dim, gru_hidden_dim)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=gru_hidden_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True  # 批处理维度在前
        )
        
        # GRU输出后的全连接层
        self.fc1 = nn.Linear(gru_hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.to(device)

    def forward(self, state, action, hidden=None):
        """
        参数:
            state: 状态张量，形状为 (batch_size, seq_len, state_dim) 或 (batch_size, state_dim)
            action: 动作张量，形状为 (batch_size, seq_len, action_dim) 或 (batch_size, action_dim)
            hidden: GRU的初始隐藏状态，形状为 (1, batch_size, gru_hidden_dim)，可选
        """
        # 如果输入不是序列形式（没有时间维度），添加时间维度
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # 变为 (batch_size, 1, state_dim)
            action = action.unsqueeze(1)  # 变为 (batch_size, 1, action_dim)
        
        # 拼接状态和动作
        x = torch.cat([state, action], dim=2)  # 形状: (batch_size, seq_len, state_dim+action_dim)
        
        # 投影到GRU输入维度
        x = self.input_proj(x)  # 形状: (batch_size, seq_len, gru_hidden_dim)
        
        # 通过GRU层
        if hidden is None:
            # 如果没有提供初始隐藏状态，创建一个零初始化的隐藏状态
            batch_size = x.size(0)
            hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
        
        gru_out, hidden = self.gru(x, hidden)  # gru_out形状: (batch_size, seq_len, gru_hidden_dim)
        
        # 取GRU的最后一个时间步的输出
        last_out = gru_out[:, -1, :]  # 形状: (batch_size, gru_hidden_dim)
        
        # 通过全连接层计算Q值
        x = F.relu(self.fc1(last_out))
        x = F.relu(self.fc2(x))
        q = self.q(x)  # 形状: (batch_size, 1)

        return q, hidden

    def init_hidden(self, batch_size):
        """初始化GRU的隐藏状态"""
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)



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



