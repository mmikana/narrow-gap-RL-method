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


class SACAgent:
    def __init__(self, state_dim, action_dim, memo_capacity, lr_actor, lr_critic, gamma, tau,
                 layer1_dim, layer2_dim, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # 探索参数
        self.exploration_noise = 0.1
        self.exploration_decay = 0.9995
        self.min_exploration = 0.01

        self.Replay_Buffer = ReplayMemory(memo_capacity=memo_capacity, state_dim=state_dim, action_dim=action_dim)

        # 网络初始化
        self.critic_1 = CriticNetwork(lr_critic=lr_critic, state_dim=state_dim, action_dim=action_dim,
                                      fc1_dim=layer1_dim, fc2_dim=layer2_dim)
        self.critic_2 = CriticNetwork(lr_critic=lr_critic, state_dim=state_dim, action_dim=action_dim,
                                      fc1_dim=layer1_dim, fc2_dim=layer2_dim)
        self.value = ValueNetwork(lr_critic=lr_critic, state_dim=state_dim,
                                  fc1_dim=layer1_dim, fc2_dim=layer2_dim)
        self.target_value = ValueNetwork(lr_critic=lr_critic, state_dim=state_dim,
                                         fc1_dim=layer1_dim, fc2_dim=layer2_dim)
        self.actor = ActorNetwork(lr_actor=lr_actor, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=layer1_dim, fc2_dim=layer2_dim, max_action=1.0)  # 动作范围[-1,1]

        # 初始化目标网络权重
        self.update_target_network(1.0)

    def update_target_network(self, tau):
        """软更新目标网络"""
        with torch.no_grad():
            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_action(self, state, add_noise=False):
        state = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)

        with torch.no_grad():
            mu, sigma = self.actor(state)
            action = torch.tanh(mu)  # 使用确定性策略进行推理

            if add_noise:
                # 添加探索噪声
                noise = torch.normal(0, self.exploration_noise, size=action.shape).to(device)
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)

        action = action.squeeze(0).cpu().numpy()

        # 衰减探索噪声
        if add_noise:
            self.exploration_noise = max(self.min_exploration,
                                         self.exploration_noise * self.exploration_decay)

        return action

    def update(self):
        if self.Replay_Buffer.memo_counter < self.batch_size:
            return

        # 从回放缓冲区采样
        state, action, reward, next_state, done = self.Replay_Buffer.sample_memory(batch_size=self.batch_size)

        # 转换为tensor
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device)

        # 更新价值网络
        with torch.no_grad():
            next_value = self.target_value(next_state).squeeze()
            target_q = reward + self.gamma * next_value * (1 - done)

        # 更新critic网络
        current_q1 = self.critic_1(state, action).squeeze()
        current_q2 = self.critic_2(state, action).squeeze()

        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_2.optimizer.step()

        # 更新actor网络
        new_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        min_q = torch.min(
            self.critic_1(state, new_actions).squeeze(),
            self.critic_2(state, new_actions).squeeze()
        )

        actor_loss = (log_probs.squeeze() - min_q).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor.optimizer.step()

        # 更新value网络
        current_value = self.value(state).squeeze()
        with torch.no_grad():
            min_q_new = torch.min(
                self.critic_1(state, new_actions.detach()).squeeze(),
                self.critic_2(state, new_actions.detach()).squeeze()
            )
            target_value = min_q_new - log_probs.squeeze().detach()

        value_loss = F.mse_loss(current_value, target_value)

        self.value.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value.optimizer.step()

        # 软更新目标网络
        self.update_target_network(self.tau)

    def save_models(self, path):
        """保存所有模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic_1.state_dict(),
            'critic2': self.critic_2.state_dict(),
            'value': self.value.state_dict(),
            'target_value': self.target_value.state_dict()
        }, path)

    def load_models(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic1'])
        self.critic_2.load_state_dict(checkpoint['critic2'])
        self.value.load_state_dict(checkpoint['value'])
        self.target_value.load_state_dict(checkpoint['target_value'])