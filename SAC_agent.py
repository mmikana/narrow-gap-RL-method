import torch
import torch.nn.functional as F

from ReplayMemory import ReplayMemory
from model_def import CriticNetwork, ValueNetwork, ActorNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

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
        state = torch.tensor(state, dtype=torch.float).to(device) # .unsqueeze(0)

        with torch.no_grad():
            action = self.actor.get_action(state)

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

        # self.Replay_Buffer.reset()

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