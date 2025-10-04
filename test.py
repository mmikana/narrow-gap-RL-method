import torch
import os
import torch.nn as nn
from Env.Env import Quad2NGEnv
import numpy as np

# Set device for operation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Type:", device)

# Initialize env
env = Quad2NGEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Load para
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + "/ddpg_actor_2025_04_20_11_00_30.pth"

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2

        return x


actor = Actor(action_dim, action_dim, hidden_dim=256).to (device)
actor.load_state_dict(torch.load(actor_path))

# 初始化跟踪变量
reward_buffer = []
episode_lengths = []
collision_rates = []
success_rates = []
reward_best = -np.inf

# Test phase
episode_num = 30
step_num = 200
for episode_i in range(episode_num):
    reward_episode = 0
    state, info = env.reset()
    episode_collision = False
    episode_success = False
    steps_per_episodes = 0

    for step_i in range(step_num):
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state, reward, done, terminated, info = env.step(action)
        state = next_state
        reward_episode += reward

        # 记录碰撞和成功
        if info.get("collision", False):
            episode_collision = True
        if info.get("goal_achieved", False):
            episode_success = True

        if terminated:
            break
        steps_per_episodes += 1
        # 记录本回合数据
        reward_buffer.append(reward_episode)
        episode_lengths.append(steps_per_episodes)
        collision_rates.append(1 if episode_collision else 0)
        success_rates.append(1 if episode_success else 0)

    # 计算滑动平均奖励
    window = min(len(reward_buffer), 10)
    reward_avg = np.mean(reward_buffer[-window:])

    # TODO evaluation

env.close()

