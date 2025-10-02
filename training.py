import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from Env.Env import QuadFlyEnv
from uhtk.mcv_log_manager import LogManager
from uhtk.VISUALIZE.mcom import mcom
from SAC_agent import SACAgent

torch.set_num_threads(1)

episode_num = 200000
step_num = 200
plot_reward = False
env = QuadFlyEnv(goal_position=np.array([3.0, 3.0, 3.0]))

# 获取状态和动作维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = SACAgent(state_dim=state_dim, action_dim=action_dim, memo_capacity=10000, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, tau=0.05, layer1_dim=256, layer2_dim=256, batch_size=256)

# 初始化跟踪变量
reward_buffer = []
episode_lengths = []
collision_rates = []
success_rates = []
reward_best = -np.inf

# 训练进度跟踪
print("Starting training...")
print(f"{'Episode':<8} {'Reward':<10} {'Avg Reward':<12} {'Length':<8} {'Success':<8} {'Collision':<10}")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# mcv = mcom(draw_mode='Img',path='./episode_plots', rapid_flush=True)
# mcv_manager = LogManager(mcv=mcv, who='training.py')

# 训练循环
for episode_i in range(episode_num):
    reward_episode = 0
    state, info = env.reset()
    episode_collision = False
    episode_success = False
    steps_per_episodes = 0

    for step_i in range(step_num):
        # 获取动作
        action = agent.get_action(state, add_noise=True)
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)

        # 存储经验
        agent.Replay_Buffer.add_memory(state, action, reward, next_state, terminated)

        # 更新状态和奖励
        reward_episode += reward
        state = next_state

        # 更新agent
        if step_i % 4 == 0:
            agent.update()

        # 记录碰撞和成功
        if info.get("collision", False):
            episode_collision = True
        if info.get("goal_achieved", False):
            episode_success = True

        if terminated or truncated:
            break
        steps_per_episodes += 1

    # 记录本回合数据
    reward_buffer.append(reward_episode)
    episode_lengths.append(steps_per_episodes)
    collision_rates.append(1 if episode_collision else 0)
    success_rates.append(1 if episode_success else 0)

    # mcv_manager.log_trivial({
    #     "reward_episode": reward_episode,
    #     "steps_per_episodes": steps_per_episodes,
    # })
    # mcv_manager.log_trivial_finalize(print=True)

    # 计算滑动平均奖励
    window = min(len(reward_buffer), 10)
    reward_avg = np.mean(reward_buffer[-window:])

    # 打印训练进度
    print(
        f'{episode_i:<8} {reward_episode:<10.2f} {reward_avg:<12.2f} {steps_per_episodes:<8} {episode_success:<8}'
        f' {episode_collision:<10}')

    # 计算平滑奖励（与后续绘图逻辑保持一致）
    smoothed_reward = gaussian_filter(reward_buffer, sigma=5)

env.close()

# 绘制训练曲线
if plot_reward:
    plt.figure(figsize=(15, 10))

    # 奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(reward_buffer, color='purple', alpha=0.3, label='Episode Reward')
    smoothed_reward = gaussian_filter(reward_buffer, sigma=5)
    plt.plot(smoothed_reward, color='purple', linewidth=2, label='Smoothed Reward')
    plt.title('Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # 回合长度曲线
    plt.subplot(2, 2, 2)
    plt.plot(episode_lengths, color='blue', alpha=0.3, label='Episode Length')
    smoothed_length = gaussian_filter(episode_lengths, sigma=5)
    plt.plot(smoothed_length, color='blue', linewidth=2, label='Smoothed Length')
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)

    # 成功率曲线
    plt.subplot(2, 2, 3)
    window_size = 20
    success_rates_smoothed = []
    for i in range(len(success_rates)):
        start = max(0, i - window_size + 1)
        success_rates_smoothed.append(np.mean(success_rates[start:i + 1]) * 100)
    plt.plot(success_rates_smoothed, color='green', linewidth=2)
    plt.title(f'Success Rate (Last {window_size} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    plt.grid(True)

print("Training completed!")
print(f"Best average reward: {reward_best:.2f}")
print(f"Final success rate: {np.mean(success_rates[-20:]) * 100:.1f}%")
