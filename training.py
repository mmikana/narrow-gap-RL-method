import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from episode_visualizer import EpisodeVisualizer
from ENV import QuadrotorEnv
from SAC_agent import SACAgent


# The directory to save model
current_path = os.path.dirname(os.path.realpath(__file__))
model_dir = current_path + '/models/'
os.makedirs(model_dir, exist_ok=True)  # 确保模型目录存在
timestamp = time.strftime('%Y%m%d-%H%M%S')

# Create environment
env = QuadrotorEnv()
STATE_DIM = env.observation_space.shape[0]  # 应该是12
ACTION_DIM = env.action_space.shape[0]  # 应该是4


print(f"State dimension: {STATE_DIM}")
print(f"Action dimension: {ACTION_DIM}")

# Set parameters
EPISODE_NUM = 10
STEP_NUM = 1000
VISUALIZE_INTERVAL = 10
SAVE_EPISODE_PLOTS = True

# Choose Agent111
agent = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, memo_capacity=10000, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, tau=0.005, layer1_dim=256, layer2_dim=256, batch_size=256)

# Initialization
reward_buffer = []
episode_lengths = []
collision_rates = []
success_rates = []
reward_best = -np.inf  # 初始最佳奖励
PLOT_REWARD = True

# Training progress tracking
print("Starting training...")
print(f"{'Episode':<8} {'Reward':<10} {'Avg Reward':<12} {'Length':<8} {'Success':<8} {'Collision':<10}")

# training
for episode_i in range(EPISODE_NUM):
    reward_episode = 0
    state, info = env.reset()
    episode_collision = False
    episode_success = False
    steps_per_episodes = 0

    for step_i in range(STEP_NUM):
        # 获取动作（添加探索噪声）
        action = agent.get_action(state, add_noise=True)

        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)

        # 存储经验
        agent.Replay_Buffer.add_memory(state, action, reward, next_state, terminated)

        # 更新状态和奖励
        reward_episode += reward
        state = next_state

        # 更新agent（定期更新）
        if step_i % 4 == 0:  # 每4步更新一次，提高数据效率
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

    # 计算滑动平均奖励
    if len(reward_buffer) > 10:
        reward_avg = np.mean(reward_buffer[-10:])  # 最近10个回合的平均奖励
    else:
        reward_avg = np.mean(reward_buffer)

    # 计算成功率（最近50个回合）
    if len(success_rates) > 50:
        recent_success_rate = np.mean(success_rates[-50:]) * 100
    else:
        recent_success_rate = np.mean(success_rates) * 100 if success_rates else 0

    # 保存最佳模型
    if reward_avg > reward_best:
        reward_best = reward_avg
        torch.save(agent.actor.state_dict(), model_dir + f'quadrotor_actor_best_{timestamp}.pth')
        torch.save(agent.critic_1.state_dict(), model_dir + f'quadrotor_critic1_best_{timestamp}.pth')
        torch.save(agent.critic_2.state_dict(), model_dir + f'quadrotor_critic2_best_{timestamp}.pth')
        print(f"✓ Saving model with best avg reward: {reward_best:.2f}")

    # 定期可视化episode
    if (episode_i + 1) % VISUALIZE_INTERVAL == 0 and episode_i > 0:
        print(f"📊 可视化第 {episode_i} 回合的飞行数据...")

        # 获取当前episode数据
        episode_data = env.get_episode_data()

        # 创建可视化器
        visualizer = EpisodeVisualizer()

        # 生成详细分析图
        visualizer.visualize_episode(
            trajectory=episode_data['trajectory'],
            orientations=episode_data['orientations'],
            velocities=episode_data['velocities'],
            narrow_gap=env.NarrowGap,
            goal_position=env.goal_position,
            step_interval=max(1, len(episode_data['trajectory']) // 20)  # 动态调整箭头密度
        )

        # 保存分析图
        if SAVE_EPISODE_PLOTS:
            plot_dir = current_path + '/episode_plots/'
            os.makedirs(plot_dir, exist_ok=True)
            visualizer.save_plot(f"{plot_dir}episode_{episode_i}_{timestamp}.png")

        plt.show()
        plt.close('all')  # 关闭所有图表避免内存泄漏

    # 打印训练进度
    print(
        f'{episode_i:<8} {reward_episode:<10.2f} {reward_avg:<12.2f} {steps_per_episodes:<8} {episode_success:<8}'
        f' {episode_collision:<10}')

    # 如果连续成功，增加难度（课程学习）
    if recent_success_rate > 70 and hasattr(env, 'increase_difficulty'):
        env.increase_difficulty()
        print(f"🎯 Increasing difficulty! Current level: {env.current_difficulty}")

env.close()


# 保存最终模型
torch.save(agent.actor.state_dict(), model_dir + f'quadrotor_actor_final_{timestamp}.pth')
torch.save(agent.critic_1.state_dict(), model_dir + f'quadrotor_critic1_final_{timestamp}.pth')
torch.save(agent.critic_2.state_dict(), model_dir + f'quadrotor_critic2_final_{timestamp}.pth')

# 绘制训练曲线
if PLOT_REWARD:
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

    # 成功率曲线（滑动窗口）
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

    # 碰撞率曲线（滑动窗口）
    plt.subplot(2, 2, 4)
    collision_rates_smoothed = []
    for i in range(len(collision_rates)):
        start = max(0, i - window_size + 1)
        collision_rates_smoothed.append(np.mean(collision_rates[start:i + 1]) * 100)

    plt.plot(collision_rates_smoothed, color='red', linewidth=2)
    plt.title(f'Collision Rate (Last {window_size} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Collision Rate (%)')
    plt.ylim(0, 100)
    plt.grid(True)

    plt.tight_layout()
    training_plots = current_path +'/training_plots/'
    plt.savefig(f"{training_plots}Training_Results_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

print("Training completed!")
print(f"Best average reward: {reward_best:.2f}")
print(f"Final success rate: {np.mean(success_rates[-20:]) * 100:.1f}%")
