import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
import yaml
from scipy.ndimage import gaussian_filter
from importlib import import_module  # 用于动态导入类


torch.set_num_threads(1)
# 加载配置文件
def load_config(config_path):
    # 读取文件时指定 encoding='utf-8'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 自动生成时间戳
    config['timestamp'] = time.strftime('%Y%m%d-%H%M%S')
    # 创建目录
    for dir_key in ['model_dir', 'episode_plot_dir', 'training_plot_dir']:
        os.makedirs(config['paths'][dir_key], exist_ok=True)
    return config


# 动态导入类（支持不同算法、环境、可视化器）
def dynamic_import(class_path):
    module_name, class_name = class_path.rsplit('.', 1)
    module = import_module(module_name)
    return getattr(module, class_name)


# 主训练函数
def main(config_path='config.yaml'):
    config = load_config(config_path)

    # 动态创建环境
    env_class = dynamic_import(f"Env.{config['env']['type']}")  # 环境在Env模块中
    env = env_class(**config['env']['parameters'])

    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    config['agent']['parameters']['state_dim'] = state_dim
    config['agent']['parameters']['action_dim'] = action_dim

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # 动态创建智能体
    agent_class = dynamic_import(f"SAC_agent.{config['agent']['type']}")  # 假设智能体在SAC_agent模块中
    agent = agent_class(**config['agent']['parameters'])

    # 初始化跟踪变量
    reward_buffer = []
    episode_lengths = []
    collision_rates = []
    success_rates = []
    reward_best = -np.inf

    # 训练进度跟踪
    print("Starting training...")
    print(f"{'Episode':<8} {'Reward':<10} {'Avg Reward':<12} {'Length':<8} {'Success':<8} {'Collision':<10}")

    # 训练循环
    for episode_i in range(config['training']['episode_num']):
        reward_episode = 0
        state, info = env.reset()
        episode_collision = False
        episode_success = False
        steps_per_episodes = 0

        for step_i in range(config['training']['step_num']):
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

        # 计算滑动平均奖励
        window = min(len(reward_buffer), 10)
        reward_avg = np.mean(reward_buffer[-window:])

        # 计算成功率
        success_window = min(len(success_rates), 50)
        recent_success_rate = np.mean(success_rates[-success_window:]) * 100 if success_rates else 0

        # 保存最佳模型
        if config['training']['save_best_model'] and reward_avg > reward_best:
            reward_best = reward_avg
            model_dir = config['paths']['model_dir']
            torch.save(agent.actor.state_dict(), f"{model_dir}quadrotor_actor_best_{config['timestamp']}.pth")
            torch.save(agent.critic_1.state_dict(), f"{model_dir}quadrotor_critic1_best_{config['timestamp']}.pth")
            torch.save(agent.critic_2.state_dict(), f"{model_dir}quadrotor_critic2_best_{config['timestamp']}.pth")
            print(f"✓ Saving model with best avg reward: {reward_best:.2f}")

        # 定期可视化
        if (episode_i + 1) % config['training']['visualize_interval'] == 0 and episode_i > 0:
            print(f"📊 可视化第 {episode_i} 回合的飞行数据...")
            episode_data = env.get_episode_data()

            # 动态创建可视化器
            visualizer_class = dynamic_import(f"episode_visualizer.{config['visualizer']['type']}")
            visualizer = visualizer_class(**config['visualizer']['parameters'])

            # 可视化参数
            step_interval = max(1, len(episode_data['trajectory']) // 20)
            visualizer.visualize_episode(
                trajectory=episode_data['trajectory'],
                orientations=episode_data['orientations'],
                velocities=episode_data['velocities'],
                narrow_gap=env.NarrowGap,
                goal_position=env.goal_position,
                step_interval=step_interval
            )

            # 保存可视化结果
            if config['training']['save_episode_plots']:
                plot_dir = config['paths']['episode_plot_dir']
                visualizer.save_plot(f"{plot_dir}episode_{episode_i}_{config['timestamp']}.png")

            plt.show()
            plt.close('all')

        # 打印训练进度
        print(
            f'{episode_i:<8} {reward_episode:<10.2f} {reward_avg:<12.2f} {steps_per_episodes:<8} {episode_success:<8}'
            f' {episode_collision:<10}')

        # 课程学习（增加难度）
        if recent_success_rate > 70 and hasattr(env, 'increase_difficulty'):
            env.increase_difficulty()
            print(f"🎯 Increasing difficulty! Current level: {env.current_difficulty}")

    env.close()

    # 保存最终模型
    model_dir = config['paths']['model_dir']
    torch.save(agent.actor.state_dict(), f"{model_dir}quadrotor_actor_final_{config['timestamp']}.pth")
    torch.save(agent.critic_1.state_dict(), f"{model_dir}quadrotor_critic1_final_{config['timestamp']}.pth")
    torch.save(agent.critic_2.state_dict(), f"{model_dir}quadrotor_critic2_final_{config['timestamp']}.pth")

    # 绘制训练曲线
    if config['training']['plot_reward']:
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
        window_size = config['training']['window_size']
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

        # 碰撞率曲线
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
        training_plots = config['paths']['training_plot_dir']
        plt.savefig(f"{training_plots}Training_Results_{config['timestamp']}.png", dpi=300, bbox_inches='tight')
        plt.show()

    print("Training completed!")
    print(f"Best average reward: {reward_best:.2f}")
    print(f"Final success rate: {np.mean(success_rates[-20:]) * 100:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
