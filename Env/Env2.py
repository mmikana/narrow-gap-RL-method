import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json

from .QuadrotorDynamics import QuadrotorDynamics
from .NarrowGap import NarrowGap
from .collision_detector import CollisionDetector
from .i3utils import Vector3


class QuadFlyEnv(gym.Env):
    """无人机定点控制环境（无地面/边界版本）"""

    def __init__(self, goal_position=np.array([3.0, 3.0, 3.0]), visualize = True):
        # 无人机动力学初始化
        self.uav = QuadrotorDynamics()

        # 动作空间：归一化的控制指令[thrust, roll, pitch, yaw]，范围[-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # 观测空间：位置(3) + 速度(3) + 姿态(3) + 角速度(3) + 目标位置(3)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(15,),  # 包含目标位置信息
            dtype=np.float32
        )

        # 目标位置设置
        self.goal_position = np.array(goal_position, dtype=np.float64)
        self.initial_position = np.array([0, 0, 0], dtype=np.float64)  # 默认初始位置

        # 训练参数
        self.max_steps = 200  # 单轮最大步数
        self.current_step = 0
        self.current_episode = 0

        # 奖励配置（无边界相关惩罚）
        self.reward_config = {
            'goal_reward': 100.0,  # 到达目标奖励
            'distance_weight': 1.0,  # 距离权重
            'ideal_velocity': 3.0,
            'velocity_weight': 1.0,  # 速度权重
            'orientation_weight': 0,  # 姿态权重
            'angular_velocity_weight': 0,  # 角速度权重
            'goal_tolerance': 0.2  # 到达目标的位置容差(m)
        }

        # 初始化奖励计算器
        self.reward_calculator = self.RewardCalculator(self.reward_config)

        # 数据记录
        self.plot = True   # 是否画图
        self.trajectory_history = []
        self.orientation_history = []
        self.velocity_history = []
        self.reward_history = []


    class RewardCalculator:
        """奖励计算封装类，负责处理QuadFlyEnv的奖励计算逻辑"""

        def __init__(self, config):
            # 从配置中初始化奖励参数
            self.goal_reward = config.get('goal_reward', 100.0)
            self.distance_weight = config.get('distance_weight', 1.0)
            self.ideal_velocity = config.get('ideal_velocity', 3.0)
            self.velocity_weight = config.get('velocity_weight', 0.0)
            self.goal_tolerance = config.get('goal_tolerance', 0.2)

        def calculate_reward(self, uav, goal_position, goal_achieved):
            """计算单步奖励"""
            reward_step = 0.0
            # 到达目标奖励
            if goal_achieved:
                reward_step += self.goal_reward

            # 距离惩罚（基于到目标的距离）
            distance = abs(np.linalg.norm(uav.position - goal_position))
            reward_distance = self.distance_weight * np.exp(-distance)
            reward_step += reward_distance

            # 速度惩罚
            e_vel = abs(np.linalg.norm(uav.velocity) - self.ideal_velocity)
            reward_vel = self.velocity_weight * np.exp(-e_vel)
            reward_step += reward_vel

            return reward_step

    def set_goal(self, goal_position):
        """设置新的目标位置"""
        self.goal_position = np.array(goal_position, dtype=np.float64)

    def set_initial_position(self, initial_position):
        """设置初始位置"""
        self.initial_position = np.array(initial_position, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 重置无人机状态
        self.uav.reset(
            position=self.initial_position.copy(),
            orientation=[0, 0, 0]  # 初始姿态水平deg
        )
        self.goal_position = self.goal_position

        # 应用状态随机化增加训练鲁棒性
        self.apply_randomization()

        # 重置历史记录
        self.trajectory_history = []
        self.orientation_history = []
        self.velocity_history = []
        self.reward_history = []

        return self.get_obs(), {}

    def get_obs(self):
        """获取观测：无人机和NG"""
        uav_pos = np.copy(self.uav.position)
        uav_vel = np.copy(self.uav.velocity)
        uav_ori = np.copy(self.uav.orientation)
        uav_ang_vel = np.copy(self.uav.angular_velocity)

        
        ########################### normalize ##################################
        to_goal_pos = Vector3(self.goal_position - np.copy(self.uav.position))
        to_goal_pos = to_goal_pos.rev_rotate_zyx_self(uav_ori[0], uav_ori[1], uav_ori[2]).vec
        to_goal_pos /= 20.

        uav_pos /= 20.

        uav_vel /= 6  # 2* ideal_vel

        uav_ori[0] /= np.pi
        uav_ori[1] /= (0.5 * np.pi)
        uav_ori[2] /= np.pi

        uav_ang_vel[0] /= (32 * np.pi)
        uav_ang_vel[1] /= (32 * np.pi)
        uav_ang_vel[2] /= (2 * np.pi)
        ########################################################################

        normalized_obs = np.concatenate([
            to_goal_pos,
            uav_pos,
            uav_vel,
            uav_ori,
            uav_ang_vel
        ]).astype(np.float32)
        return normalized_obs

    def step(self, action):
        self.current_step += 1

        # 执行动作：将归一化动作转换为电机转速并更新动力学
        motor_speeds = self.uav.normalized_action_to_motor_speeds(action)
        self.uav.update(motor_speeds)

        terminated = (
                self.check_goal_achieved() or
                self.current_step >= self.max_steps
        )
        truncated = False

        # 计算奖励（使用封装的奖励计算器）
        reward = self.reward_calculator.calculate_reward(
            self.uav,
            self.goal_position,
            self.check_goal_achieved()
        )

        # 记录数据
        self.trajectory_history.append(self.uav.position.copy())
        self.orientation_history.append(self.uav.orientation.copy())
        self.velocity_history.append(self.uav.velocity.copy())
        self.reward_history.append(reward)

        #  保存绘制
        if terminated or truncated:
            self.current_episode += 1
            if self.plot:
                save_data_dir = os.path.join("QuadFlyEnv_data")
                data_filepath = self.save_fly_data(save_data_dir)
                self.save_fly_data()
                # from Env.episode_visualizer import EpisodeVisualizer
                # visualizer = EpisodeVisualizer()
                # save_plot_dir = os.path.join("QuadFlyEnv_plot")
                # visualizer.draw_fly_data(data_filepath=data_filepath, save_plot_dir=save_plot_dir)

        # 额外信息
        info = {
            "goal_achieved": self.check_goal_achieved(),
            "distance_to_goal": np.linalg.norm(self.uav.position - self.goal_position),
            "steps": self.current_step
        }

        return self.get_obs(), reward, terminated, truncated, info

    def check_goal_achieved(self):
        """检查是否到达目标位置"""
        distance = np.linalg.norm(self.uav.position - self.goal_position)
        return distance < self.reward_config['goal_tolerance']

    #  TODO 数量级需要修改
    def apply_randomization(self):
        """应用状态随机化，增加训练泛化性"""
        # 位置微小扰动
        self.uav.position += np.random.normal(0, 0.05, 3)
        # 姿态微小扰动
        self.uav.orientation += np.random.normal(0, 0.002, 3)
        # 初始速度微小扰动
        self.uav.velocity += np.random.normal(0, 0.1, 3)
        # 物理参数微小扰动
        self.uav.mass *= (1 + np.random.normal(0, 0.05))
        self.uav.k_d *= (1 + np.random.normal(0, 0.05))

    '''
    def get_episode_data(self):
        """获取 episode 数据用于分析"""
        return {
            'trajectory': np.array(self.trajectory_history),
            'rewards': np.array(self.reward_history),
            'goal_position': self.goal_position,
            'initial_position': self.initial_position
        }

    def visualize_step(self, visualizer, max_steps=None):
        """实时可视化当前step的状态"""
        episode_data = self.get_episode_data()
        visualizer.update_dynamic(
            trajectory=episode_data['trajectory'],
            orientations=np.array([self.uav.orientation.copy() for _ in episode_data['trajectory']]),
            velocities=np.array([self.uav.velocity.copy() for _ in episode_data['trajectory']]),
            goal_position=self.goal_position,
            max_steps=max_steps or self.max_steps
        )
    '''

    def save_fly_data(self, save_dir=None):
        # 如果未提供保存目录，则使用默认目录
        if save_dir is None:
            save_dir = f"QuadFlyEnv_data"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"episode{self.current_episode}_reward{sum(self.reward_history):.2f}_goal_achieved{self.check_goal_achieved()}.json"
        filepath = os.path.join(save_dir, filename)

        episode_data = {
            'trajectory': [arr.tolist() for arr in self.trajectory_history],
            'orientations': [arr.tolist() for arr in self.orientation_history],
            'velocities': [arr.tolist() for arr in self.velocity_history],
            'rewards': self.reward_history,
            'goal_position': self.goal_position.tolist(),
            'total_steps': self.current_step
        }

        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2)

        return filepath

    def close(self):
        pass
