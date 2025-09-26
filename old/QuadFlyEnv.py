import numpy as np
import gymnasium as gym
from gymnasium import spaces

from QuadrotorDynamics import QuadrotorDynamics


class QuadFlyEnv(gym.Env):
    """无人机定点控制环境（无地面/边界版本）"""

    def __init__(self, goal_position=np.array([5.0, 5.0, 5.0])):
        # 无人机动力学初始化
        self.uav = QuadrotorDynamics()

        # 动作空间：归一化的控制指令[thrust, roll, pitch, yaw]，范围[-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # 观测空间：位置(3) + 速度(3) + 姿态(3) + 角速度(3) + 目标位置(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # 包含目标位置信息
            dtype=np.float32
        )

        # 目标位置设置
        self.goal_position = np.array(goal_position, dtype=np.float64)
        self.initial_position = np.array([0, 0, 0], dtype=np.float64)  # 默认初始位置

        # 训练参数
        self.max_steps = 200  # 单轮最大步数
        self.current_step = 0

        # 奖励配置（无边界相关惩罚）
        self.reward_config = {
            'goal_reward': 100.0,  # 到达目标奖励
            'distance_weight': 1.0,  # 距离权重
            'ideal_velocity': 1.0,
            'velocity_weight': 0,  # 速度权重
            'orientation_weight': 0,  # 姿态权重
            'angular_velocity_weight': 0,  # 角速度权重
            'goal_tolerance': 0.2  # 到达目标的位置容差(m)
        }

        # 数据记录
        self.trajectory_history = []
        self.reward_history = []

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
            orientation=[0, 0, 0]  # 初始姿态水平
        )

        # 应用状态随机化增加训练鲁棒性
        self.apply_randomization()

        self.current_step = 0
        # 重置历史记录
        self.trajectory_history = []
        self.reward_history = []

        return self.get_obs(), {}

    def get_obs(self):
        """获取观测：无人机状态 + 目标位置"""
        uav_obs = self.uav.get_obs()  # 12维基础观测
        return np.concatenate([uav_obs, self.goal_position]).astype(np.float32)

    def step(self, action):
        self.current_step += 1

        # 执行动作：将归一化动作转换为电机转速并更新动力学
        motor_speeds = self.uav.normalized_action_to_motor_speeds(action)
        self.uav.update(motor_speeds)

        # 检查终止条件（仅判断是否到达目标或步数耗尽）
        goal_achieved = self.check_goal_achieved()
        terminated = goal_achieved
        truncated = self.current_step >= self.max_steps

        # 计算奖励
        reward = self.calculate_reward(goal_achieved)

        # 记录数据
        self.trajectory_history.append(self.uav.position.copy())
        self.reward_history.append(reward)

        # 额外信息
        info = {
            "goal_achieved": goal_achieved,
            "distance_to_goal": np.linalg.norm(self.uav.position - self.goal_position),
            "steps": self.current_step
        }

        return self.get_obs(), reward, terminated, truncated, info

    def calculate_reward(self, goal_achieved):
        reward_step = 0.0

        # 到达目标奖励
        if goal_achieved:
            reward_step += self.reward_config['goal_reward']

        # 距离奖励（基于到目标的距离）
        distance = np.linalg.norm(self.uav.position - self.goal_position)
        reward_step -= self.reward_config['distance_weight'] * distance

        # # 速度惩罚（抑制过快移动）
        # velocity_error = np.linalg.norm(self.uav.velocity - self.reward_config['ideal_velocity'])
        # reward_step -= self.reward_config['velocity_weight'] * velocity_error
        #
        # # 姿态惩罚（保持水平姿态）
        # orientation_norm = np.linalg.norm(self.uav.orientation)
        # reward_step -= self.reward_config['orientation_weight'] * orientation_norm
        #
        # # 角速度惩罚（抑制剧烈旋转）
        # angular_vel_norm = np.linalg.norm(self.uav.angular_velocity)
        # reward_step -= self.reward_config['angular_velocity_weight'] * angular_vel_norm

        return reward_step

    def check_goal_achieved(self):
        """检查是否到达目标位置"""
        distance = np.linalg.norm(self.uav.position - self.goal_position)
        return distance < self.reward_config['goal_tolerance']

    def apply_randomization(self):
        """应用状态随机化，增加训练泛化性"""
        # 位置微小扰动
        self.uav.position += np.random.normal(0, 0.05, 3)
        # 姿态微小扰动
        self.uav.orientation += np.random.normal(0, 0.02, 3)
        # 初始速度微小扰动
        self.uav.velocity += np.random.normal(0, 0.1, 3)
        # 物理参数微小扰动
        self.uav.mass *= (1 + np.random.normal(0, 0.05))
        self.uav.k_d *= (1 + np.random.normal(0, 0.05))

    def get_episode_data(self):
        """获取 episode 数据用于分析"""
        return {
            'trajectory': np.array(self.trajectory_history),
            'rewards': np.array(self.reward_history),
            'goal_position': self.goal_position,
            'initial_position': self.initial_position
        }

    def close(self):
        pass
