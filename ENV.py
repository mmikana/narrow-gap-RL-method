import numpy as np
import gymnasium as gym
from gymnasium import spaces

from QuadrotorDynamics import QuadrotorDynamics
from NarrowGap import NarrowGap
from collision_detector import CollisionDetector


class QuadrotorEnv(gym.Env):
    def __init__(self):
        # UAV动力学初始化
        self.uav = QuadrotorDynamics()
        # 动作：归一化的转速[u1,u2,u3,u4], Quadrotor中可以更改为标准动作输出，可以具有更好的物理解释，
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
        # 状态：位置(3) + 速度(3) + 姿态(3) + 角速度(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,))
        # NarrowGap初始化，wall中心坐标，高长厚度，gap中心坐标、长宽
        self.detector = CollisionDetector()

        self.NarrowGap = NarrowGap()
        self.goal_position = np.array([0, 0.25, 0])  # 缝隙后25cm处

        self.gamma = 0.99
        self.max_steps = 500
        self.current_step = 0

        # Reward settings
        self.reward_achievegoal = 1000
        self.collision_penalty = 1000

        # curriculum learning
        self.current_difficulty = 0
        self.difficulty_levels = [
            {'gap_size': (1.0, 0.38), 'tilt': 0},
            {'gap_size': (0.8, 0.36), 'tilt': 10},
            {'gap_size': (0.7, 0.34), 'tilt': 20},
        ]

        # 数据记录
        self.trajectory_history = []
        self.orientation_history = []
        self.velocity_history = []
        self.reward_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav.reset(position=[0, 0, 1], orientation=[0, 0, 0])
        self.apply_randomization()
        self.current_step = 0
        # 重置轨迹记录
        self.uav_trajectory = [self.uav.position.copy()]
        self.uav_orientation_history = [self.uav.orientation.copy()]

        return self.uav.get_obs(), {}

    def step(self, action):
        self.current_step += 1
        reward_step = 0

        # 动作执行
        motor_speeds = self.uav.normalized_action_to_motor_speeds(action)
        self.uav.update(motor_speeds)

        # 记录轨迹和姿态
        self.uav_trajectory.append(self.uav.position.copy())
        self.uav_orientation_history.append(self.uav.orientation.copy())

        # 奖励计算
        if self.achieve_goal():
            reward_step += self.reward_achievegoal
        else:
            distance = np.linalg.norm(self.uav.position - self.goal_position)
            reward_step += distance
            orientation_penalty = -np.linalg.norm(self.uav.orientation) * 0.1
            velocity_penalty = -np.linalg.norm(self.uav.velocity) * 0.05
            reward_step += orientation_penalty + velocity_penalty

        # 终止条件
        terminated = (
                self.detector.efficient_collision_check(self.uav, self.NarrowGap) or
                self.achieve_goal() or
                np.linalg.norm(self.uav.position) > 20.0 or
                self.current_step >= self.max_steps
        )

        truncated = False

        # 碰撞惩罚
        if self.detector.efficient_collision_check(self.uav, self.NarrowGap):
            reward_step += self.collision_penalty

        info = {
            "collision": self.detector.efficient_collision_check(self.uav, self.NarrowGap),
            "goal_achieved": self.achieve_goal(),
            "distance_to_goal": np.linalg.norm(self.uav.position - self.goal_position),
            "steps": self.current_step
        }

        # 记录数据
        self.trajectory_history.append(self.uav.position.copy())
        self.orientation_history.append(self.uav.orientation.copy())
        self.velocity_history.append(self.uav.velocity.copy())
        self.reward_history.append(reward_step)

        return self.uav.get_obs(), reward_step, terminated, truncated, info

    def apply_randomization(self):
        """应用环境随机化"""
        # 位置和姿态随机化
        self.uav.position += np.random.normal(0, 0.002, 3)
        self.uav.orientation += np.random.normal(0, 0.01, 3)
        self.uav.velocity += np.random.normal(0, 0.05, 3)
        self.uav.angular_velocity += np.random.normal(0, 0.05, 3)

        # 质量随机化
        self.uav.mass *= (1 + np.random.normal(0, 0.1))

    def increase_difficulty(self):
        """增加环境难度"""
        if self.current_difficulty < len(self.difficulty_levels) - 1:
            self.current_difficulty += 1
            level = self.difficulty_levels[self.current_difficulty]
            self.NarrowGap = NarrowGap(
                gap_length=level['gap_size'][0],
                gap_height=level['gap_size'][1],
                tilt=level['tilt']
            )

    def enter_gap(self):
        """判断 UAV 中心是否进入 GAP 的 3D 包围盒"""
        # 获取缝隙的四个角点
        gap_corners = self.NarrowGap.get_gap_corners()

        # 计算缝隙的边界
        gap_min = np.min(gap_corners, axis=0)
        gap_max = np.max(gap_corners, axis=0)

        # 检查无人机中心是否在缝隙边界内
        return np.all(gap_min <= self.uav.position) and np.all(self.uav.position <= gap_max)

    def achieve_goal(self):
        # 计算无人机到缝隙平面的距离（沿法向量方向）
        dist_to_gap = (self.uav.position - self.NarrowGap.center) @ self.NarrowGap.normal

        # 判断是否到达达目标区域
        goal_distance = 0.5 * self.NarrowGap.gap_thickness + 0.5 * self.uav.size[0]
        if dist_to_gap > goal_distance:
            return True
        return False

    def close(self):
        pass

    def get_episode_data(self):
        """获取整个episode的数据"""
        # 确保所有数组长度一致
        min_length = min(len(self.trajectory_history),
                         len(self.orientation_history),
                         len(self.velocity_history))

        return {
            'trajectory': np.array(self.trajectory_history[:min_length]),
            'orientations': np.array(self.orientation_history[:min_length]),
            'velocities': np.array(self.velocity_history[:min_length]),
            'rewards': np.array(self.reward_history[:min_length])
        }