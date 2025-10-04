import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json

from .QuadrotorDynamics import QuadrotorDynamics
from .NarrowGap import NarrowGap
from .collision_detector import CollisionDetector
from .i3utils import Vector3


class Quad2NGEnv(gym.Env):
    def __init__(self):
        self.uav = QuadrotorDynamics()
        # 动作：归一化的转速[u1,u2,u3,u4]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
        # 状态：位置(3) + 速度(3) + 姿态(3) + 角速度(3) + NG姿态(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(15,))
        # 碰撞检测与缝隙环境初始化
        self.detector = CollisionDetector()
        self.NarrowGap = NarrowGap()

        #  TODO  new variable
        self.goal_tolerance = 0.1

        self.gamma = 0.99
        #  TODO need debug
        self.max_steps = 500
        self.current_step = 0

        #  TODO Reward need debug
        self.reward_config = {
            'reward_achievegoal': 0,
            'collision_penalty': -40,
            'e_0_p': 1.2,
            'orientation_weight': 0,
            'position_weight': 1,
            'speed_weight': 0,
            'ideal_speed': 1.5
        }
        # 初始化奖励计算器
        self.reward_calculator = self.RewardCalculator(self.reward_config)

        # TODO
        self.current_difficulty = 0
        self.difficulty_levels = [
            {'gap_size': (3.0 * self.uav.size[0], 3.0 * self.uav.size[2]), 'tilt': 0},
            {'gap_size': (2.5 * self.uav.size[0], 3.0 * self.uav.size[2]), 'tilt': 10},
            {'gap_size': (2.0 * self.uav.size[0], 3.0 * self.uav.size[2]), 'tilt': 20},
        ]

        # 数据记录
        self.trajectory_history = []
        self.orientation_history = []
        self.velocity_history = []
        self.reward_history = []
        self.current_episode = 0
        self.plot = True

        # curriculum learning
        self.gap_scales = [3.0, 2.5, 2.0]  # L1, L2, L3
        self.rotation_ranges = [
            (-20, 20),  # R_A
            (-40, 40),  # R_B
            (-60, 60)  # R_C
        ]
        self.tilt_ranges = [
            (-10, 10),  # T_A
            (-20, 20),  # T_B
            (-30, 30)  # T_C
        ]
        self.levels = self._generate_levels()  # TODO
        self.current_level_idx = 0
        self.success_counter = 0
        self.unlock_threshold = 50
        self._update_environment()  # TODO

    def _generate_levels(self):
        """生成27个关卡的配置列表（按阶段顺序排列）"""
        levels = []
        # 阶段1-3：L1（3.0倍缝隙）+ 所有旋转/倾斜组合
        for rotation_idx in [0, 1, 2]:  # R_A, R_B, R_C 依次递进
            for t_idx in [0, 1, 2]:  # T_A→T_B→T_C
                levels.append({
                    'gap_scale': self.gap_scales[0],
                    'rotation_range': self.rotation_ranges[rotation_idx],
                    'tilt_range': self.tilt_ranges[t_idx]
                })

        # 阶段4-6：L2（2.5倍缝隙）+ 所有旋转/倾斜组合
        for rotation_idx in [0, 1, 2]:
            for t_idx in [0, 1, 2]:
                levels.append({
                    'gap_scale': self.gap_scales[1],
                    'rotation_range': self.rotation_ranges[rotation_idx],
                    'tilt_range': self.tilt_ranges[t_idx]
                })

        # 阶段7-9：L3（2.0倍缝隙）+ 所有旋转/倾斜组合
        for rotation_idx in [0, 1, 2]:
            for t_idx in [0, 1, 2]:
                levels.append({
                    'gap_scale': self.gap_scales[2],
                    'rotation_range': self.rotation_ranges[rotation_idx],
                    'tilt_range': self.tilt_ranges[t_idx]
                })
        return levels

    def _update_environment(self):
        """根据当前关卡更新缝隙参数（随机采样角度）"""
        if self.current_level_idx >= len(self.levels):
            self.current_level_idx = len(self.levels) - 1  # 保持最高难度

        level = self.levels[self.current_level_idx]
        # 计算缝隙尺寸（基于无人机尺寸）
        gap_length = level['gap_scale'] * self.uav.size[0]
        gap_height = level['gap_scale'] * self.uav.size[2]

        # 随机采样rotation和tilt角度（在当前关卡范围内）
        rotation = np.random.uniform(
            level['rotation_range'][0],
            level['rotation_range'][1]
        )
        tilt = np.random.uniform(
            level['tilt_range'][0],
            level['tilt_range'][1]
        )

        # 更新缝隙
        self.NarrowGap = NarrowGap(
            gap_length=gap_length,
            gap_height=gap_height,
            rotation=rotation,  # 旋转角（度）
            tilt=tilt  # 倾斜角（度）
        )

    def _check_level_unlock(self, goal_achieved):
        """检查是否解锁下一关卡"""
        if goal_achieved:
            self.success_count += 1
            # 达到解锁条件且未到最后一关
            if (self.success_count >= self.unlock_threshold and
                self.current_level_idx < len(self.levels) - 1):
                self.current_level_idx += 1
                self._update_environment()
                self.success_count = 0  # 重置计数器
                print(f"解锁新关卡 {self.current_level_idx + 1}/27 | 配置: {self.levels[self.current_level_idx]}")
        else:
            self.success_count = 0  # 失败则重置连续计数

    class RewardCalculator:
        def __init__(self, config):
            # 从配置中初始化奖励相关参数
            self.reward_achievegoal = config.get('reward_achievegoal')
            self.collision_penalty = config.get('collision_penalty')
            self.e_0_p = config.get('e_0_p', 1.2)
            self.orientation_weight = config.get('orientation_weight')
            self.position_weight = config.get('position_weight')
            self.speed_weight = config.get('speed_weight')
            self.ideal_speed = config.get('ideal_speed')

        def calculate_reward(self, uav, narrow_gap, goal_position, collision):
            """计算单步奖励"""
            reward_step = 0

            # 碰撞惩罚
            if collision:
                reward_step += self.collision_penalty

            # 到达目标奖励
            e_t_p = np.linalg.norm(uav.position - goal_position)
            goal_distance = 0.5 * narrow_gap.gap_thickness + 0.5 * uav.size[0]
            if e_t_p < goal_distance:
                reward_step += self.reward_achievegoal

            # 位置奖励
            reward_distance = self.position_weight * np.exp(-e_t_p)
            reward_step += reward_distance

            # 姿态奖励
            h_t_p = np.maximum(1 - (e_t_p / self.e_0_p), 0)
            e_t_psi = np.arccos(np.dot(uav.inertial_x, narrow_gap.gap_x)
                                / (np.linalg.norm(uav.inertial_x) * np.linalg.norm(narrow_gap.gap_x) + 1e-10))
            e_t_phi = np.arccos(np.dot(uav.inertial_y, narrow_gap.gap_y)
                                / (np.linalg.norm(uav.inertial_y) * np.linalg.norm(narrow_gap.gap_y) + 1e-10))
            e_t_theta = np.arccos(np.dot(uav.inertial_z, narrow_gap.gap_z)
                                / (np.linalg.norm(uav.inertial_z) * np.linalg.norm(narrow_gap.gap_z) + 1e-10))
            e_t_ori_2 = np.square(e_t_phi) + np.square(e_t_theta) + np.square(e_t_psi)
            reward_orientation = -self.orientation_weight * np.square(h_t_p) * (1 - np.exp(-e_t_ori_2))
            reward_step += reward_orientation

            # 速度奖励
            e_speed = abs(np.linalg.norm(uav.velocity) - self.ideal_speed)
            speed_reward = -self.speed_weight * e_speed / self.ideal_speed
            reward_step += speed_reward

            return reward_step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav.reset(position=[0, 0, 1], orientation=[0, 0, 0])
        self.apply_randomization()
        self.current_step = 0

        self.apply_randomization()

        self.trajectory_history = []
        self.orientation_history = []
        self.velocity_history = []
        self.reward_history = []

        self._update_environment()
        return self.get_obs(), {}

    def get_obs(self):
        uav_pos = np.copy(self.uav.position)
        uav_vel = np.copy(self.uav.velocity)
        uav_ori = np.copy(self.uav.orientation)
        uav_ang_vel = np.copy(self.uav.angular_velocity)

        to_goal_pos = Vector3(self.NarrowGap.center - np.copy(self.uav.position))
        to_goal_pos = to_goal_pos.rev_rotate_zyx_self(uav_ori[0], uav_ori[1], uav_ori[2]).vec
        pos_limit = 5 * to_goal_pos
        to_goal_pos /= pos_limit
        uav_vel /= 9  # 3 * ideal_vel

        uav_ori[0] /= np.pi
        uav_ori[1] /= (0.5 * np.pi)
        uav_ori[2] /= np.pi

        uav_ang_vel[0] /= 300
        uav_ang_vel[1] /= 300
        uav_ang_vel[2] /= 50

        ng_roll = np.copy(self.NarrowGap.rotation)
        ng_pitch = np.copy(self.NarrowGap.tilt)
        ng_roll /= np.pi
        ng_pitch /= (0.5 * np.pi)
        ng_ori = [ng_roll, ng_pitch, 0]
        normalized_obs = np.concatenate([
            to_goal_pos,
            uav_vel,
            uav_ori,
            uav_ang_vel,
            ng_ori
        ]).astype(np.float32)
        return normalized_obs

    def step(self, action):
        self.current_step += 1
        # 动作执行
        motor_speeds = self.uav.normalized_action_to_motor_speeds(action)
        self.uav.update(motor_speeds)

        # 检查碰撞状态
        collision = self.detector.efficient_collision_check(self.uav, self.NarrowGap)
        # 终止条件
        terminated = (
                collision or
                self.achieve_goal() or
                self.current_step >= self.max_steps
        )
        truncated = False
        # 计算奖励
        reward_step = self.reward_calculator.calculate_reward(self.uav, self.NarrowGap, self.NarrowGap.center,
                                                              collision)

        # 记录数据
        self.trajectory_history.append(self.uav.position.copy())
        self.orientation_history.append(self.uav.orientation.copy())
        self.velocity_history.append(self.uav.velocity.copy())
        self.reward_history.append(reward_step)

        #  保存绘制
        if terminated or truncated:
            self.current_episode += 1
            if self.plot:
                save_data_dir = os.path.join("Quad2NGEnv_data")
                data_filepath = self.save_fly_data(save_data_dir)
                self.save_fly_data()
                from Env.episode_visualizer import EpisodeVisualizer
                visualizer = EpisodeVisualizer()
                save_plot_dir = os.path.join("QuadFlyEnv_plot")
                visualizer.draw_fly_data(data_filepath=data_filepath, save_plot_dir=save_plot_dir)

        self._check_level_unlock(self.achieve_goal())

        info = {
            "collision": collision,
            "goal_achieved": self.achieve_goal(),
            "distance_to_goal": np.linalg.norm(self.uav.position - self.NarrowGap.center),
            "steps": self.current_step
        }

        return self.get_obs(), reward_step, terminated, truncated, info

    def apply_randomization(self):
        """应用环境随机化"""
        # 位置和姿态随机化
        self.uav.position += np.random.normal(0, 0.002, 3)
        self.uav.orientation += np.random.normal(0, 0.02, 3)
        self.uav.velocity += np.random.normal(0, 0.05, 3)
        self.uav.angular_velocity += np.random.normal(0, 0.05, 3)

    #  TODO
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
        dist_to_gap = np.linalg.norm(self.uav.position - self.NarrowGap.center)

        # 判断是否到达达目标区域
        if dist_to_gap < self.goal_tolerance:
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

    def save_fly_data(self, save_dir=None):
        # 如果未提供保存目录，则使用默认目录
        if save_dir is None:
            save_dir = f"Quad2NGEnv_data"
        os.makedirs(save_dir, exist_ok=True)

        # 构建包含缝隙信息的文件名
        filename = (f"episode{self.current_episode}_reward{sum(self.reward_history):.2f}_"
                    f"goal_achieved{self.achieve_goal()}.json")
        filepath = os.path.join(save_dir, filename)

        # 收集缝隙相关数据
        gap_data = {
            'center': self.NarrowGap.center.tolist(),
            'gap_length': self.NarrowGap.gap_length,
            'gap_height': self.NarrowGap.gap_height,
            'gap_thickness': self.NarrowGap.gap_thickness,
            'rotation': self.NarrowGap.rotation,
            'tilt': self.NarrowGap.tilt,
        }

        episode_data = {
            'trajectory': [arr.tolist() for arr in self.trajectory_history],
            'orientations': [arr.tolist() for arr in self.orientation_history],
            'velocities': [arr.tolist() for arr in self.velocity_history],
            'rewards': self.reward_history,
            'goal_position': (self.NarrowGap.center[0] + self.NarrowGap.gap_half_thickness),
            'narrow_gap': gap_data,
            'total_steps': self.current_step,
            'goal_achieved': self.achieve_goal()
        }

        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2)

        return filepath