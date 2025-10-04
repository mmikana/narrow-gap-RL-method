"""
2025/9/28
修改空气动力学
关于orientation所有的计算全都使用rad进行计算，只在接口处留有角度到弧度的变换
"""
import numpy as np
from scipy.spatial.transform import Rotation


class QuadrotorDynamics:
    """
    四旋翼无人机动力学模型

    实现了论文中的四旋翼物理模型，包括：
    - 电机推力计算
    - 姿态动力学
    - 位置动力学

    """

    def __init__(self):
        """初始化无人机物理参数（与论文完全一致）"""
        # 状态变量
        self.position = np.array([0.0, 10, 0.0], dtype=np.float64)
        self.orientation = np.radians(np.zeros(3, dtype=np.float64))  # [roll,pitch,yaw] [-Π,Π],[-0.5Π,0.5Π][-Π,Π]
        self.velocity = np.zeros(3, dtype=np.float64)  # [vX,vY,vZ] 线速度 (m/s)
        self.angular_velocity = np.zeros(3, dtype=np.float64)  # [ωX,ωY,ωZ] 角速度 (rad/s)

        #  时间步
        self.dt = 0.02  # 无人机仿真环境的单步运行时间，与update函数中的dt保持一致

        # 平动动力学参数
        self.mass = 1.2  # 质量 (kg)
        self.g = 9.81  # 重力加速度
        self.k_d = 0.1  # 空气阻力系数

        # 转动动力学参数
        self.size = np.array([0.47, 0.47, 0.23])  # 假设无人机尺寸 0.47x0.47x0.23m
        self.arm_length = self.size[0] * np.sqrt(2) / 2  # X型机臂对角线长度 (m)
        self.inertia = np.diag([0.007, 0.007, 0.014])  # 惯性矩阵 (kg·m²)  # TODO
        self.C_T = 6e-6  # 推力系数 (N/(rad/s)^2)
        self.C_M = 8e-8  # 扭矩系数 (N·m/(rad/s)^2)
        self.d_phi, self.d_theta, self.d_psi = 0.01, 0.01, 0.02  # 需要根据实际调整

        # 电机参数
        self.min_motor_speed = 100.0   # 最小转速 (rad/s)
        self.max_motor_speed = 1000.0  # 最大转速 (rad/s)
        self.delta_M = 0.05  # 电机的单步响应时间间隔，可根据实际情况调整
        self.c = np.exp(-self.dt / self.delta_M)  # 电机滞后响应系数
        self.prev_motor_speeds = np.zeros(4)  # 上一时刻的电机转速

        # 计算最大物理量（基于电机最大转速）
        self.max_thrust = 4 * self.C_T * (self.max_motor_speed ** 2)  # 总推力最大值
        self.max_roll_torque = 2 * self.arm_length * self.C_T * (self.max_motor_speed ** 2)  # 最大滚转力矩
        self.max_pitch_torque = 2 * self.arm_length * self.C_T * (self.max_motor_speed ** 2)  # 最大俯仰力矩
        self.max_yaw_torque = 2 * self.C_M * (self.max_motor_speed ** 2)  # 最大偏航力矩

        # 机体坐标和惯性坐标朝向初始化
        self.local_x = np.array([1, 0, 0])
        self.local_y = np.array([0, 1, 0])
        self.local_z = np.array([0, 0, 1])
        self.inertial_x = np.array([1, 0, 0])
        self.inertial_y = np.array([0, 1, 0])
        self.inertial_z = np.array([0, 0, 1])


    def reset(self, position, orientation):
        """重置无人机状态
        Args:
            position: [x,y,z] 初始位置 (m)
            orientation: [roll,pitch,yaw] 初始姿态 (rad)
        """
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.radians(orientation)
        self.orientation = np.array(self.orientation, dtype=np.float64)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    @property
    def rot(self):
        """获取当前旋转矩阵"""
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler('xyz', self.orientation).as_matrix()

    # def get_obs(self):
    #     """获取当前无人机的观测状态，返回一个NumPy数组。
    #     Returns:
    #         np.ndarray: 形状为 (12,) 的数组，包含：
    #             - 位置 [x, y, z]
    #             - 速度 [vX, vY, vZ]
    #             - 姿态 [roll, pitch, yaw]
    #             - 角速度 [ωX, ωY, ωZ]
    #     """
    #     observation = np.concatenate([
    #         self.position,  # [x, y, z]
    #         self.velocity,  # [vX, vY, vZ]
    #         self.orientation,  # [roll, pitch, yaw]
    #         self.angular_velocity  # [ωX, ωY, ωZ]
    #     ])
    #     return observation.astype(np.float32)  # 统一数据类型

    def update(self, motor_speeds, dt=0.2):

        # 根据电机滞后响应公式更新电机转速
        current_motor_speeds = self.c * self.prev_motor_speeds + (1 - self.c) * motor_speeds
        self.prev_motor_speeds = current_motor_speeds

        # 保存当前状态
        current_orientation = self.orientation.copy()
        current_angular_velocity = self.angular_velocity.copy()
        current_position = self.position.copy()
        current_velocity = self.velocity.copy()

        # 计算总推力（在RK4步骤中保持不变）
        thrust, angular_acc = self.compute_angle_acc(motor_speeds, current_orientation, current_angular_velocity)

        # 角速度和姿态的更新
        self.orientation, self.angular_velocity = \
            (self.rk4_update_from_derivatives(current_orientation, current_angular_velocity, angular_acc, dt))
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi
        # 角度范围约束：pitch限制在±π/2，其他角保持±π
        roll, pitch, yaw = self.orientation
        pitch_clamped = np.clip(pitch, -np.pi / 2, np.pi / 2)
        roll_clamped = (roll + np.pi) % (2 * np.pi) - np.pi
        yaw_clamped = (yaw + np.pi) % (2 * np.pi) - np.pi
        self.orientation = np.array([roll_clamped, pitch_clamped, yaw_clamped])

        rot = Rotation.from_euler('xyz', self.orientation).as_matrix()

        linear_acc = self.compute_linear_acc(thrust, rot, current_velocity)
        # 位置和速度更新
        self.position, self.velocity = \
            (self.rk4_update_from_derivatives(current_position, current_velocity, linear_acc, dt))

        # 惯性坐标系指向更新
        self.inertial_x = rot @ self.local_x
        self.inertial_y = rot @ self.local_y
        self.inertial_z = rot @ self.local_z

    def normalized_action_to_motor_speeds(self, normalized_action):
        """
        将归一化动作直接映射为电机转速
        """
        # 确保输入在[-1, 1]范围内（防止异常值）
        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        # 线性映射：[-1, 1] -> [min_motor_speed, max_motor_speed]
        # 公式：speed = min + (max - min) * (action + 1) / 2
        motor_speeds = self.min_motor_speed + (self.max_motor_speed - self.min_motor_speed) * (
                    normalized_action + 1) / 2

        return motor_speeds

    def get_vertices(self):
        """计算无人机的8个顶点坐标（世界坐标系）"""
        # 无人机半尺寸
        half_x = self.size[0] / 2
        half_y = self.size[1] / 2
        half_z = self.size[2] / 2

        # 机体坐标系下的顶点
        local_vertices = np.array([
            [half_x, half_y, half_z],  # 前右上
            [half_x, half_y, -half_z],  # 前右下
            [half_x, -half_y, half_z],  # 前左上
            [half_x, -half_y, -half_z],  # 前左下
            [-half_x, half_y, half_z],  # 后右上
            [-half_x, half_y, -half_z],  # 后右下
            [-half_x, -half_y, half_z],  # 后左上
            [-half_x, -half_y, -half_z],  # 后左下
        ], dtype=np.float64)

        # 转换到世界坐标系
        rot_matrix = self.rot  # 从姿态获取旋转矩阵
        world_vertices = np.array([
            self.position + rot_matrix @ v for v in local_vertices
        ])

        return world_vertices

    def compute_angle_acc(self, motor_speeds, orientation, angular_velocity):
        """计算角加速度（考虑姿态相关的气动力矩）"""
        thrusts = self.C_T * np.square(motor_speeds)
        torques = self.C_M * np.square(motor_speeds)

        #  升力和转矩，0123分别对应四个坐标系的电机，02cw，13ccw
        total_thrust = np.sum(thrusts)
        tau_phi = np.sqrt(2) / 2 * self.arm_length * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])
        tau_theta = np.sqrt(2) / 2 * self.arm_length * (-thrusts[0] + thrusts[1] + thrusts[2] - thrusts[3])
        tau_psi = np.sum([-torques[0], torques[1], -torques[2], torques[3]])

        # 姿态相关的阻尼系数（示例：俯仰角越大，俯仰阻尼越小）
        roll, pitch, yaw = orientation
        d_phi_effective = self.d_phi * (1 - 0.3 * abs(pitch) / np.pi)  # 俯仰角影响滚转阻尼
        d_theta_effective = self.d_theta * (1 - 0.2 * abs(roll) / np.pi)  # 滚转角影响俯仰阻尼
        d_psi_effective = self.d_psi

        angular_acc = np.array([
            (tau_phi - (self.inertia[1, 1] - self.inertia[2, 2]) * angular_velocity[1] * angular_velocity[2]
             - d_phi_effective * angular_velocity[0]) / self.inertia[0, 0],
            (tau_theta - (self.inertia[2, 2] - self.inertia[0, 0]) * angular_velocity[0] * angular_velocity[2]
             - d_theta_effective * angular_velocity[1]) / self.inertia[1, 1],
            (tau_psi - (self.inertia[0, 0] - self.inertia[1, 1]) * angular_velocity[0] * angular_velocity[1]
             - d_psi_effective * angular_velocity[2]) / self.inertia[2, 2]
        ])
        return total_thrust, angular_acc

    def compute_linear_acc(self, thrust, rot, velocity):
        thrust_vector = rot @ np.array([0, 0, thrust])
        gravity = np.array([0, 0, -self.g * self.mass])
        drag_force = -self.k_d * velocity * np.linalg.norm(velocity)
        linear_acc = (thrust_vector + gravity + drag_force) / self.mass
        return linear_acc



    def rk4_update_from_derivatives(self, current_state, current_rate, acceleration, dt):
        """
        使用RK4方法更新状态和速率（通用实现）
        """
        # 确保输入都是3元素数组
        assert len(current_state) == 3, "状态必须是3元素数组"
        assert len(current_rate) == 3, "速率必须是3元素数组"
        assert len(acceleration) == 3, "加速度必须是3元素数组"

        # 定义状态导数函数：状态的导数是速率，速率的导数是加速度
        def state_derivative(state_vector):
            # 状态向量 = [状态(3), 速率(3)]
            # 导数向量 = [速率(3), 加速度(3)]
            return np.concatenate([state_vector[3:6], acceleration])

        # 初始状态向量：合并状态和速率（共6个元素）
        initial_state = np.concatenate([current_state, current_rate])

        # 计算RK4的四个k值
        k1 = state_derivative(initial_state)
        k2 = state_derivative(initial_state + 0.5 * dt * k1)
        k3 = state_derivative(initial_state + 0.5 * dt * k2)
        k4 = state_derivative(initial_state + dt * k3)

        # 计算更新后的状态向量
        updated_state = initial_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 拆分出新的状态和速率
        new_state = updated_state[:3]
        new_rate = updated_state[3:6]

        return new_state, new_rate
