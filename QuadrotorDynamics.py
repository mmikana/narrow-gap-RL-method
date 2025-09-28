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
        self.max_motor_speed = 1500.0  # 最大转速 (rad/s)
        self.delta_M = 0.05  # 电机的单步响应时间间隔，可根据实际情况调整
        self.c = np.exp(-self.dt / self.delta_M)  # 电机滞后响应系数
        self.prev_motor_speeds = np.zeros(4)  # 上一时刻的电机转速

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

    def get_obs(self):
        """获取当前无人机的观测状态，返回一个NumPy数组。
        Returns:
            np.ndarray: 形状为 (12,) 的数组，包含：
                - 位置 [x, y, z]
                - 速度 [vX, vY, vZ]
                - 姿态 [roll, pitch, yaw]
                - 角速度 [ωX, ωY, ωZ]
        """
        observation = np.concatenate([
            self.position,  # [x, y, z]
            self.velocity,  # [vX, vY, vZ]
            self.orientation,  # [roll, pitch, yaw]
            self.angular_velocity  # [ωX, ωY, ωZ]
        ])
        return observation.astype(np.float32)  # 统一数据类型

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
        angular_acc, linear_acc = self.compute_acceleration(motor_speeds, current_orientation,
                                                            current_angular_velocity, current_velocity)

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

        # 位置和速度更新
        self.position, self.velocity = \
            (self.rk4_update_from_derivatives(current_position, current_velocity, linear_acc, dt))

        # 惯性坐标系指向更新
        rot = Rotation.from_euler('xyz', self.orientation).as_matrix()
        self.inertial_x = rot @ self.local_x
        self.inertial_y = rot @ self.local_y
        self.inertial_z = rot @ self.local_z

    #  TODO
    def normalized_action_to_motor_speeds(self, normalized_action):

        # 控制分配矩阵（公式3的伪逆）
        allocation_matrix = np.array([
            [np.sqrt(2) / 2 * self.arm_length * self.C_T,
             np.sqrt(2) / 2 * self.arm_length * self.C_T,
             -np.sqrt(2) / 2 * self.arm_length * self.C_T,
             -np.sqrt(2) / 2 * self.arm_length * self.C_T],
            [-np.sqrt(2) / 2 * self.arm_length * self.C_T,
             np.sqrt(2) / 2 * self.arm_length * self.C_T,
             -np.sqrt(2) / 2 * self.arm_length * self.C_T,
             np.sqrt(2) / 2 * self.arm_length * self.C_T],
            [-self.C_M, self.C_M, -self.C_M, self.C_M],
            [self.C_T, self.C_T, self.C_T, self.C_T]
        ])
        thrust_norm, roll_norm, pitch_norm, yaw_norm = normalized_action
        # 计算最大物理量（与原方法保持一致的基准值）
        max_thrust = 4 * self.C_T * self.max_motor_speed ** 2
        max_roll = 2 * self.arm_length * self.C_T * self.max_motor_speed ** 2
        max_pitch = 2 * self.arm_length * self.C_T * self.max_motor_speed ** 2
        max_yaw = 2 * self.C_M * self.max_motor_speed ** 2

        # 反归一化：直接映射（不裁剪）
        thrust_physical = (thrust_norm + 1) / 2 * max_thrust  # [-1,1] → [0, max_thrust]（无裁剪时可能超出）
        roll_physical = roll_norm * max_roll  # [-1,1] → [-max_roll, max_roll]（无裁剪时可能超出）
        pitch_physical = pitch_norm * max_pitch
        yaw_physical = yaw_norm * max_yaw
        control_input = np.array([thrust_physical, roll_physical, pitch_physical, yaw_physical]).reshape(-1, 1)

        # 计算电机转速平方并裁剪
        motor_speeds_sq = np.linalg.pinv(allocation_matrix) @ control_input
        motor_speeds = np.sqrt(motor_speeds_sq)
        return np.clip(motor_speeds, self.min_motor_speed, self.max_motor_speed)

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

    def compute_acceleration(self, motor_speeds, orientation, angular_velocity, velocity):
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

        rot = Rotation.from_euler('xyz', orientation).as_matrix()
        thrust_vector = rot @ np.array([0, 0, total_thrust])
        gravity = np.array([0, 0, -self.g * self.mass])
        drag_force = -self.k_d * velocity * np.linalg.norm(velocity)
        linear_acc = (thrust_vector + gravity + drag_force) / self.mass

        return angular_acc, linear_acc

    def rk4_update_from_derivatives(self, current_state, first_deriv, second_deriv, dt):

        # 定义状态向量：合并状态值和一阶导数 [state, first_deriv]
        def state_derivative(state_vector):
            # 状态向量的导数 = [一阶导数, 二阶导数]
            return np.concatenate([state_vector[1:], second_deriv])

        # 初始状态向量：[当前状态, 一阶导数]
        initial_state = np.concatenate([current_state, first_deriv])

        # RK4四步计算（无需额外参数，直接使用二阶导数）
        k1 = state_derivative(initial_state)
        k2 = state_derivative(initial_state + 0.5 * dt * k1)
        k3 = state_derivative(initial_state + 0.5 * dt * k2)
        k4 = state_derivative(initial_state + dt * k3)

        # 计算新状态向量
        new_state_vector = initial_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 拆分更新后的状态和一阶导数
        new_state = new_state_vector[:len(current_state)]
        new_first_deriv = new_state_vector[len(current_state):]

        return new_state, new_first_deriv
