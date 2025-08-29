import numpy as np
from scipy.spatial.transform import Rotation


class QuadrotorDynamics:
    """四旋翼无人机动力学模型

    实现了论文中的四旋翼物理模型，包括：
    - 电机推力计算
    - 姿态动力学
    - 位置动力学
    标准动作到底是F和τ还是转速需要进行调试
    """

    def __init__(self):
        """初始化无人机物理参数（与论文完全一致）"""
        # 状态变量
        self.position = np.array([0.0, 10, 0.0], dtype=np.float64)
        self.orientation = np.zeros(3, dtype=np.float64)
        self.orientation = np.radians(self.orientation)  # [roll,pitch,yaw] 角度,[0,2Π],[-0.5Π,0.5Π][0,2Π]
        self.velocity = np.zeros(3, dtype=np.float64)  # [vX,vY,vZ] 线速度 (m/s),Body Frame
        self.angular_velocity = np.zeros(3, dtype=np.float64)  # [ωX,ωY,ωZ] 角速度 (rad/s),Body Frame

        # 朝向
        self.local_x = np.array([1, 0, 0])
        self.local_y = np.array([0, 1, 0])
        self.local_z = np.array([0, 0, 1])
        self.inertial_x = np.array([1, 0, 0])
        self.inertial_y = np.array([1, 0, 0])
        self.inertial_z = np.array([1, 0, 0])

        # 物理参数（论文第5页C）
        self.mass = 1.2  # 质量 (kg)
        self.inertia = np.diag([0.007, 0.007, 0.014])  # 惯性矩阵 (kg·m²)
        self.size = np.array([0.47, 0.47, 0.23])  # 假设无人机尺寸 0.47x0.47x0.23m
        self.arm_length = self.size[0] * np.sqrt(2) / 2  # X型机臂对角线长度 (m)
        self.C_T = 6e-6  # 推力系数 (N/(rad/s)^2)
        self.C_M = 8e-8  # 扭矩系数 (N·m/(rad/s)^2)
        self.max_motor_speed = 1500  # 电机最大转速 (rad/s)


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

    def update(self, motor_speeds, dt=0.02):
        """基于电机转速更新动力学状态（实现论文公式1-3）
        Args:
            motor_speeds: [ω1,ω2,ω3,ω4] 四个电机转速 (rad/s)
            dt: 时间步长 (s)
        """
        # 1. 计算推力和扭矩（公式3）
        thrusts = self.C_T * np.square(motor_speeds)
        torques = self.C_M * np.square(motor_speeds)

        # 2. 计算总推力和力矩
        total_thrust = np.sum(thrusts)  # 总推力 (N)
        tau_phi = self.arm_length * (thrusts[0] - thrusts[1] - thrusts[2] + thrusts[3])  # 滚转力矩
        tau_theta = self.arm_length * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])  # 俯仰力矩
        tau_psi = np.sum([torques[0], -torques[1], torques[2], -torques[3]])  # 偏航力矩

        # 3. 计算角加速度（公式1）
        omega = self.angular_velocity
        angular_acc = np.array([
            (tau_phi - (self.inertia[1, 1] - self.inertia[2, 2]) * omega[1] * omega[2]) / self.inertia[0, 0],
            (tau_theta - (self.inertia[2, 2] - self.inertia[0, 0]) * omega[0] * omega[2]) / self.inertia[1, 1],
            (tau_psi - (self.inertia[0, 0] - self.inertia[1, 1]) * omega[0] * omega[1]) / self.inertia[2, 2]
        ])

        # 4. 更新角速度和姿态（欧拉法）
        self.angular_velocity += angular_acc * dt
        self.orientation += self.angular_velocity * dt
        # 角度归一化到[-π, π]
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

        '''
        # 4. 更新角速度和姿态（二阶泰勒展开）
        self.orientation += (self.angular_velocity * dt + 0.5 * self.angular_velocity * dt ** 2)
        self.angular_velocity += angular_acc * dt
        # 角度归一化到[-π, π]
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi
        '''

        rot = Rotation.from_euler('xyz', self.orientation).as_matrix()
        # 局部坐标变惯性坐标系
        self.inertial_x = rot @ self.local_x
        self.inertial_y = rot @ self.local_y
        self.inertial_z = rot @ self.local_z
        # 5. 计算线加速度（论文公式2）
        thrust_vector = rot @ np.array([0, 0, total_thrust])  # 机体坐标系转世界坐标系
        gravity = np.array([0, 0, -9.81 * self.mass])  # 重力
        drag_force = -0.1 * np.abs(self.velocity) * self.velocity  # 简化的空气阻力模型

        acceleration = (thrust_vector + gravity + drag_force) / self.mass

        # 6. 更新速度和位置（欧拉法）
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        '''
        # 6. 更新速度和位置（二阶泰勒展开法）
        self.position += (self.velocity * dt + 0.5 * acceleration * dt ** 2)
        self.velocity += acceleration * dt
        '''

    def normalized_motor_speeds2motor_speeds(self, normalized_motor_speeds):
        """将归一化的电机转速转换为真实电机转速
        Args:
            normalized_motor_speeds: [ω1_norm, ω2_norm, ω3_norm, ω4_norm] 归一化转速，范围[-1, 1]
                                     其中+1对应最大正转转速，-1对应最大反转转速
        Returns:
            [ω1, ω2, ω3, ω4] 真实电机转速 (rad/s)
        """
        # 将归一化转速映射到真实转速范围：[-max_motor_speed, max_motor_speed]
        motor_speeds = normalized_motor_speeds * self.max_motor_speed

        return motor_speeds

    def physical_to_normalized_action(self, thrust_physical, roll_physical, pitch_physical, yaw_physical):
        """将物理量转换为标准化动作[-1,1]范围，确保与normalized_action_to_motor_speeds对齐

        Args:
            thrust_physical: 总推力 (N) 范围 [0, max_thrust]
            roll_physical: 滚转力矩 (N·m) 范围 [-max_roll, max_roll]
            pitch_physical: 俯仰力矩 (N·m) 范围 [-max_pitch, max_pitch]
            yaw_physical: 偏航力矩 (N·m) 范围 [-max_yaw, max_yaw]

        Returns:
            [thrust_norm, roll_norm, pitch_norm, yaw_norm] 各在[-1,1]范围
        """
        # 计算最大推力（4个电机全速运转）
        max_thrust = 4 * self.C_T * self.max_motor_speed ** 2

        # 计算最大力矩（基于电机最大转速和分配矩阵）
        max_roll = 2 * self.arm_length * self.C_T * self.max_motor_speed ** 2  # 最大滚转力矩
        max_pitch = 2 * self.arm_length * self.C_T * self.max_motor_speed ** 2  # 最大俯仰力矩
        max_yaw = 2 * self.C_M * self.max_motor_speed ** 2  # 最大偏航力矩

        # 归一化处理
        thrust_norm = 2 * (thrust_physical / max_thrust) - 1  # [0, max_thrust] → [-1, 1]
        roll_norm = roll_physical / max_roll  # [-max_roll, max_roll] → [-1, 1]
        pitch_norm = pitch_physical / max_pitch  # [-max_pitch, max_pitch] → [-1, 1]
        yaw_norm = yaw_physical / max_yaw  # [-max_yaw, max_yaw] → [-1, 1]

        # 裁剪到[-1, 1]范围
        return np.clip([thrust_norm, roll_norm, pitch_norm, yaw_norm], -1, 1)

    def normalized_action_to_motor_speeds(self, action):
        """将标准化动作[-1,1]^4转换为电机转速
        Args:
            action: [thrust, roll, pitch, yaw] 各在[-1,1]范围
        Returns:
            [ω1,ω2,ω3,ω4] 电机转速 (rad/s)
        """
        # 将thrust从[-1,1]映射到[0,1]
        u1 = (action[0] + 1) / 2  # 总推力控制量 [0,1]
        u2 = action[1]  # 滚转控制量 [-1,1]
        u3 = action[2]  # 俯仰控制量 [-1,1]
        u4 = action[3]  # 偏航控制量 [-1,1]

        # 控制分配矩阵（公式3的伪逆）
        allocation_matrix = np.array([
            [self.C_T, self.C_T, self.C_T, self.C_T],
            [self.arm_length * self.C_T, -self.arm_length * self.C_T, -self.arm_length * self.C_T,
             self.arm_length * self.C_T],
            [self.arm_length * self.C_T, self.arm_length * self.C_T, -self.arm_length * self.C_T,
             -self.arm_length * self.C_T],
            [self.C_M, -self.C_M, self.C_M, -self.C_M]
        ])

        # 计算控制输入
        control_input = np.array([
            u1 * self.max_motor_speed ** 2 * self.C_T * 4,  # 总推力
            u2 * self.max_motor_speed ** 2 * self.C_T * self.arm_length * 2,  # 滚转
            u3 * self.max_motor_speed ** 2 * self.C_T * self.arm_length * 2,  # 俯仰
            u4 * self.max_motor_speed ** 2 * self.C_M * 2  # 偏航
        ])
        # 计算电机转速平方并裁剪
        motor_speeds_sq = np.linalg.pinv(allocation_matrix) @ control_input
        return np.sqrt(np.clip(motor_speeds_sq, 0, self.max_motor_speed ** 2))

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
