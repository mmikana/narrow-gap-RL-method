import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class EpisodeVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 12))
        self.setup_layout()

    def setup_layout(self):
        """设置图表布局"""
        # 3D轨迹图
        self.ax_3d = self.fig.add_subplot(231, projection='3d')
        # 位置随时间变化
        self.ax_position = self.fig.add_subplot(234)
        # 姿态角随时间变化
        self.ax_orientation = self.fig.add_subplot(235)
        # 速度随时间变化
        self.ax_velocity = self.fig.add_subplot(236)
        # 姿态示意图
        self.ax_attitude = self.fig.add_subplot(132)

        # 设置标题
        self.ax_3d.set_title('3D Flight Trajectory', fontsize=12, fontweight='bold')
        self.ax_position.set_title('Position vs Time', fontsize=12, fontweight='bold')
        self.ax_orientation.set_title('Orientation (Euler Angles)', fontsize=12, fontweight='bold')
        self.ax_velocity.set_title('Velocity vs Time', fontsize=12, fontweight='bold')
        self.ax_attitude.set_title('Attitude Visualization', fontsize=12, fontweight='bold')

    def visualize_episode(self, trajectory, orientations, velocities, narrow_gap, goal_position, step_interval=5):
        """
        可视化整个episode的飞行数据

        Parameters:
        trajectory: 位置轨迹数组 [n_steps, 3]
        orientations: 姿态角数组 [n_steps, 3] (roll, pitch, yaw in radians)
        velocities: 速度数组 [n_steps, 3]
        narrow_gap: NarrowGap对象（新版定义，仅包含缝隙）
        goal_position: 目标位置 [3]
        step_interval: 每隔多少步显示一个姿态箭头
        """
        n_steps = len(trajectory)
        time_steps = np.arange(n_steps)

        # 转换姿态角为度
        orientations_deg = np.degrees(orientations)

        # 1. 绘制3D轨迹
        self._plot_3d_trajectory(trajectory, orientations, narrow_gap, goal_position,
                                 step_interval)

        # 2. 绘制位置随时间变化
        self._plot_position_vs_time(time_steps, trajectory)

        # 3. 绘制姿态角随时间变化
        self._plot_orientation_vs_time(time_steps, orientations_deg)

        # 4. 绘制速度随时间变化
        self._plot_velocity_vs_time(time_steps, velocities)

        # 5. 绘制姿态示意图
        self._plot_attitude_diagram(trajectory, orientations)

        plt.tight_layout()
        plt.show()

    def _plot_3d_trajectory(self, trajectory, orientations, narrow_gap, goal_position, step_interval):
        """绘制3D轨迹图，适配新版NarrowGap定义"""
        # 绘制轨迹
        self.ax_3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                        'b-', alpha=0.6, linewidth=2, label='Trajectory')

        # 绘制轨迹点（颜色表示时间）
        scatter = self.ax_3d.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                                     c=np.arange(len(trajectory)), cmap='viridis',
                                     s=20, alpha=0.8, label='Position')

        # 绘制姿态箭头（每隔step_interval步）
        for i in range(0, len(trajectory), step_interval):
            if i < len(trajectory) and i < len(orientations):
                pos = trajectory[i]
                rot = R.from_euler('xyz', [orientations[i][0], orientations[i][1], orientations[i][2]]).as_matrix()

                # 绘制姿态箭头
                arrow_length = 0.2
                colors = ['red', 'green', 'blue']
                for j, color in enumerate(colors):
                    direction = rot[:, j] * arrow_length
                    self.ax_3d.quiver(pos[0], pos[1], pos[2],
                                      direction[0], direction[1], direction[2],
                                      color=color, arrow_length_ratio=0.2, linewidth=1, alpha=0.7)

        # 绘制缝隙（适配新版NarrowGap，仅绘制缝隙本身的8个角点）
        gap_corners = narrow_gap.get_gap_corners()  # 从NarrowGap对象获取计算好的角点

        # 定义缝隙的12条棱边（连接8个角点形成长方体）
        edges = [
            # 正面四边形（x正方向）
            [0, 1], [1, 2], [2, 3], [3, 0],
            # 背面四边形（x负方向）
            [4, 5], [5, 6], [6, 7], [7, 4],
            # 连接正背面的四条边
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # 绘制缝隙的所有棱边
        for edge in edges:
            p1 = gap_corners[edge[0]]
            p2 = gap_corners[edge[1]]
            self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            'r-', linewidth=3, label='Gap' if edge == edges[0] else "")

        # 绘制目标点
        self.ax_3d.scatter(goal_position[0], goal_position[1], goal_position[2],
                           c='gold', s=200, marker='*', label='Goal')

        # 绘制起始点
        self.ax_3d.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                           c='green', s=100, marker='o', label='Start')

        # 绘制结束点
        self.ax_3d.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                           c='red', s=100, marker='x', label='End')

        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.legend()
        self.ax_3d.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=self.ax_3d, shrink=0.8)
        cbar.set_label('Time Step')

    def _plot_position_vs_time(self, time_steps, trajectory):
        """绘制位置随时间变化"""
        self.ax_position.plot(time_steps, trajectory[:, 0], 'r-', label='X', linewidth=2)
        self.ax_position.plot(time_steps, trajectory[:, 1], 'g-', label='Y', linewidth=2)
        self.ax_position.plot(time_steps, trajectory[:, 2], 'b-', label='Z', linewidth=2)

        self.ax_position.set_xlabel('Time Step')
        self.ax_position.set_ylabel('Position (m)')
        self.ax_position.legend()
        self.ax_position.grid(True, alpha=0.3)

    def _plot_orientation_vs_time(self, time_steps, orientations_deg):
        """绘制姿态角随时间变化"""
        self.ax_orientation.plot(time_steps, orientations_deg[:, 0], 'r-', label='Roll', linewidth=2)
        self.ax_orientation.plot(time_steps, orientations_deg[:, 1], 'g-', label='Pitch', linewidth=2)
        self.ax_orientation.plot(time_steps, orientations_deg[:, 2], 'b-', label='Yaw', linewidth=2)

        self.ax_orientation.set_xlabel('Time Step')
        self.ax_orientation.set_ylabel('Angle (degrees)')
        self.ax_orientation.legend()
        self.ax_orientation.grid(True, alpha=0.3)

    def _plot_velocity_vs_time(self, time_steps, velocities):
        """绘制速度随时间变化"""
        self.ax_velocity.plot(time_steps, velocities[:, 0], 'r-', label='Vx', linewidth=2, alpha=0.8)
        self.ax_velocity.plot(time_steps, velocities[:, 1], 'g-', label='Vy', linewidth=2, alpha=0.8)
        self.ax_velocity.plot(time_steps, velocities[:, 2], 'b-', label='Vz', linewidth=2, alpha=0.8)

        # 计算并绘制速度模长
        speed = np.linalg.norm(velocities, axis=1)
        self.ax_velocity.plot(time_steps, speed, 'k--', label='Total Speed', linewidth=2)

        self.ax_velocity.set_xlabel('Time Step')
        self.ax_velocity.set_ylabel('Velocity (m/s)')
        self.ax_velocity.legend()
        self.ax_velocity.grid(True, alpha=0.3)

    def _plot_attitude_diagram(self, trajectory, orientations):
        """绘制姿态示意图"""
        # 选择几个关键点显示详细姿态
        key_indices = [0, len(trajectory) // 4, len(trajectory) // 2, 3 * len(trajectory) // 4, -1]

        for i, idx in enumerate(key_indices):
            if idx < len(trajectory) and idx < len(orientations):
                roll, pitch, yaw = orientations[idx]

                # 在2D平面上显示姿态
                x_offset = i * 0.3
                y_offset = 0

                # 绘制无人机位置
                self.ax_attitude.scatter(x_offset, y_offset, c='blue', s=100)

                # 绘制姿态箭头
                arrow_length = 0.1

                # 滚转（红色）
                roll_x = arrow_length * np.sin(roll)
                roll_y = arrow_length * np.cos(roll)
                self.ax_attitude.arrow(x_offset, y_offset, roll_x, roll_y,
                                       head_width=0.03, head_length=0.02, fc='red', ec='red')

                # 俯仰（绿色）
                pitch_x = arrow_length * np.sin(pitch)
                pitch_y = arrow_length * np.cos(pitch)
                self.ax_attitude.arrow(x_offset, y_offset, pitch_x, pitch_y,
                                       head_width=0.03, head_length=0.02, fc='green', ec='green')

                # 偏航（蓝色）
                yaw_x = arrow_length * np.sin(yaw)
                yaw_y = arrow_length * np.cos(yaw)
                self.ax_attitude.arrow(x_offset, y_offset, yaw_x, yaw_y,
                                       head_width=0.03, head_length=0.02, fc='blue', ec='blue')

                # 添加时间步标签
                self.ax_attitude.text(x_offset, y_offset - 0.15, f't={idx}',
                                      ha='center', fontsize=8)

        self.ax_attitude.set_xlim(-0.5, 1.5)
        self.ax_attitude.set_ylim(-0.5, 0.5)
        self.ax_attitude.set_aspect('equal')
        self.ax_attitude.grid(True, alpha=0.3)
        self.ax_attitude.set_title('Key Attitudes at Different Time Steps')

        # 添加图例
        self.ax_attitude.plot([], [], 'ro-', label='Roll')
        self.ax_attitude.plot([], [], 'go-', label='Pitch')
        self.ax_attitude.plot([], [], 'bo-', label='Yaw')
        self.ax_attitude.legend()

    def save_plot(self, filename, dpi=300):
        """保存图表"""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"✅ 图表已保存: {filename}")