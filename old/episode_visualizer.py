import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


class EpisodeVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 15), constrained_layout=True)  # 启用constrained_layout
        # 设置constrained_layout的间距参数
        self.setup_layout()
        # 添加颜色映射和归一化器
        self.cmap = plt.get_cmap('viridis')
        self.norm = None

    def setup_layout(self):
        """设置图表布局 - 二行三列布局"""
        # 创建GridSpec，2行3列，调整行高比例
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[4, 5])  # 第一行高度比例小，第二行大
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_position = self.fig.add_subplot(gs[0, 1])
        self.ax_velocity = self.fig.add_subplot(gs[0, 2])
        self.ax_orientation = self.fig.add_subplot(gs[1, :])  # 第二行整行

        # 设置标题
        self.ax_3d.set_title('3D Flight Trajectory', fontsize=12, fontweight='bold')
        self.ax_position.set_title('Position vs Time', fontsize=12, fontweight='bold')
        self.ax_orientation.set_title('Orientation (Euler Angles) vs Time', fontsize=12, fontweight='bold')
        self.ax_velocity.set_title('Velocity vs Time', fontsize=12, fontweight='bold')

    def visualize_episode(self, trajectory, orientations, velocities, narrow_gap, goal_position, step_interval=5):
        """
        可视化整个episode的飞行数据
        """
        n_steps = len(trajectory)
        time_steps = np.arange(n_steps)

        # 初始化颜色归一化器（基于时间步）
        self.norm = Normalize(vmin=0, vmax=n_steps - 1)

        # 转换姿态角为度
        orientations_deg = np.degrees(orientations)

        # 1. 绘制3D轨迹
        self._plot_3d_trajectory(trajectory, orientations, narrow_gap, goal_position,
                                 step_interval)

        # 2. 绘制位置随时间变化
        self._plot_position_vs_time(time_steps, trajectory, narrow_gap)

        # 3. 绘制姿态角随时间变化
        self._plot_orientation_vs_time(time_steps, orientations_deg, narrow_gap)

        # 4. 绘制速度随时间变化
        self._plot_velocity_vs_time(time_steps, velocities)

        # 添加时间步颜色条，放在3D轨迹图附近
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
                            ax=self.ax_3d,  # 仅关联到3D轨迹图
                            shrink=0.8)  # 调整大小以适应3D图
        cbar.set_label('Time Step')

    def _plot_3d_trajectory(self, trajectory, orientations, narrow_gap, goal_position, step_interval):
        """绘制3D轨迹图，适配新版NarrowGap定义"""
        # 绘制轨迹
        for i in range(len(trajectory) - 1):
            # 获取当前线段的颜色（基于时间步）
            color = self.cmap(self.norm(i))
            # 绘制当前线段（连接第i和i+1个点）
            self.ax_3d.plot(trajectory[i:i + 2, 0], trajectory[i:i + 2, 1], trajectory[i:i + 2, 2],
                            color=color, alpha=0.8, linewidth=1.5, label='Trajectory' if i == 0 else "")

        # 计算轨迹在各轴上的范围，用于动态调整箭头长度
        x_range = np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])
        y_range = np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])
        z_range = np.max(trajectory[:, 2]) - np.min(trajectory[:, 2])
        # 取各轴范围的最大值作为参考尺度，箭头长度设为该尺度的5%（可根据需要调整比例）
        max_range = min(x_range, y_range, z_range)
        arrow_length = max_range * 0.05  # 动态计算箭头长度
        # 避免箭头过短或过长，设置上下限
        arrow_length = np.clip(arrow_length, 0.2, 1.0)

        # 绘制姿态箭头（每隔step_interval步）
        for i in range(0, len(trajectory), step_interval):
            if i < len(trajectory) and i < len(orientations):
                pos = trajectory[i]
                rot = R.from_euler('xyz', [orientations[i][0], orientations[i][1], orientations[i][2]]).as_matrix()

                # 绘制姿态箭头
                colors = ['red', 'yellow', 'blue']
                for j, color in enumerate(colors):
                    direction = rot[:, j] * arrow_length
                    self.ax_3d.quiver(pos[0], pos[1], pos[2],
                                      direction[0], direction[1], direction[2],
                                      color=color, arrow_length_ratio=0.2, linewidth=1, alpha=0.7)

        # 绘制缝隙
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
                            'r-', linewidth=1, label='Gap' if edge == edges[0] else "")

        # 绘制目标点
        self.ax_3d.scatter(goal_position[0], goal_position[1], goal_position[2],
                           c='gold', s=20, marker='*', label='Goal')

        # 绘制起始点
        self.ax_3d.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                           c='green', s=20, marker='o', label='Start')

        # 绘制结束点
        self.ax_3d.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                           c='red', s=20, marker='x', label='End')

        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.legend()
        self.ax_3d.grid(True, alpha=0.3)

    def _plot_position_vs_time(self, time_steps, trajectory, narrow_gap):
        """绘制位置随时间变化"""
        self.ax_position.plot(time_steps, trajectory[:, 0], 'r-', label='X', linewidth=2)
        self.ax_position.plot(time_steps, trajectory[:, 1], 'g-', label='Y', linewidth=2)
        self.ax_position.plot(time_steps, trajectory[:, 2], 'b-', label='Z', linewidth=2)

        # 获取narrowgap的中心位置
        gap_center = narrow_gap.center

        # 在各轴上标注缝隙中心位置
        self.ax_position.axhline(y=gap_center[0], color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                                 label='Gap Center X')
        self.ax_position.axhline(y=gap_center[1], color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                                 label='Gap Center Y')
        self.ax_position.axhline(y=gap_center[2], color='blue', linestyle='--', alpha=0.7, linewidth=1.5,
                                 label='Gap Center Z')

        self.ax_position.set_xlabel('Time Step')
        self.ax_position.set_ylabel('Position (m)')
        self.ax_position.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')  # 将图例放在右侧避免重叠
        self.ax_position.grid(True, alpha=0.3)
        self.ax_position.set_title('Position vs Time', fontsize=12, fontweight='bold')

    def _plot_orientation_vs_time(self, time_steps, orientations_deg, narrow_gap):
        """绘制姿态角随时间变化（长图）"""
        self.ax_orientation.plot(time_steps, orientations_deg[:, 0], 'r-', label='Roll', linewidth=2)
        self.ax_orientation.plot(time_steps, orientations_deg[:, 1], 'g-', label='Pitch', linewidth=2)
        self.ax_orientation.plot(time_steps, orientations_deg[:, 2], 'b-', label='Yaw', linewidth=2)

        # 获取narrowgap的倾斜角度（度）
        gap_tilt_deg = np.degrees(narrow_gap.tilt)  # tilt对应pitch
        gap_rotation_deg = np.degrees(narrow_gap.rotation)  # rotation对应roll

        # 标注缝隙的tilt（对应pitch）
        self.ax_orientation.axhline(y=gap_tilt_deg, color='green', linestyle='--',
                                    alpha=0.7, linewidth=2, label=f'Gap Tilt (Pitch): {gap_tilt_deg:.1f}°')

        # 标注缝隙的rotation（对应roll）
        self.ax_orientation.axhline(y=gap_rotation_deg, color='red', linestyle='--',
                                    alpha=0.7, linewidth=2, label=f'Gap Rotation (Roll): {gap_rotation_deg:.1f}°')

        # 添加理想对齐区域（±5度的容差范围）
        self.ax_orientation.axhspan(gap_tilt_deg - 5, gap_tilt_deg + 5, color='green', alpha=0.1,
                                    label='Ideal Pitch Range')
        self.ax_orientation.axhspan(gap_rotation_deg - 5, gap_rotation_deg + 5, color='red', alpha=0.1,
                                    label='Ideal Roll Range')

        self.ax_orientation.set_xlabel('Time Step')
        self.ax_orientation.set_ylabel('Angle (degrees)')
        self.ax_orientation.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')
        self.ax_orientation.grid(True, alpha=0.3)
        self.ax_orientation.set_title('Orientation vs Time', fontsize=12,
                                      fontweight='bold')

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

    def save_plot(self, filename, dpi=300):
        """保存图表"""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"✅ 图表已保存: {filename}")
