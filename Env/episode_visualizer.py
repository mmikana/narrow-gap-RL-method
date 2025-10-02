import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import os
import json

class EpisodeVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 15), constrained_layout=True)
        self.setup_layout()
        self.cmap = plt.get_cmap('viridis')
        self.norm = None

    def setup_layout(self):
        """设置图表布局 - 二行三列布局"""
        gs = GridSpec(3, 3, figure=self.fig, height_ratios=[4, 5, 4])  # 增加一行用于奖励图
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_position = self.fig.add_subplot(gs[0, 1])
        self.ax_velocity = self.fig.add_subplot(gs[0, 2])
        self.ax_orientation = self.fig.add_subplot(gs[1, :])
        self.ax_reward = self.fig.add_subplot(gs[2, :])

        # 设置标题
        self.ax_3d.set_title('3D Flight Trajectory', fontsize=12, fontweight='bold')
        self.ax_position.set_title('Position vs step', fontsize=12, fontweight='bold')
        self.ax_orientation.set_title('Orientation vs step', fontsize=12, fontweight='bold')
        self.ax_velocity.set_title('Velocity vs step', fontsize=12, fontweight='bold')
        self.ax_reward.set_title('Reward vs step', fontsize=12, fontweight='bold')

    def draw_fly_data(self, data_filepath, save_plot_dir=None, step_interval=None):
        """
        核心新增方法：读取本地飞行数据文件，生成可视化并保存图表
        参数说明：
            data_filepath: 本地飞行数据JSON文件路径（如"fly_data_20240520_153000/episode1_reward123.45_goal_achievedTrue.json"）
            save_plot_dir: 图表保存目录（默认与数据文件同目录）
            step_interval: 姿态箭头绘制间隔（可选，默认自动计算）
        返回：
            plot_filepath: 图表保存路径
        """
        # 1. 验证数据文件是否存在
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"飞行数据文件不存在：{data_filepath}")

        # 2. 读取JSON数据并转换为numpy数组（适配visualize_episode要求的格式）
        with open(data_filepath, 'r') as f:
            raw_data = json.load(f)

        # 3. 数据格式转换：列表 → numpy数组（确保维度和类型正确）
        visualized_data = self._convert_raw_data(raw_data)

        # 4. 处理图表保存目录（默认与数据文件同目录，避免手动指定）
        if save_plot_dir is None:
            # 提取数据文件所在目录（如从"a/b/c.json"中提取"a/b"）
            save_plot_dir = os.path.dirname(data_filepath)
        os.makedirs(save_plot_dir, exist_ok=True)  # 确保目录存在

        # 5. 生成图表文件名（基于数据文件名，替换后缀为.png）
        data_filename = os.path.basename(data_filepath)
        plot_filename = os.path.splitext(data_filename)[0] + ".png"
        plot_filepath = os.path.join(save_plot_dir, plot_filename)

        # 6. 传递参数调用原有可视化逻辑
        if step_interval is not None:
            visualized_data['step_interval'] = step_interval

        self.visualize_episode(**visualized_data)

        # 7. 保存图表到本地
        self.save_plot(plot_filepath)

        # 8. 清理当前画布（避免多轮可视化时图像重叠）
        plt.close(self.fig)
        self.fig = plt.figure(figsize=(20, 20), constrained_layout=True)
        self.setup_layout()

        print(f"完整流程完成：数据文件 → 可视化图表\n数据路径：{data_filepath}\n图表路径：{plot_filepath}")
        return plot_filepath

    def visualize_episode(self, **kwargs):
        """
        可视化episode数据，自动适配有无窄缝的环境

        支持的参数:
            trajectory: 无人机位置轨迹 (N, 3)
            orientations: 无人机姿态 (N, 3)
            velocities: 无人机速度 (N, 3)
            narrow_gap: 窄缝对象 (可选，仅含窄缝的环境需要)
            goal_position: 目标位置 (3,)
            step_interval: 姿态箭头绘制间隔 (可选)
        """
        # 提取基础数据并验证
        required_keys = ['trajectory', 'orientations', 'velocities', 'goal_position']
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"缺少必要参数: {key}")

        # 从关键字参数中提取数据
        trajectory = kwargs['trajectory']
        orientations = kwargs['orientations']
        velocities = kwargs['velocities']
        rewards = kwargs['rewards']
        goal_position = kwargs['goal_position']
        narrow_gap = kwargs.get('narrow_gap', None)  # 可选参数，无窄缝环境可为None

        # 计算步长间隔（动态调整）
        n_steps = len(trajectory)
        step_interval = kwargs.get('step_interval', max(1, n_steps // 20))

        # 初始化颜色映射
        self.norm = Normalize(vmin=0, vmax=n_steps - 1)
        time_steps = np.arange(n_steps)
        orientations_deg = np.degrees(orientations)  # 转换为角度

        # 判断是否为窄缝环境
        has_narrow_gap = narrow_gap is not None

        # 绘制所有子图
        self._plot_3d_trajectory(
            trajectory, orientations, goal_position, narrow_gap,
            has_narrow_gap, step_interval
        )
        self._plot_position_vs_time(
            time_steps, trajectory, goal_position, narrow_gap, has_narrow_gap
        )
        self._plot_orientation_vs_time(
            time_steps, orientations_deg, narrow_gap, has_narrow_gap
        )
        self._plot_velocity_vs_time(time_steps, velocities)
        self._plot_reward_vs_time(time_steps, rewards)

        # 添加时间颜色条
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
            ax=self.ax_3d,
            shrink=0.8
        )
        cbar.set_label('Time Step')

    def _plot_3d_trajectory(self, trajectory, orientations, goal_position,
                            narrow_gap, has_narrow_gap, step_interval):
        """绘制3D轨迹，根据是否有窄缝动态调整"""
        # 绘制飞行轨迹
        for i in range(len(trajectory) - 1):
            color = self.cmap(self.norm(i))
            self.ax_3d.plot(
                trajectory[i:i + 2, 0], trajectory[i:i + 2, 1], trajectory[i:i + 2, 2],
                color=color, alpha=0.8, linewidth=1.5, label='Trajectory' if i == 0 else ""
            )

        # 计算姿态箭头长度
        if len(trajectory) > 1:
            x_range = np.ptp(trajectory[:, 0])  # 峰峰值
            y_range = np.ptp(trajectory[:, 1])
            z_range = np.ptp(trajectory[:, 2])
            max_range = max(x_range, y_range, z_range) if (x_range + y_range + z_range) > 0 else 1.0
            arrow_length = np.clip(max_range * 0.05, 0.2, 1.0)
        else:
            arrow_length = 0.5

        # 绘制姿态箭头
        for i in range(0, len(trajectory), step_interval):
            if i < len(orientations):  # 防止索引越界
                pos = trajectory[i]
                # 转换欧拉角到旋转矩阵
                rot = R.from_euler('xyz', orientations[i]).as_matrix()

                # 绘制三个轴的箭头
                axes_colors = ['red', 'green', 'blue']  # X, Y, Z轴
                for j, color in enumerate(axes_colors):
                    direction = rot[:, j] * arrow_length
                    self.ax_3d.quiver(
                        pos[0], pos[1], pos[2],
                        direction[0], direction[1], direction[2],
                        color=color, arrow_length_ratio=0.2, linewidth=1.5, alpha=0.7
                    )

        # 绘制窄缝（如果存在）
        if has_narrow_gap:
            try:
                gap_corners = narrow_gap.get_gap_corners()  # 获取缝隙角点
                # 定义缝隙的边
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # 正面
                    [4, 5], [5, 6], [6, 7], [7, 4],  # 背面
                    [0, 4], [1, 5], [2, 6], [3, 7]  # 连接边
                ]
                for edge in edges:
                    p1 = gap_corners[edge[0]]
                    p2 = gap_corners[edge[1]]
                    self.ax_3d.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        'r-', linewidth=2, label='Narrow Gap' if edge == edges[0] else ""
                    )
            except AttributeError:
                print("警告: 窄缝对象缺少get_gap_corners()方法，无法绘制窄缝")

        # 绘制目标点、起点和终点
        self.ax_3d.scatter(
            goal_position[0], goal_position[1], goal_position[2],
            c='gold', s=100, marker='*', label='Goal Position'
        )
        self.ax_3d.scatter(
            trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
            c='green', s=80, marker='o', label='Start'
        )
        self.ax_3d.scatter(
            trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
            c='red', s=80, marker='x', label='End'
        )

        # 设置3D图属性
        self.ax_3d.set_xlabel('X Position (m)')
        self.ax_3d.set_ylabel('Y Position (m)')
        self.ax_3d.set_zlabel('Z Position (m)')
        self.ax_3d.legend(loc='upper left', fontsize=8)
        self.ax_3d.grid(True, alpha=0.3)

    def _plot_position_vs_time(self, time_steps, trajectory, goal_position,
                               narrow_gap, has_narrow_gap):
        """绘制位置随时间变化"""
        # 绘制位置曲线
        self.ax_position.plot(time_steps, trajectory[:, 0], 'r-', label='X Position', linewidth=2)
        self.ax_position.plot(time_steps, trajectory[:, 1], 'g-', label='Y Position', linewidth=2)
        self.ax_position.plot(time_steps, trajectory[:, 2], 'b-', label='Z Position', linewidth=2)

        # 绘制目标位置参考线
        self.ax_position.axhline(y=goal_position[0], color='red', linestyle='--',
                                 alpha=0.5, linewidth=1.5, label='X Goal')
        self.ax_position.axhline(y=goal_position[1], color='green', linestyle='--',
                                 alpha=0.5, linewidth=1.5, label='Y Goal')
        self.ax_position.axhline(y=goal_position[2], color='blue', linestyle='--',
                                 alpha=0.5, linewidth=1.5, label='Z Goal')

        # 绘制窄缝中心（如果存在）
        if has_narrow_gap:
            try:
                gap_center = narrow_gap.center
                self.ax_position.axhline(y=gap_center[0], color='red', linestyle=':',
                                         alpha=0.3, linewidth=1, label='X Gap Center')
                self.ax_position.axhline(y=gap_center[1], color='green', linestyle=':',
                                         alpha=0.3, linewidth=1, label='Y Gap Center')
                self.ax_position.axhline(y=gap_center[2], color='blue', linestyle=':',
                                         alpha=0.3, linewidth=1, label='Z Gap Center')
            except AttributeError:
                print("警告: 窄缝对象缺少center属性，无法绘制窄缝中心参考线")

        # 设置坐标轴属性
        self.ax_position.set_xlabel('Time Step')
        self.ax_position.set_ylabel('Position (m)')
        self.ax_position.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        self.ax_position.grid(True, alpha=0.3)

    def _plot_orientation_vs_time(self, time_steps, orientations_deg, narrow_gap, has_narrow_gap):
        """绘制姿态角随时间变化"""
        # 绘制姿态曲线
        self.ax_orientation.plot(time_steps, orientations_deg[:, 0], 'r-', label='Roll', linewidth=2)
        self.ax_orientation.plot(time_steps, orientations_deg[:, 1], 'g-', label='Pitch', linewidth=2)
        self.ax_orientation.plot(time_steps, orientations_deg[:, 2], 'b-', label='Yaw', linewidth=2)

        # 绘制窄缝姿态参考（如果存在）
        if has_narrow_gap:
            try:
                # 假设窄缝有tilt和rotation属性表示其角度
                gap_tilt = np.degrees(narrow_gap.tilt)
                gap_rotation = np.degrees(narrow_gap.rotation)

                self.ax_orientation.axhline(y=gap_tilt, color='green', linestyle='--',
                                            alpha=0.5, linewidth=1.5, label=f'Gap Tilt ({gap_tilt:.1f}°)')
                self.ax_orientation.axhline(y=gap_rotation, color='red', linestyle='--',
                                            alpha=0.5, linewidth=1.5, label=f'Gap Rotation ({gap_rotation:.1f}°)')

                # 理想姿态范围
                self.ax_orientation.axhspan(gap_tilt - 5, gap_tilt + 5, color='green', alpha=0.1)
                self.ax_orientation.axhspan(gap_rotation - 5, gap_rotation + 5, color='red', alpha=0.1)
            except AttributeError:
                print("警告: 窄缝对象缺少姿态相关属性，无法绘制窄缝姿态参考")

        # 设置坐标轴属性
        self.ax_orientation.set_xlabel('Time Step')
        self.ax_orientation.set_ylabel('Angle (degrees)')
        self.ax_orientation.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        self.ax_orientation.grid(True, alpha=0.3)

    def _plot_velocity_vs_time(self, time_steps, velocities):
        """绘制速度随时间变化"""
        # 绘制速度分量
        self.ax_velocity.plot(time_steps, velocities[:, 0], 'r-', label='VX', linewidth=2, alpha=0.8)
        self.ax_velocity.plot(time_steps, velocities[:, 1], 'g-', label='VY', linewidth=2, alpha=0.8)
        self.ax_velocity.plot(time_steps, velocities[:, 2], 'b-', label='VZ', linewidth=2, alpha=0.8)

        # 绘制合速度
        speed = np.linalg.norm(velocities, axis=1)
        self.ax_velocity.plot(time_steps, speed, 'k--', label='Speed Magnitude', linewidth=2)

        # 设置坐标轴属性
        self.ax_velocity.set_xlabel('Time Step')
        self.ax_velocity.set_ylabel('Velocity (m/s)')
        self.ax_velocity.legend(loc='upper right', fontsize=8)
        self.ax_velocity.grid(True, alpha=0.3)

    def _plot_reward_vs_time(self, time_steps, rewards):
        """绘制奖励随时间变化曲线"""
        # 绘制每步奖励曲线
        self.ax_reward.plot(time_steps, rewards, 'purple', linewidth=2, label='Step Reward')

        # 计算并绘制累积奖励曲线
        cumulative_rewards = np.cumsum(rewards)
        self.ax_reward.plot(time_steps, cumulative_rewards, 'orange', linestyle='--',
                            linewidth=2, label='Cumulative Reward')

        # 绘制奖励平均值参考线
        mean_reward = np.mean(rewards)
        self.ax_reward.axhline(y=mean_reward, color='gray', linestyle=':',
                               alpha=0.7, label=f'Mean Reward: {mean_reward:.2f}')

        # 设置坐标轴属性
        self.ax_reward.set_xlabel('Time Step')
        self.ax_reward.set_ylabel('Reward Value')
        self.ax_reward.legend(loc='upper right', fontsize=8)
        self.ax_reward.grid(True, alpha=0.3)

    def _convert_raw_data(self, raw_data):
        """
        辅助方法：将JSON读取的原始列表数据转换为visualize_episode所需的numpy数组格式
        处理逻辑：确保轨迹、姿态、速度等核心数据为(N,3)数组，奖励为(N,)数组
        """
        # 核心数据转换（必选字段）
        converted = {
            # 轨迹：列表 → (N,3) numpy数组
            'trajectory': np.array(raw_data['trajectory'], dtype=np.float64),
            # 姿态：列表 → (N,3) numpy数组（原始为弧度，适配后续角度转换）
            'orientations': np.array(raw_data['orientations'], dtype=np.float64),
            # 速度：列表 → (N,3) numpy数组
            'velocities': np.array(raw_data['velocities'], dtype=np.float64),
            # 奖励：列表 → (N,) numpy数组（兼容单步奖励和累积奖励绘制）
            'rewards': np.array(raw_data['rewards'], dtype=np.float64),
            # 目标位置：列表 → (3,) numpy数组
            'goal_position': np.array(raw_data['goal_position'], dtype=np.float64)
        }

        # 调试信息：打印数组形状以确认维度
        print(f"轨迹数据形状: {converted['trajectory'].shape}")
        print(f"姿态数据形状: {converted['orientations'].shape}")
        print(f"速度数据形状: {converted['velocities'].shape}")
        print(f"奖励数据形状: {converted['rewards'].shape}")
        print(f"目标位置形状: {converted['goal_position'].shape}")

        # 验证核心数据维度（避免可视化时索引越界）
        n_steps = len(converted['trajectory'])
        assert len(converted['orientations']) == n_steps, "姿态数据与轨迹步数不匹配"
        assert len(converted['velocities']) == n_steps, "速度数据与轨迹步数不匹配"
        assert len(converted['rewards']) == n_steps, "奖励数据与轨迹步数不匹配"
        assert converted['goal_position'].shape == (3,), "目标位置维度应为(3,)"

        return converted

    def save_plot(self, filename, dpi=300):
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {filename}")

    def show(self):
        """显示图表"""
        plt.show()
