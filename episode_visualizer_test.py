import numpy as np
import matplotlib.pyplot as plt
from episode_visualizer import EpisodeVisualizer
from NarrowGap import NarrowGap


def generate_test_data():
    """生成测试用的飞行数据"""
    # 创建模拟轨迹（螺旋上升）
    n_steps = 100
    time = np.linspace(0, 4 * np.pi, n_steps)

    # 轨迹：螺旋线
    trajectory = np.zeros((n_steps, 3))
    trajectory[:, 0] = 0.5 * np.sin(time)  # X: 正弦波
    trajectory[:, 1] = 0.5 * np.cos(time)  # Y: 余弦波
    trajectory[:, 2] = 0.1 * time + 0.5  # Z: 线性上升

    # 姿态：随时间变化的欧拉角
    orientations = np.zeros((n_steps, 3))
    orientations[:, 0] = 0.2 * np.sin(time)  # Roll: 小幅度摆动
    orientations[:, 1] = 0.3 * np.cos(0.5 * time)  # Pitch: 中等幅度摆动
    orientations[:, 2] = time  # Yaw: 持续旋转

    # 速度：轨迹的导数
    velocities = np.zeros((n_steps, 3))
    velocities[:, 0] = 0.5 * np.cos(time)  # Vx
    velocities[:, 1] = -0.5 * np.sin(time)  # Vy
    velocities[:, 2] = 0.1 * np.ones(n_steps)  # Vz

    return trajectory, orientations, velocities


def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试 EpisodeVisualizer 基本功能...")

    # 生成测试数据
    trajectory, orientations, velocities = generate_test_data()

    # 创建缝隙环境
    narrow_gap = NarrowGap(
        center=(0, 0, 1.5),
        wall_length=2.0,
        wall_height=2.0,
        wall_thickness=0.1,
        wall_tilt=20,  # 20度倾斜
        gap_length=0.7,
        gap_height=0.36,
        rotation=0
    )

    # 目标位置（缝隙后方）
    goal_position = np.array([0, 0, 2.0])

    # 创建可视化器
    visualizer = EpisodeVisualizer()

    print("📊 生成可视化图表...")

    # 测试可视化
    try:
        visualizer.visualize_episode(
            trajectory=trajectory,
            orientations=orientations,
            velocities=velocities,
            narrow_gap=narrow_gap,
            goal_position=goal_position,
            step_interval=10  # 每10步显示一个姿态箭头
        )
        print("✅ 可视化成功完成！")

        # 测试保存功能
        visualizer.save_plot("test_episode_visualization.png")
        print("✅ 图表保存成功！")

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()

    return True


def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 测试边界情况...")

    # 测试空数据
    try:
        visualizer = EpisodeVisualizer()
        narrow_gap = NarrowGap()

        # 空数据测试
        empty_traj = np.array([]).reshape(0, 3)
        empty_ori = np.array([]).reshape(0, 3)
        empty_vel = np.array([]).reshape(0, 3)

        visualizer.visualize_episode(
            empty_traj, empty_ori, empty_vel, narrow_gap, np.array([0, 0, 1])
        )
        print("❌ 空数据测试应该抛出异常")
    except:
        print("✅ 空数据正确处理")

    # 测试单点数据
    try:
        single_traj = np.array([[0, 0, 1]])
        single_ori = np.array([[0, 0, 0]])
        single_vel = np.array([[0, 0, 0]])

        visualizer = EpisodeVisualizer()
        narrow_gap = NarrowGap()

        visualizer.visualize_episode(
            single_traj, single_ori, single_vel, narrow_gap, np.array([0, 0, 1])
        )
        print("✅ 单点数据测试通过")
    except Exception as e:
        print(f"❌ 单点数据测试失败: {e}")


def test_different_step_intervals():
    """测试不同的步间隔"""
    print("\n🧪 测试不同的步间隔...")

    trajectory, orientations, velocities = generate_test_data()
    narrow_gap = NarrowGap()

    # 测试不同的步间隔
    intervals = [1, 5, 10, 20]

    for interval in intervals:
        try:
            visualizer = EpisodeVisualizer()
            visualizer.visualize_episode(
                trajectory, orientations, velocities, narrow_gap,
                np.array([0, 0, 2]), step_interval=interval
            )
            plt.close()  # 关闭图表避免重叠
            print(f"✅ 步间隔 {interval} 测试通过")
        except Exception as e:
            print(f"❌ 步间隔 {interval} 测试失败: {e}")


def test_orientation_conversion():
    """测试姿态角转换"""
    print("\n🧪 测试姿态角转换...")

    # 测试弧度到度的转换
    orientations_rad = np.array([
        [np.pi / 4, np.pi / 6, np.pi / 2],  # 45°, 30°, 90°
        [np.pi / 2, 0, -np.pi / 4],  # 90°, 0°, -45°
        [0, np.pi / 3, np.pi]  # 0°, 60°, 180°
    ])

    orientations_deg = np.degrees(orientations_rad)

    print("弧度角度转换测试:")
    for i, (rad, deg) in enumerate(zip(orientations_rad, orientations_deg)):
        print(f"  第{i}组: {rad} → {deg}")

    # 验证转换正确性
    expected = np.array([[45, 30, 90], [90, 0, -45], [0, 60, 180]])
    assert np.allclose(orientations_deg, expected, atol=1e-6), "角度转换错误"
    print("✅ 姿态角转换测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("EpisodeVisualizer 测试套件")
    print("=" * 50)

    # 运行所有测试
    test_basic_functionality()
    test_edge_cases()
    test_different_step_intervals()
    test_orientation_conversion()

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)