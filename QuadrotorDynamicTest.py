import numpy as np
import pytest
from QuadrotorDynamics import QuadrotorDynamics


def test_initialization():
    """测试初始化状态是否正确"""
    quad = QuadrotorDynamics()

    # 检查初始状态是否为零
    assert np.allclose(quad.position, np.zeros(3))
    assert np.allclose(quad.orientation, np.zeros(3))
    assert np.allclose(quad.velocity, np.zeros(3))
    assert np.allclose(quad.angular_velocity, np.zeros(3))

    # 检查物理参数是否符合预期
    assert quad.mass == 1.2
    assert np.allclose(quad.inertia, np.diag([0.007, 0.007, 0.014]))
    assert quad.max_motor_speed == 1500


def test_reset():
    """测试状态重置功能"""
    quad = QuadrotorDynamics()
    init_pos = [1.0, 2.0, 3.0]
    init_orient = [0.1, 0.2, 0.3]

    quad.reset(init_pos, init_orient)

    assert np.allclose(quad.position, init_pos)
    assert np.allclose(quad.orientation, init_orient)
    assert np.allclose(quad.velocity, np.zeros(3))  # 重置后速度应为零
    assert np.allclose(quad.angular_velocity, np.zeros(3))  # 重置后角速度应为零


def test_get_obs():
    """测试观测状态输出是否正确"""
    quad = QuadrotorDynamics()
    quad.position = [1, 2, 3]
    quad.velocity = [4, 5, 6]
    quad.orientation = [0.1, 0.2, 0.3]
    quad.angular_velocity = [0.4, 0.5, 0.6]

    obs = quad.get_obs()

    # 检查观测维度是否正确
    assert obs.shape == (12,)
    # 检查观测内容是否正确
    expected = np.concatenate([
        [1, 2, 3],  # 位置
        [4, 5, 6],  # 速度
        [0.1, 0.2, 0.3],  # 姿态
        [0.4, 0.5, 0.6]  # 角速度
    ])
    assert np.allclose(obs, expected.astype(np.float32))


def test_normalized_motor_speeds2motor_speeds():
    """测试归一化转速到真实转速的转换"""
    quad = QuadrotorDynamics()

    # 测试边界值
    assert np.allclose(
        quad.normalized_motor_speeds2motor_speeds([1, 1, 1, 1]),
        [1500, 1500, 1500, 1500]
    )
    assert np.allclose(
        quad.normalized_motor_speeds2motor_speeds([-1, -1, -1, -1]),
        [-1500, -1500, -1500, -1500]
    )

    # 测试中间值
    assert np.allclose(
        quad.normalized_motor_speeds2motor_speeds([0.5, -0.5, 0, 0.3]),
        [750, -750, 0, 450]
    )


def test_update_zero_input():
    """测试零输入时的动力学更新（应仅受重力影响）"""
    quad = QuadrotorDynamics()
    quad.reset([0, 0, 10], [0, 0, 0])  # 初始位置10m高度，零姿态
    initial_z = quad.position[2]

    # 零电机转速输入（无推力）
    quad.update([0, 0, 0, 0], dt=0.1)

    # 检查位置变化（应下落）
    assert quad.position[2] < initial_z  # z坐标减小（下落）
    # 检查水平位置不变
    assert np.allclose(quad.position[:2], [0, 0])
    # 检查姿态不变（无扭矩输入）
    assert np.allclose(quad.orientation, [0, 0, 0])


def test_update_thrust_balance():
    """测试推力平衡时的状态（悬停情况）"""
    quad = QuadrotorDynamics()
    quad.reset([0, 0, 10], [0, 0, 0])

    # 计算平衡重力所需的电机转速（总推力=重力）
    gravity = quad.mass * 9.81
    thrust_per_motor = gravity / 4  # 四个电机平均分配
    motor_speed = np.sqrt(thrust_per_motor / quad.C_T)  # 单个电机所需转速

    # 应用平衡推力
    quad.update([motor_speed, motor_speed, motor_speed, motor_speed], dt=0.1)

    # 检查z方向速度变化很小（近似悬停）
    assert np.isclose(quad.velocity[2], 0, atol=1e-3)
    # 检查姿态不变（无扭矩）
    assert np.allclose(quad.orientation, [0, 0, 0])


def test_orientation_normalization():
    """测试姿态角归一化是否正确"""
    quad = QuadrotorDynamics()

    # 测试超过π的角度
    quad.orientation = [3 * np.pi / 2, 0, 0]
    quad.update([0, 0, 0, 0], dt=0.1)  # 触发姿态更新
    assert np.isclose(quad.orientation[0], -np.pi / 2)  # 3π/2 → -π/2

    # 测试低于-π的角度
    quad.orientation = [-3 * np.pi / 2, 0, 0]
    quad.update([0, 0, 0, 0], dt=0.1)
    assert np.isclose(quad.orientation[0], np.pi / 2)  # -3π/2 → π/2


if __name__ == "__main__":
    pytest.main([__file__])