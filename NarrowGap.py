"""
2025/08/19
Narrow Gap的定义方法：center坐标[X,X,X]，默认wall和gap的中心重合；wall的length、height、thickness
相对于水平面的倾斜角度tilt；gap的length、width；wall和center同时绕center的旋转角度rotation（为了方便sat检测）；默认xy为水平面；
关于角度：接口处应该是角度，在进行实际运算的时候转成弧度；
2025/08/28
去除关于墙的定义，Narrow Gap的定义方法：center坐标[x,y,z]，gap的length、height、thickness，length<>y thickness<>x height<>z
相对于xy面的倾斜角度tilt；绕center的旋转角度rotation
"""
from scipy.spatial.transform import Rotation as R
import numpy as np


class NarrowGap:
    def __init__(self,
                 center=(0.0, 0.0, 0.0),  # 缝隙的中心点
                 gap_length=0.7,  # 缝隙在y方向的长度
                 gap_height=0.36,  # 缝隙在z方向的高度
                 gap_thickness=0.1,  # 缝隙在x方向的厚度
                 tilt=30,  # 缝隙绕y轴相对于xy面的倾斜角（度）
                 rotation=30  # 缝隙绕中心的旋转角度（度）
                 ):
        # 转换为float64并标准化单位（度→弧度）
        self.center = np.array(center, dtype=np.float64)
        self.gap_length = np.float64(gap_length)
        self.gap_height = np.float64(gap_height)
        self.gap_thickness = np.float64(gap_thickness)

        self.tilt = np.radians(np.float64(tilt))  # 转为弧度
        self.rotation = np.radians(np.float64(rotation))  # 转为弧度

        # 计算半尺寸
        self.gap_half_length = self.gap_length / 2
        self.gap_half_height = self.gap_height / 2
        self.gap_half_thickness = self.gap_thickness / 2

        # 计算缝隙的局部坐标系（包含旋转）
        self._compute_local_frame()

        # 预计算角点
        self.gap_corners = self._get_gap_corners()

    def _compute_local_frame(self):
        """计算缝隙的局部坐标系，动态生成法向量"""
        # 1. 初始基向量
        initial_x = np.array([1, 0, 0], dtype=np.float64)  # 对应thickness方向
        initial_y = np.array([0, 1, 0], dtype=np.float64)  # 对应length方向
        initial_z = np.array([0, 0, 1], dtype=np.float64)  # 对应height方向

        # 2. 应用tilt（绕y轴旋转，相对于xy面的倾斜）
        tilt_rot = R.from_euler('y', self.tilt)
        rotated_x = tilt_rot.apply(initial_x)
        rotated_y = tilt_rot.apply(initial_y)
        rotated_z = tilt_rot.apply(initial_z)

        # 3. 应用rotation绕中心旋转
        rot = R.from_euler('x', self.rotation)
        self.gap_x = rot.apply(rotated_x)  # x方向（thickness）
        self.gap_y = rot.apply(rotated_y)  # y方向（length）
        self.gap_z = rot.apply(rotated_z)  # z方向（height）

    def _get_gap_corners(self):
        """计算缝隙的8个角点（考虑厚度方向）"""
        # 正面（x正方向）4个角点
        front = np.array([
            self.center + self.gap_y * (-self.gap_half_length) + self.gap_z * (-self.gap_half_height) + self.gap_x * (
                self.gap_half_thickness),
            self.center + self.gap_y * self.gap_half_length + self.gap_z * (-self.gap_half_height) + self.gap_x * (
                self.gap_half_thickness),
            self.center + self.gap_y * self.gap_half_length + self.gap_z * self.gap_half_height + self.gap_x * (
                self.gap_half_thickness),
            self.center + self.gap_y * (-self.gap_half_length) + self.gap_z * self.gap_half_height + self.gap_x * (
                self.gap_half_thickness),
        ], dtype=np.float64)

        # 背面（x负方向）4个角点
        back = np.array([
            self.center + self.gap_y * (-self.gap_half_length) + self.gap_z * (-self.gap_half_height) - self.gap_x * (
                self.gap_half_thickness),
            self.center + self.gap_y * self.gap_half_length + self.gap_z * (-self.gap_half_height) - self.gap_x * (
                self.gap_half_thickness),
            self.center + self.gap_y * self.gap_half_length + self.gap_z * self.gap_half_height - self.gap_x * (
                self.gap_half_thickness),
            self.center + self.gap_y * (-self.gap_half_length) + self.gap_z * self.gap_half_height - self.gap_x * (
                self.gap_half_thickness),
        ], dtype=np.float64)

        return np.concatenate([front, back], axis=0)

    def get_gap_corners(self):
        """返回缝隙的8个角点（用于可视化和碰撞检测）"""
        return self.gap_corners.copy()
