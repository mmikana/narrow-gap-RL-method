'''
2025/08/19
Narrow Gap的定义方法：center坐标[X,X,X]，默认wall和gap的中心重合；wall的length、height、thickness
相对于水平面的倾斜角度tilt；gap的length、width；wall和center同时绕center的旋转角度rotation（为了方便sat检测）；默认xy为水平面；
关于角度：接口处应该是角度，在进行实际运算的时候转成弧度；
'''
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class NarrowGap:
    def __init__(self,
                 center=(0.0, 0.0, 0.0),  # 墙和缝隙的共同中心点
                 wall_length=2.0,  # 墙在y方向的长度,对应gap_length
                 wall_height=1.5,  # 墙在z方向的高度，对应gap_height
                 wall_thickness=0.1,  # 墙在x方向的厚度
                 wall_tilt=0.0,  # 墙面绕y轴倾斜角（度）
                 gap_length=0.7,  # 缝隙在y方向的长度，对应uav的长宽0.47
                 gap_height=0.36,  # 缝隙在z方向的高度，对应uav的厚度0.23
                 rotation=0.0  # 缝隙center旋转的角度，从
                 ):
        # 转换为float64并标准化单位（度→弧度）
        self.center = np.array(center, dtype=np.float64)
        self.wall_length = np.float64(wall_length)
        self.wall_height = np.float64(wall_height)
        self.wall_thickness = np.float64(wall_thickness)

        self.wall_tilt = np.radians(np.float64(wall_tilt))  # 转为弧度
        self.gap_length = np.float64(gap_length)
        self.gap_height = np.float64(gap_height)
        self.rotation = np.radians(np.float64(rotation))  # 转为弧度

        # 计算半尺寸
        self.wall_half_length = self.wall_length / 2
        self.wall_half_height = self.wall_height / 2
        self.wall_half_thickness = self.wall_thickness / 2
        self.gap_half_length = self.gap_length / 2
        self.gap_half_height = self.gap_height / 2

        # 核心：计算墙面和缝隙的局部坐标系（包含旋转）
        self._compute_local_frames()

        # 预计算角点
        self.wall_front_corners = self._get_wall_front_corners()
        self.gap_front_corners = self._get_gap_front_corners()

    def _compute_local_frames(self):
        """计算墙面和缝隙的局部坐标系，动态生成法向量"""
        # 1. 初始基向量
        initial_x = np.array([1, 0, 0], dtype=np.float64)
        initial_y = np.array([0, 1, 0], dtype=np.float64)  # 初始法向量方向
        initial_z = np.array([0, 0, 1], dtype=np.float64)

        # 2. 应用wall_tilt（绕y轴旋转）
        tilt_rot = R.from_euler('y', self.wall_tilt)
        rotated_x = tilt_rot.apply(initial_x)
        rotated_y = tilt_rot.apply(initial_y)
        rotated_z = tilt_rot.apply(initial_z)

        # 3. 应用rotation绕在yz面绕中心旋转
        gap_rot = R.from_euler('x', self.rotation)
        self.wall_x = gap_rot.apply(rotated_x)
        self.wall_y = gap_rot.apply(rotated_y)  # 法向量受旋转影响
        self.wall_z = gap_rot.apply(rotated_z)

        # 4. 缝隙局部坐标系与墙面一致
        self.gap_x = self.wall_x.copy()
        self.gap_z = self.wall_z.copy()

        # 5. 计算缝隙法向量（动态生成，受两个旋转参数影响）
        self.normal = self.wall_x.copy()  # 现在是经过两次旋转后的法向量

    def _get_wall_front_corners(self):
        """计算墙面正面（y正方向）的4个角点"""
        return np.array([
            # 左下（x负，z负）
            self.center + self.wall_x * (-self.wall_half_length) + self.wall_z * (-self.wall_half_height),
            # 右下（x正，z负）
            self.center + self.wall_x * (self.wall_half_length) + self.wall_z * (-self.wall_half_height),
            # 右上（x正，z正）
            self.center + self.wall_x * (self.wall_half_length) + self.wall_z * (self.wall_half_height),
            # 左上（x负，z正）
            self.center + self.wall_x * (-self.wall_half_length) + self.wall_z * (self.wall_half_height),
        ], dtype=np.float64)

    def _get_gap_front_corners(self):
        """计算缝隙正面的4个角点"""
        return np.array([
            # 缝隙左下
            self.center + self.gap_x * (-self.gap_half_length) + self.gap_z * (-self.gap_half_height),
            # 缝隙右下
            self.center + self.gap_x * (self.gap_half_length) + self.gap_z * (-self.gap_half_height),
            # 缝隙右上
            self.center + self.gap_x * (self.gap_half_length) + self.gap_z * (self.gap_half_height),
            # 缝隙左上
            self.center + self.gap_x * (-self.gap_half_length) + self.gap_z * (self.gap_half_height),
        ], dtype=np.float64)

    def get_wall_corners(self):
        """返回墙面正面4个角点（用于可视化）"""
        return self.wall_front_corners.copy()

    def get_gap_corners(self):
        """返回缝隙正面4个角点（用于可视化）"""
        return self.gap_front_corners.copy()