import numpy as np
from scipy.spatial.transform import Rotation as R
from QuadrotorDynamics import QuadrotorDynamics
from NarrowGap import NarrowGap


class Box3D:
    """三维长方体类，用于表示UAV、缺口和墙体边框"""

    def __init__(self, center, size, orientation):
        """
        初始化三维长方体
        :param center: 中心坐标 (x, y, z)
        :param size: 尺寸 (length, width, depth)
        :param orientation: 欧拉角 (roll, pitch, yaw)，单位弧度，顺序为X→Y→Z轴
        """
        self.center = np.array(center, dtype=np.float64)
        self.size = np.array(size, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)

    def __repr__(self):
        return f"Box3D(center={self.center}, size={self.size}, orientation={self.orientation})"


def euler_to_rotation(euler):
    """将欧拉角转换为旋转矩阵"""
    return R.from_euler('xyz', euler).as_matrix()


def decompose_wall(ng: NarrowGap) -> list:
    """
    根据NarrowGap参数，拆分出4个实体边框
    :param ng: NarrowGap对象
    :return: 4个边框的Box3D对象列表 [上, 下, 左, 右]
    """
    # 从NarrowGap获取尺寸参数
    wall_length = ng.wall_length
    wall_height = ng.wall_height
    wall_thickness = ng.wall_thickness
    gap_length = ng.gap_length
    gap_height = ng.gap_height

    # 验证墙体尺寸是否大于缺口尺寸
    if wall_length <= gap_length or wall_height <= gap_height:
        raise ValueError("墙体尺寸必须大于缺口尺寸")

    # 计算尺寸差
    delta_length = wall_length - gap_length  # 长度方向总间隙
    delta_width = wall_height - gap_height  # 宽度方向总间隙

    # 边框尺寸
    top_bottom_size = [gap_length, delta_width / 2, wall_thickness]  # 上下边框
    left_right_size = [delta_length / 2, gap_height, wall_thickness]  # 左右边框

    # 获取旋转矩阵
    rotation = euler_to_rotation([0, ng.wall_tilt, ng.rotation])

    # 计算各边框的中心位置
    top_center = ng.center + rotation @ np.array([0, gap_height / 2 + delta_width / 4, 0])
    bottom_center = ng.center + rotation @ np.array([0, -gap_height / 2 - delta_width / 4, 0])
    left_center = ng.center + rotation @ np.array([-gap_length / 2 - delta_length / 4, 0, 0])
    right_center = ng.center + rotation @ np.array([gap_length / 2 + delta_length / 4, 0, 0])

    # 创建并返回4个边框
    return [
        Box3D(top_center, top_bottom_size, [0, ng.wall_tilt, ng.rotation]),
        Box3D(bottom_center, top_bottom_size, [0, ng.wall_tilt, ng.rotation]),
        Box3D(left_center, left_right_size, [0, ng.wall_tilt, ng.rotation]),
        Box3D(right_center, left_right_size, [0, ng.wall_tilt, ng.rotation])
    ]


def get_rotated_vertices(box):
    """计算旋转后的长方体顶点坐标"""
    half_size = box.size / 2
    local_vertices = np.array([
        [half_size[0], half_size[1], half_size[2]],
        [half_size[0], half_size[1], -half_size[2]],
        [half_size[0], -half_size[1], half_size[2]],
        [half_size[0], -half_size[1], -half_size[2]],
        [-half_size[0], half_size[1], half_size[2]],
        [-half_size[0], half_size[1], -half_size[2]],
        [-half_size[0], -half_size[1], half_size[2]],
        [-half_size[0], -half_size[1], -half_size[2]]
    ])
    rotation = euler_to_rotation(box.orientation)
    return np.array([rotation @ v + box.center for v in local_vertices])


def get_face_normals(vertices):
    """从顶点获取长方体的6个面法线"""
    faces = [
        [0, 1, 3, 2], [4, 5, 7, 6],  # 前后面
        [0, 1, 5, 4], [2, 3, 7, 6],  # 上下
        [0, 2, 6, 4], [1, 3, 7, 5]  # 左右
    ]
    normals = []
    for face in faces:
        v1 = vertices[face[1]] - vertices[face[0]]
        v2 = vertices[face[2]] - vertices[face[0]]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normals.append(normal / norm)
    return normals

def get_edge_vectors(vertices):
    """从顶点获取长方体的12条边向量"""
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # 前面
        [4, 5], [5, 7], [7, 6], [6, 4],  # 后面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接边
    ]
    return [vertices[e[1]] - vertices[e[0]] for e in edges]


def get_edge_cross_axes(verticesA, verticesB):
    """计算两个长方体边向量的叉积作为潜在分离轴"""
    edgesA = get_edge_vectors(verticesA)
    edgesB = get_edge_vectors(verticesB)
    cross_axes = []
    for edgeA in edgesA:
        for edgeB in edgesB:
            cross = np.cross(edgeA, edgeB)
            norm = np.linalg.norm(cross)
            if norm > 1e-10:
                cross_axes.append(cross / norm)
    return cross_axes


def project_vertices(vertices, axis):
    """将顶点投影到指定轴上，返回投影的最小值和最大值"""
    projections = [np.dot(v, axis) for v in vertices]
    return min(projections), max(projections)


def sat_collision(boxA, boxB):
    """使用分离轴定理检测两个长方体是否碰撞"""
    verticesA = get_rotated_vertices(boxA)
    verticesB = get_rotated_vertices(boxB)

    axes = []
    axes.extend(get_face_normals(verticesA))
    axes.extend(get_face_normals(verticesB))
    axes.extend(get_edge_cross_axes(verticesA, verticesB))

    for axis in axes:
        minA, maxA = project_vertices(verticesA, axis)
        minB, maxB = project_vertices(verticesB, axis)
        if maxA < minB - 1e-6 or maxB < minA - 1e-6:
            return False
    return True


def uav_to_box3d(uav: QuadrotorDynamics) -> Box3D:
    """将QuadrotorDynamics对象转换为Box3D对象"""
    return Box3D(
        center=uav.position,
        size=uav.size,
        orientation=uav.orientation
    )


def check_collision(uav: QuadrotorDynamics, ng: NarrowGap) -> bool:
    """
    检查UAV是否与墙上的缺口边框发生碰撞
    :param uav: QuadrotorDynamics实例
    :param ng: NarrowGap实例
    :return: 布尔值，True表示发生碰撞，False表示未发生碰撞
    """
    uav_box = uav_to_box3d(uav)
    borders = decompose_wall(ng)

    for border in borders:
        if sat_collision(uav_box, border):
            return True
    return False


# 测试代码
if __name__ == "__main__":
    # 创建测试环境
    ng = NarrowGap(
        center=(5, 0, 1),
        wall_length=2.0,
        wall_height=2.0,
        wall_thickness=0.1,
        wall_pitch=30,  # 度
        gap_length=0.7,
        gap_height=0.36,
        rotation=0
    )

    # 创建无人机实例
    uav = QuadrotorDynamics()
    uav.reset(position=[0, 0, 1], orientation=[0, 0, 0])

    # 测试1: UAV在初始位置，无碰撞
    print("测试1 - UAV在初始位置:", check_collision(uav, ng))  # 应返回False

    # 测试2: UAV在缝隙中心，无碰撞
    uav.position = np.array([5, 0, 1])
    print("测试2 - UAV在缝隙中心:", check_collision(uav, ng))  # 应返回False

    # 测试3: UAV靠近上边框，应碰撞
    uav.position = np.array([5, 0.5, 1])
    print("测试3 - UAV靠近上边框:", check_collision(uav, ng))  # 应返回True

    # 测试4: UAV靠近右边框，应碰撞
    uav.position = np.array([5.5, 0, 1])
    print("测试4 - UAV靠近右边框:", check_collision(uav, ng))  # 应返回True

    # 测试5: UAV旋转后靠近边框
    uav.position = np.array([4.5, 0, 1])
    uav.orientation = np.array([0, 0, np.pi / 4])  # 45度偏航
    print("测试5 - UAV旋转后靠近左边框:", check_collision(uav, ng))  # 应返回True