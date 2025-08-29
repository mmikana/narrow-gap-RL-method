import numpy as np


class CollisionDetector:
    """四旋翼与矩形通道碰撞检测系统（修复左壁误判问题）"""

    def __init__(self):
        self.epsilon = 1e-4  # 增大容差减少浮点误差影响
        self.safety_margin = 0.01  # 安全余量，避免边界误判

    def check_channel_collision(self, quadrotor, gap):
        quad_vertices = quadrotor.get_vertices()

        # 重新定义四个平面（修正左壁法线方向和判断逻辑）
        planes = [
            # 上平面（天花板）
            {
                'normal': -gap.gap_z,
                'point': gap.center + gap.gap_z * gap.gap_half_height,
                'type': 'ceiling',
                'inside_side': lambda d: d > self.safety_margin  # 明确侵入内部才判定
            },
            # 下平面（地板）
            {
                'normal': gap.gap_z,
                'point': gap.center - gap.gap_z * gap.gap_half_height,
                'type': 'floor',
                'inside_side': lambda d: d > self.safety_margin
            },
            # 左平面（左侧壁）- 关键修正：法线方向和侵入判断
            {
                'normal': gap.gap_y,  # 修正法线方向（原方向可能反了）
                'point': gap.center + gap.gap_y * gap.gap_half_length,
                'type': 'left_wall',
                'inside_side': lambda d: d < -self.safety_margin  # 左壁内侧在负方向
            },
            # 右平面（右侧壁）
            {
                'normal': -gap.gap_y,  # 同步修正右壁法线
                'point': gap.center - gap.gap_y * gap.gap_half_length,
                'type': 'right_wall',
                'inside_side': lambda d: d < -self.safety_margin
            }
        ]

        for plane in planes:
            if self._check_plane_collision(quad_vertices, plane):
                return True  # 碰撞返回True

        return False  # 未碰撞返回False

    def _check_plane_collision(self, vertices, plane):
        """改进的平面碰撞检测：严格判断顶点是否侵入通道内部"""
        normal = plane['normal']
        plane_point = plane['point']
        inside_side = plane['inside_side']

        # 计算所有顶点到平面的带符号距离
        signed_distances = [np.dot(vertex - plane_point, normal) for vertex in vertices]

        # 关键修复：只有存在顶点明确侵入通道内侧时才继续检测
        if not any(inside_side(d) for d in signed_distances):
            return False

        # SAT检测：检查所有可能的分离轴
        axes = self._get_sat_axes_for_plane(vertices, normal)
        for axis in axes:
            if not self._overlap_on_axis(vertices, plane_point, axis, normal):
                return False

        return True

    def _get_sat_axes_for_plane(self, vertices, plane_normal):
        """生成更全面的分离轴，确保倾斜状态下的检测准确性"""
        axes = [plane_normal]

        if len(vertices) >= 8:
            # 重新计算四旋翼的边向量（与get_vertices定义保持一致）
            edges = [
                vertices[1] - vertices[0],  # z方向边
                vertices[2] - vertices[0],  # y方向边
                vertices[4] - vertices[0],  # x方向边
                vertices[3] - vertices[2],  # z方向边
                vertices[5] - vertices[4],  # z方向边
                vertices[6] - vertices[4]  # y方向边
            ]

            # 添加边向量作为分离轴
            for edge in edges:
                edge_norm = np.linalg.norm(edge)
                if edge_norm > self.epsilon:
                    axes.append(edge / edge_norm)

            # 添加法线与边向量的叉乘轴（处理倾斜情况）
            for edge in edges:
                edge_norm = np.linalg.norm(edge)
                if edge_norm > self.epsilon:
                    normalized_edge = edge / edge_norm
                    cross_axis = np.cross(plane_normal, normalized_edge)
                    cross_norm = np.linalg.norm(cross_axis)
                    if cross_norm > self.epsilon:
                        axes.append(cross_axis / cross_norm)

        return axes

    def _overlap_on_axis(self, vertices, plane_point, axis, plane_normal):
        """改进投影重叠判断，增加方向补偿"""
        # 四旋翼顶点投影
        quad_proj = [np.dot(vertex, axis) for vertex in vertices]
        quad_min, quad_max = min(quad_proj), max(quad_proj)

        # 平面投影
        plane_proj = np.dot(plane_point, axis)

        # 根据平面法线与轴的夹角动态调整容差
        dot_product = np.dot(axis, plane_normal)
        tolerance = self.epsilon / abs(dot_product) if abs(dot_product) > self.epsilon else self.epsilon

        # 检查投影是否重叠（增加方向判断）
        if dot_product > 0:
            # 平面法线与轴同向：平面外侧为投影小于平面位置
            return not (quad_max < plane_proj - tolerance)
        else:
            # 平面法线与轴反向：平面外侧为投影大于平面位置
            return not (quad_min > plane_proj + tolerance)

    def simple_distance_check(self, quadrotor, gap):
        """快速距离检查，过滤明显无碰撞的情况"""
        quad_position = quadrotor.position
        rel_pos = quad_position - gap.center

        # 计算四旋翼在缝隙局部坐标系中的位置
        local_y = np.dot(rel_pos, gap.gap_y)
        local_z = np.dot(rel_pos, gap.gap_z)

        # 考虑四旋翼自身尺寸的边界检查
        quad_half_y = quadrotor.size[1] / 2  # 四旋翼y方向半尺寸
        quad_half_z = quadrotor.size[2] / 2  # 四旋翼z方向半尺寸

        # 只有当四旋翼可能进入缝隙范围时才进行详细检测
        if (abs(local_y) > gap.gap_half_length + quad_half_y + self.safety_margin or
                abs(local_z) > gap.gap_half_height + quad_half_z + self.safety_margin):
            return False
        return True

    def efficient_collision_check(self, quadrotor, gap):
        if not self.simple_distance_check(quadrotor, gap):
            return False  # 未碰撞返回False
        return self.check_channel_collision(quadrotor, gap)  # 返回碰撞检测结果（True/False）