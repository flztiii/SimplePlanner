#! /usr/bin/env python
#! -*- coding: utf-8 -*-

from jps import *
import numpy as np
import copy
from scipy.spatial import KDTree



# 进行比较，考虑数值误差
class Compare:
    EPS = 1e-8

    def __init__(self) -> None:
        pass

    @staticmethod
    def large(v1, v2):
        if v1 > v2 and np.abs(v1 - v2) > Compare.EPS:
            return True
        else:
            return False
        
    @staticmethod
    def small(v1, v2):
        if v1 < v2 and np.abs(v1 - v2) > Compare.EPS:
            return True
        else:
            return False
    
    @staticmethod
    def equal(v1, v2):
        if np.abs(v1 - v2) <= Compare.EPS:
            return True
        else:
            return False

# 椭圆
class Ellipse:
    def __init__(self, C: np.array, d: np.array) -> None:
        # 椭圆参数，对称矩阵
        self.C_ = C
        # 椭圆参数，平移相连
        self.d_ = d

    # 判断点是否在椭圆内
    def inside(self, p: np.array, include_bound):
        if include_bound:
            return Compare.small(self.dist(p), 1) or Compare.equal(self.dist(p), 1)
        else:
            return Compare.small(self.dist(p), 1)
    
    # 得到椭圆内的点
    def insidePoints(self, points: list, include_bound=True):
        inside_points = list()
        for p in points:
            if self.inside(p, include_bound):
                inside_points.append(p)
        return inside_points
    
    # 得到与椭圆相对距离最近的点
    def closestPoint(self, points: list):
        min_dist = float("inf")
        closest_point = None
        for p in points:
            d = self.dist(p)
            if Compare.large(min_dist, d):
                min_dist = d
                closest_point = p
        return closest_point

    # 计算点与椭圆相对距离（非真实距离）
    def dist(self, p: np.array):
        return np.linalg.norm(np.dot(np.linalg.inv(self.C_), (p - self.d_)))

# 超平面
class Hyperplane:
    def __init__(self, n: np.array, d: np.array) -> None:
        # 法向量
        self.n_ = n
        # 平移向量
        self.d_ = d

    # 计算距离
    def signDist(self, p: np.array):
        return np.dot(self.n_, (p - self.d_))
    
# 多边形
class Polygon:
    def __init__(self, hyper_planes: list[Hyperplane]) -> None:
        self.hyper_planes_ = hyper_planes

    # 判断点是否在多边形内
    def inside(self, p: np.array, include_bound = True):
        is_inside = True
        if include_bound:
            for hyper_plane in self.hyper_planes_:
                if Compare.large(hyper_plane.signDist(p), 0):
                    is_inside = False
                    break
        else:
            for hyper_plane in self.hyper_planes_:
                if Compare.large(hyper_plane.signDist(p), 0) or Compare.equal(hyper_plane.signDist(p), 0):
                    is_inside = False
                    break
        return is_inside
    
    # 得到多边形的顶点
    def getVerticals(self):
        # 计算边的交点
        inter_points = self.getInterPoints()
        # 得到在多边形内的交点
        inside_inter_points = list()
        for point in inter_points:
            if self.inside(point):
                inside_inter_points.append(point)
        if len(inside_inter_points) > 0:
            # 进行交点排序
            return self.pointSort(inside_inter_points)
        else:
            return list()

    # 计算边的交点
    def getInterPoints(self):
        inter_points = list()
        # 遍历每两个平面
        for i in range(len(self.hyper_planes_)):
            for j in range(i + 1, len(self.hyper_planes_)):
                plane_1 = self.hyper_planes_[i]
                plane_2 = self.hyper_planes_[j]
                # 判断是否平行
                if Compare.equal(plane_1.n_[0] * plane_2.n_[1], plane_1.n_[1] * plane_2.n_[0]):
                    #平行
                    continue
                else:
                    # 不平行，计算交点
                    inter_x = (plane_2.n_[1] * np.dot(plane_1.n_, plane_1.d_) - plane_1.n_[1] * np.dot(plane_2.n_, plane_2.d_)) / (plane_1.n_[0] * plane_2.n_[1] - plane_1.n_[1] * plane_2.n_[0])
                    inter_y = (plane_2.n_[0] * np.dot(plane_1.n_, plane_1.d_) - plane_1.n_[0] * np.dot(plane_2.n_, plane_2.d_)) / (plane_1.n_[1] * plane_2.n_[0] - plane_1.n_[0] * plane_2.n_[1])
                    inter_points.append(np.array([inter_x, inter_y]))

        return inter_points

    # 进行点排序
    def pointSort(self, points: list[np.array]):
        # 计算这些点的中点
        middle_point = np.array([np.mean([p[0] for p in points]), np.mean([p[1] for p in points])])
        # 计算点与中点的角度
        record = list()
        for point in points:
            orien_v = point - middle_point
            orien = np.arctan2(orien_v[1], orien_v[0])
            record.append((point, orien))
        record = sorted(record, key=lambda x: x[1])
        sorted_point = [x[0] for x in record]
        return sorted_point

# 对空间进行凸分解
class ConvexDecomp:
    # 构造函数
    def __init__(self, consider_range) -> None:
        self.consider_range_ = consider_range
    
    # 根据输入路径对空间进行凸分解
    def decomp(self, line_points: list[np.array], obs_points: list[np.array], visualize=True):
        # 最终结果
        decomp_polygons = list()
        # 构建输入障碍物点的kdtree
        obs_kdtree = KDTree(obs_points)
        # 进行空间分解
        for i in range(len(line_points) - 1):
            # 得到当前线段
            pf, pr = line_points[i], line_points[i + 1]
            # 构建初始多面体
            init_polygon = self.initPolygon(pf, pr)
            # 过滤障碍物点
            candidate_obs_point_indexes = obs_kdtree.query_ball_point((pf + pr) / 2, np.linalg.norm([np.linalg.norm(pr - pf) / 2 + self.consider_range_, self.consider_range_]))
            local_obs_points = list()
            for index in candidate_obs_point_indexes:
                if init_polygon.inside(obs_points[index]):
                    local_obs_points.append(obs_points[index])
            # 得到初始椭圆
            ellipse = self.findEllipse(pf, pr, local_obs_points)
            # 根据初始椭圆构建多面体
            polygon = self.findPolygon(ellipse, init_polygon, local_obs_points)
            # 进行保存
            decomp_polygons.append(polygon)

            if visualize:
                # 进行可视化
                plt.figure()
                # 绘制路径段
                plt.plot([pf[1], pr[1]], [pf[0], pr[0]], color="red")
                # 绘制初始多面体
                verticals = init_polygon.getVerticals()
                # 绘制多面体顶点
                plt.plot([v[1] for v in verticals] + [verticals[0][1]], [v[0] for v in verticals] + [verticals[0][0]], color="blue", linestyle="--")
                # 绘制障碍物点
                plt.scatter([p[1] for p in local_obs_points], [p[0] for p in local_obs_points], marker="o")
                # 绘制椭圆
                ellipse_x, ellipse_y = list(), list()
                for theta in np.linspace(-np.pi, np.pi, 1000):
                    raw_point = np.array([np.cos(theta), np.sin(theta)])
                    ellipse_point = np.dot(ellipse.C_, raw_point) + ellipse.d_
                    ellipse_x.append(ellipse_point[0])
                    ellipse_y.append(ellipse_point[1])
                plt.plot(ellipse_y, ellipse_x, color="orange")
                # 绘制最终多面体
                # 得到多面体顶点
                verticals = polygon.getVerticals()
                # 绘制多面体顶点
                plt.plot([v[1] for v in verticals] + [verticals[0][1]], [v[0] for v in verticals] + [verticals[0][0]], color="green")
                plt.show()

        return decomp_polygons
    
    # 构建初始多面体
    def initPolygon(self, pf: np.array, pr: np.array) -> Polygon:
        # 记录多面体的平面
        polygon_planes = list()
        # 得到线段方向向量
        dire = self.normalize(pr - pf)
        # 得到线段法向量
        dire_h = np.array([dire[1], -dire[0]])
        # 得到平行范围
        p_1 = pf + self.consider_range_ * dire_h
        p_2 = pf - self.consider_range_ * dire_h
        polygon_planes.append(Hyperplane(dire_h, p_1))
        polygon_planes.append(Hyperplane(-dire_h, p_2))
        # 得到垂直范围
        p_3 = pr + self.consider_range_ * dire
        p_4 = pf - self.consider_range_ * dire
        polygon_planes.append(Hyperplane(dire, p_3))
        polygon_planes.append(Hyperplane(-dire, p_4))
        # 构建多面体
        polygon = Polygon(polygon_planes)
        return polygon

    # 得到初始椭圆
    def findEllipse(self, pf: np.array, pr: np.array, obs_points: list[np.array]) -> Ellipse:
        # 计算长轴
        long_axis_value = np.linalg.norm(pr - pf) / 2
        axes = np.array([long_axis_value, long_axis_value])
        # 计算旋转
        rotation = self.vec2Rotation(pr - pf)
        # 计算初始椭圆
        C = np.dot(rotation, np.dot(np.array([[axes[0], 0], [0, axes[1]]]), np.transpose(rotation)))
        d = (pr + pf) / 2
        ellipse = Ellipse(C, d)
        # 得到椭圆内的障碍物点
        inside_obs_points = ellipse.insidePoints(obs_points)
        # 对椭圆进行调整，使得全部障碍物点都在椭圆外
        while inside_obs_points:
            # 得到与椭圆距离最近的点
            closest_obs_point = ellipse.closestPoint(inside_obs_points)
            # 将最近点转到椭圆坐标系下
            closest_obs_point = np.dot(np.transpose(rotation), closest_obs_point - ellipse.d_) 
            # 根据最近点，在椭圆长轴不变的情况下对短轴进行改变，使得，障碍物点在椭圆上
            if Compare.small(closest_obs_point[0], axes[0]):
                axes[1] = np.abs(closest_obs_point[1]) / np.sqrt(1 - (closest_obs_point[0] / axes[0]) ** 2)
            # 更新椭圆
            ellipse.C_ = np.dot(rotation, np.dot(np.array([[axes[0], 0], [0, axes[1]]]), np.transpose(rotation)))
            # 更新椭圆内部障碍物
            inside_obs_points = ellipse.insidePoints(inside_obs_points, include_bound=False)
        return ellipse

    # 进行多面体的构建
    def findPolygon(self, ellipse: Ellipse, init_polygon: Polygon, obs_points: list[np.array]) -> Polygon:
        # 多面体由多个超平面构成
        polygon_planes = copy.deepcopy(init_polygon.hyper_planes_)
        # 初始化范围超平面
        remain_obs_points = obs_points
        while remain_obs_points:
            # 得到与椭圆最近障碍物
            closest_point = ellipse.closestPoint(remain_obs_points)
            # 计算该处的切平面的法向量
            norm_vector = np.dot(np.linalg.inv(ellipse.C_), np.dot(np.linalg.inv(ellipse.C_), (closest_point - ellipse.d_)))
            norm_vector = self.normalize(norm_vector)
            # 构建平面
            hyper_plane = Hyperplane(norm_vector, closest_point)
            # 保存到多面体平面中
            polygon_planes.append(hyper_plane)
            # 去除切平面外部的障碍物
            new_remain_obs_points = list()
            for point in remain_obs_points:
                if Compare.small(hyper_plane.signDist(point), 0):
                    new_remain_obs_points.append(point)
            remain_obs_points = new_remain_obs_points
        polygon = Polygon(polygon_planes)
        return polygon
    
    # 正则化向量
    def normalize(self, v: np.array):
        v_v = np.linalg.norm(v)
        if v_v == 0:
            return v
        else:
            return v / v_v

    # 求向量的朝向角
    def vec2Rotation(self, v: np.array):
        yaw = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return R

def main1():
    line_points = [np.array([-1.5, 0.0]), np.array([1.5, 0.3])]
    obs_points = [
        np.array([-0.2, 1.5]),
        np.array([0, 1.5]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1.8, 0]),
        np.array([0.8, -1]),
        np.array([-0.5, -0.5]),
        np.array([-0.75 ,-0.5]),
        np.array([-1, -0.5]),
        np.array([-1, 0.8])
    ]

    convex_decomp = ConvexDecomp(2)
    convex_decomp.decomp(line_points, obs_points)

def main2():
    # 读取图片作为占据栅格图
    map_path = os.path.dirname(__file__) + "/../data/test.png"
    if not os.path.exists(map_path):
        print("occupancy grid does not exist in ", map_path)
        exit(0)
    occupancy_grid = 255 - cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

    # 随机设置起点和终点
    np.random.seed(0)
    height, width = occupancy_grid.shape
    while True:
        start_point = Point(np.random.randint(0, height), np.random.randint(0, width))
        if occupancy_grid[start_point.x_][start_point.y_] == 0:
            break
    while True:
        goal_point = Point(np.random.randint(0, height), np.random.randint(0, width))
        if occupancy_grid[goal_point.x_][goal_point.y_] == 0 and (goal_point - start_point).value() > height:
            break
    
    # 确认规划设置环境
    plt.figure()
    plt.imshow(occupancy_grid)
    plt.plot(start_point.y_, start_point.x_, marker="o", color="red")
    plt.plot(goal_point.y_, goal_point.x_, marker="o", color="green")
    plt.show()

    # 进行路径生成
    jps = JPS()
    time_start = time.time()
    searched_path = jps.search(start_point, goal_point, occupancy_grid)
    time_end = time.time()
    print("path search time consuming: ", time_end - time_start, " s")
    # 进行可视化
    if searched_path is not None:
        xs, ys = list(), list()
        for p in searched_path:
            xs.append(p.x_)
            ys.append(p.y_)
        plt.figure()
        plt.imshow(occupancy_grid)
        plt.plot(start_point.y_, start_point.x_, marker="o", color="red")
        plt.plot(goal_point.y_, goal_point.x_, marker="o", color="green")
        plt.plot(ys, xs)
        plt.show()
    else:
        exit(0)

    # 进行格式转换
    line_points = list()
    for p in searched_path:
        line_points.append(np.array([p.x_, p.y_]))
    obs_points = list()
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i][j] > 0:
                obs_points.append(np.array([i, j]))
    # 根据生成路径进行凸分解
    consider_range = 200
    convex_decomp = ConvexDecomp(consider_range)
    time_start = time.time()
    convex_decomp.decomp(line_points, obs_points, False)
    time_end = time.time()
    print("covex decomp time consuming: ", time_end - time_start, " s")

if __name__ == "__main__":
    main1()
