#! /usr/bin/env python
#! -*- coding: utf-8 -*-

import numpy as np
from convex_decomp import *
import osqp
from scipy import sparse
import matplotlib.pyplot as plt

# 五阶伯恩斯坦多项式
class QuinticBernsteinPolynomial:
    def __init__(self, params: list, time_allocation: float) -> None:
        assert(len(params) == 6)
        self.params_ = params
        self.time_allocation_ = time_allocation

    # 计算值
    def value(self, t):
        u = t / self.time_allocation_
        return self.params_[0] * (1 - u)**5 + self.params_[1] * 5 * (1 - u)**4 * u + self.params_[2] * 10 * (1 - u)**3 * u**2 + self.params_[3] * 10 * (1 - u)**2 * u**3 + self.params_[4] * 5 * (1 - u) * u**4 + self.params_[5] * u**5

    # 计算一阶导数
    def derivative(self, t):
        u = t / self.time_allocation_
        return 1 / self.time_allocation_ * (self.params_[0] * (-5) * (1 - u)**4 + self.params_[1] * (5 * (1 - u)**4 - 20 * (1 - u)**3 * u) + self.params_[2] * (20 * (1 - u)**3 * u - 30 * (1 - u)**2 * u**2) + self.params_[3] * (30 * (1 - u)**2 * u**2 - 20 * (1 - u) * u**3) + self.params_[4] * (20 * (1 - u) * u**3 - 5 * u**4) + self.params_[5] * 5 * u**4)

    # 计算二阶导数
    def secondOrderDerivative(self, t):
        u = t / self.time_allocation_
        return (1 / self.time_allocation_)**2 * (self.params_[0] * 20 * (1 - u)**3 + self.params_[1] * 5 * (-8 * (1 - u)**3 + 12 * (1 - u)**2 * u) + self.params_[2] * 10 * (2 * (1 - u)**3 - 12 * (1 - u)**2 * u + 6 * (1 - u) * u**2) + self.params_[3] * 10 * (6 * (1 - u)**2 * u - 12 * (1 - u) * u**2 + 2 * u**3) + self.params_[4] * 5 * (12 * (1 - u) * u**2 - 8 * u**3) + self.params_[5] * 20 * u**3)

    # 计算三阶导数
    def thirdOrderDerivative(self, t):
        u = t / self.time_allocation_
        return (1 / self.time_allocation_)**3 * (self.params_[0] * (-60) * (1 - u)**2 + self.params_[1] * 5 * (36 * (1 - u)**2 - 24 * (1 - u) * u) + self.params_[2] * 10 * (-18 * (1 - u)**2 + 36 * (1 - u) * u - 6 * u**2) + self.params_[3] * 10 * (6 * (1 - u)**2 - 36 * (1 - u) * u + 18 * u**2)+ self.params_[4] * 5 * (24 * (1 - u) * u - 36 * u**2)  + self.params_[5] * 60 * u**2)

# 分段轨迹
class PieceWiseTrajectory:
    def __init__(self, x_params: list, y_params: list, time_allocations: list) -> None:
        self.segment_num_ = len(time_allocations)
        self.time_segments_ = np.cumsum(time_allocations)
        self.trajectory_segments_ = list()
        for i in range(self.segment_num_):
            self.trajectory_segments_.append((QuinticBernsteinPolynomial(x_params[i], time_allocations[i]), QuinticBernsteinPolynomial(y_params[i], time_allocations[i])))

    # 根据时间获取下标
    def index(self, t):
        for i in range(self.segment_num_):
            if t <= self.time_segments_[i]:
                return i
        return None
    
    # 得到坐标
    def getPos(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].value(t), self.trajectory_segments_[index][1].value(t)
    
    # 得到速度
    def getVel(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].derivative(t), self.trajectory_segments_[index][1].derivative(t)

    # 得到加速度
    def getAcc(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].secondOrderDerivative(t), self.trajectory_segments_[index][1].secondOrderDerivative(t)

    # 得到jerk
    def getJerk(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].thirdOrderDerivative(t), self.trajectory_segments_[index][1].thirdOrderDerivative(t)

# 轨迹优化器
class TrajectoryOptimizer:
    def __init__(self, vel_max, acc_max, jerk_max) -> None:
        # 运动上限
        self.vel_max_ = vel_max
        self.acc_max_ = acc_max
        self.jerk_max_ = jerk_max
        # 得到维度
        self.dim_ = 2
        # 得到曲线阶数
        self.degree_ = 5
        # 自由度
        self.freedom_ = self.degree_ + 1

    # 进行优化
    def optimize(self, start_state: np.array, end_state: np.array, line_points: list[np.array], polygons: list[Polygon]):
        assert(len(line_points) == len(polygons) + 1)
        # 得到分段数量
        segment_num = len(polygons)
        assert(segment_num >= 1)
        # 计算初始时间分配
        time_allocations = list()
        for i in range(segment_num):
            time_allocations.append(np.linalg.norm(line_points[i+1] - line_points[i]) / self.vel_max_)
        # 进行优化迭代
        max_inter = 10
        cur_iter = 0
        while cur_iter < max_inter:
            # 进行轨迹优化
            piece_wise_trajectory = self.optimizeIter(start_state, end_state, polygons, time_allocations, segment_num)
            # 对优化轨迹进行时间调整，以保证轨迹满足运动上限约束
            cur_iter += 1
            # 计算每一段轨迹的最大速度，最大加速度，最大jerk
            condition_fit = True
            for n in range(segment_num):
                # 得到最大速度，最大加速度，最大jerk
                t_samples = np.linspace(0, time_allocations[n], 100)
                v_max, a_max, j_max = self.vel_max_, self.acc_max_, self.jerk_max_
                for t_sample in t_samples:
                    v_max = max(v_max, np.abs(piece_wise_trajectory.trajectory_segments_[n][0].derivative(t_sample)), np.abs(piece_wise_trajectory.trajectory_segments_[n][1].derivative(t_sample)))
                    a_max = max(a_max, np.abs(piece_wise_trajectory.trajectory_segments_[n][0].secondOrderDerivative(t_sample)), np.abs(piece_wise_trajectory.trajectory_segments_[n][1].secondOrderDerivative(t_sample)))
                    j_max = max(j_max, np.abs(piece_wise_trajectory.trajectory_segments_[n][0].thirdOrderDerivative(t_sample)), np.abs(piece_wise_trajectory.trajectory_segments_[n][1].thirdOrderDerivative(t_sample)))
                # 判断是否满足约束条件
                if Compare.large(v_max, self.vel_max_) or Compare.large(a_max, self.acc_max_) or Compare.large(j_max, self.jerk_max_):
                    ratio = max(1, v_max / self.vel_max_, (a_max / self.acc_max_)**0.5, (j_max / self.jerk_max_)**(1/3))
                    time_allocations[n] = ratio * time_allocations[n]
                    condition_fit = False
            if condition_fit:
                break
        return piece_wise_trajectory
    
    # 优化迭代
    def optimizeIter(self, start_state: np.array, end_state: np.array, polygons: list[Polygon], time_allocations: list, segment_num):
        # 构建目标函数 inter (jerk)^2
        inte_jerk_square = np.array([
            [720.0, -1800.0, 1200.0, 0.0, 0.0, -120.0],
            [-1800.0, 4800.0, -3600.0, 0.0, 600.0, 0.0],
            [1200.0, -3600.0, 3600.0, -1200.0, 0.0, 0.0],
            [0.0, 0.0, -1200.0, 3600.0, -3600.0, 1200.0],
            [0.0, 600.0, 0.0, -3600.0, 4800.0, -1800.0],
            [-120.0, 0.0, 0.0, 1200.0, -1800.0, 720.0]
        ])
        # 二次项系数
        P = np.zeros((self.dim_ * segment_num * self.freedom_, self.dim_ * segment_num * self.freedom_))
        for sigma in range(self.dim_):
            for n in range(segment_num):
                for i in range(self.freedom_):
                    for j in range(self.freedom_):
                        index_i = sigma * segment_num * self.freedom_ + n * self.freedom_ + i
                        index_j = sigma * segment_num * self.freedom_ + n * self.freedom_ + j
                        P[index_i][index_j] = inte_jerk_square[i][j] / (time_allocations[n] ** 5)
        P = P * 2
        P = sparse.csc_matrix(P)
        # 一次项系数
        q = np.zeros((self.dim_ * segment_num * self.freedom_,))

        # 构建约束条件
        equality_constraints_num = 5 * self.dim_ + 3 * (segment_num - 1) * self.dim_
        inequality_constraints_num = 0
        for polygon in polygons:
            inequality_constraints_num += self.freedom_ * len(polygon.hyper_planes_)

        A = np.zeros((equality_constraints_num + inequality_constraints_num, self.dim_ * segment_num * self.freedom_))
        lb = -float("inf") * np.ones((equality_constraints_num + inequality_constraints_num,))
        ub = float("inf") * np.ones((equality_constraints_num + inequality_constraints_num,))
        
        # 构建等式约束条件（起点位置、速度、加速度；终点位置、速度；连接处的零、一、二阶导数）
        # 起点x位置
        A[0][0] = 1
        lb[0] = start_state[0]
        ub[0] = start_state[0]
        # 起点y位置
        A[1][segment_num * self.freedom_] = 1
        lb[1] = start_state[1]
        ub[1] = start_state[1]
        # 起点x速度
        A[2][0] = -5 / time_allocations[0]
        A[2][1] = 5 / time_allocations[0]
        lb[2] = start_state[2]
        ub[2] = start_state[2]
        # 起点y速度
        A[3][segment_num * self.freedom_] = -5 / time_allocations[0]
        A[3][segment_num * self.freedom_ + 1] = 5 / time_allocations[0]
        lb[3] = start_state[3]
        ub[3] = start_state[3]
        # 起点x加速度
        A[4][0] = 20 / time_allocations[0]**2
        A[4][1] = -40 / time_allocations[0]**2
        A[4][2] = 20 / time_allocations[0]**2
        lb[4] = start_state[4]
        ub[4] = start_state[4]
        # 起点y加速度
        A[5][segment_num * self.freedom_] = 20 / time_allocations[0]**2
        A[5][segment_num * self.freedom_ + 1] = -40 / time_allocations[0]**2
        A[5][segment_num * self.freedom_ + 2] = 20 / time_allocations[0]**2
        lb[5] = start_state[5]
        ub[5] = start_state[5]
        # 终点x位置
        A[6][segment_num * self.freedom_ - 1] = 1
        lb[6] = end_state[0]
        ub[6] = end_state[0]
        # 终点y位置
        A[7][self.dim_ * segment_num * self.freedom_ - 1] = 1
        lb[7] = end_state[1]
        ub[7] = end_state[1]
        # 终点x速度
        A[8][segment_num * self.freedom_ - 1] = 5 / time_allocations[-1]
        A[8][segment_num * self.freedom_ - 2] = -5 / time_allocations[-1]
        lb[8] = end_state[2]
        ub[8] = end_state[2]
        # 终点y速度
        A[9][self.dim_ * segment_num * self.freedom_ - 1] = 5 / time_allocations[-1]
        A[9][self.dim_ * segment_num * self.freedom_ - 2] = -5 / time_allocations[-1]
        lb[9] = end_state[3]
        ub[9] = end_state[3]

        # 连接处的零阶导数相等
        constraints_index = 10
        for sigma in range(self.dim_):
            for n in range(segment_num - 1):
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 1] = 1
                A[constraints_index][sigma * segment_num * self.freedom_ + (n+1) * self.freedom_] = -1
                lb[constraints_index] = 0
                ub[constraints_index] = 0
                constraints_index += 1
        # 连接处的一阶导数相等
        for sigma in range(self.dim_):
            for n in range(segment_num - 1):
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 1] = 5 / time_allocations[n]
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 2] = -5 / time_allocations[n]
                A[constraints_index][sigma * segment_num * self.freedom_ + (n+1) * self.freedom_] = 5 / time_allocations[n + 1]
                A[constraints_index][sigma * segment_num * self.freedom_ + (n+1) * self.freedom_ + 1] = -5 / time_allocations[n + 1]
                lb[constraints_index] = 0
                ub[constraints_index] = 0
                constraints_index += 1
        # 连接处的二阶导数相等
        for sigma in range(self.dim_):
            for n in range(segment_num - 1):
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 1] = 20 / time_allocations[n]**2
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 2] = -40 / time_allocations[n]**2
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 3] = 20 / time_allocations[n]**2
                A[constraints_index][sigma * segment_num * self.freedom_ + (n+1) * self.freedom_] = -20 / time_allocations[n + 1]**2
                A[constraints_index][sigma * segment_num * self.freedom_ + (n+1) * self.freedom_ + 1] = 40 / time_allocations[n + 1]**2
                A[constraints_index][sigma * segment_num * self.freedom_ + (n+1) * self.freedom_ + 2] = -20 / time_allocations[n + 1]**2
                lb[constraints_index] = 0
                ub[constraints_index] = 0
                constraints_index += 1
        
        # 构建不等式约束条件
        for n in range(segment_num):
            for k in range(self.freedom_):
                for hyper_plane in polygons[n].hyper_planes_:
                    A[constraints_index][n * self.freedom_ + k] = hyper_plane.n_[0]
                    A[constraints_index][segment_num * self.freedom_ + n * self.freedom_ + k] = hyper_plane.n_[1]
                    ub[constraints_index] = np.dot(hyper_plane.n_, hyper_plane.d_)
                    constraints_index += 1
        assert(constraints_index == equality_constraints_num + inequality_constraints_num)
        A = sparse.csc_matrix(A)
        
        # 进行qp求解
        prob = osqp.OSQP()
        prob.setup(P, q, A, lb, ub, warm_start=True)
        res = prob.solve()
        if res.info.status != "solved":
            raise ValueError("OSQP did not solve the problem!")

        # 根据参数进行轨迹解析
        trajectory_x_params, trajectory_y_params = list(), list()
        for n in range(segment_num):
            trajectory_x_params.append(res.x[self.freedom_ * n: self.freedom_ * (n+1)])
            trajectory_y_params.append(res.x[segment_num * self.freedom_ + self.freedom_ * n: segment_num * self.freedom_ + self.freedom_ * (n+1)])
        piece_wise_trajectory = PieceWiseTrajectory(trajectory_x_params, trajectory_y_params, time_allocations)
        
        return piece_wise_trajectory

if __name__ == "__main__":
    # 初始和终止状态
    start_state = np.array([-1.5, 0.0, 1.0, -0.3, -0.8, -0.8])  # [px, py, vx, vy, ax, ay]
    end_state = np.array([2.8, 1.6, -0.4, 0.7])  # [px, py, vx, vy]
    # 路径点
    line_points = [np.array([-1.5, 0.0]), np.array([0.0, 0.8]), np.array([1.5, 0.3])]
    # 障碍物点
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
    
    # 进行凸分解
    convex_decomp = ConvexDecomp(2)
    decomp_polygons = convex_decomp.decomp(line_points, obs_points, False)
    
    # 进行轨迹优化
    traj_opt = TrajectoryOptimizer(2, 4, 8)
    piece_wise_trajectory = traj_opt.optimize(start_state, end_state, line_points, decomp_polygons)
    
    # 进行可视化
    # 数据可视化
    t_samples = np.linspace(0, piece_wise_trajectory.time_segments_[-1], 100)
    # 位置可视化
    x_samples, y_samples = list(), list()
    for t_sample in t_samples:
        x, y = piece_wise_trajectory.getPos(t_sample)
        x_samples.append(x)
        y_samples.append(y)
    plt.figure()
    # 绘制障碍物点
    plt.scatter([p[0] for p in obs_points], [p[1] for p in obs_points], marker="o")
    # 绘制边界
    for polygon in decomp_polygons:
        verticals = polygon.getVerticals()
        # 绘制多面体顶点
        plt.plot([v[0] for v in verticals] + [verticals[0][0]], [v[1] for v in verticals] + [verticals[0][1]], color="green")
    plt.plot(x_samples, y_samples)
    plt.show()
    # 速度可视化
    vx_samples, vy_samples = list(), list()
    for t_sample in t_samples:
        vx, vy = piece_wise_trajectory.getVel(t_sample)
        vx_samples.append(vx)
        vy_samples.append(vy)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t_samples, vx_samples)
    plt.subplot(2, 1, 2)
    plt.plot(t_samples, vy_samples)
    plt.show()
    # 加速度可视化
    ax_samples, ay_samples = list(), list()
    for t_sample in t_samples:
        ax, ay = piece_wise_trajectory.getAcc(t_sample)
        ax_samples.append(ax)
        ay_samples.append(ay)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t_samples, ax_samples)
    plt.subplot(2, 1, 2)
    plt.plot(t_samples, ay_samples)
    plt.show()
    # jerk可视化
    jx_samples, jy_samples = list(), list()
    for t_sample in t_samples:
        jx, jy = piece_wise_trajectory.getJerk(t_sample)
        jx_samples.append(jx)
        jy_samples.append(jy)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t_samples, jx_samples)
    plt.subplot(2, 1, 2)
    plt.plot(t_samples, jy_samples)
    plt.show()