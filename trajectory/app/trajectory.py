import gymnasium as gym
import numpy as np
import airsim
import time
from gymnasium import spaces


class DroneTrajectoryEnv(gym.Env):
    """
    无人机轨迹跟踪环境 v3

    主要设计：
    1. 课程学习：从小间距开始，逐步增大难度
    2. 奖励函数：势能奖励（主驱动）+ 接近奖励 + 到达奖励
       - 避免惩罚朝目标飞行的速度
       - 只惩罚横向漂移（垂直于目标方向的速度）
    3. 观测归一化：稳定输入分布
    4. z轴约束：限制单步高度变化，防止z轴失控
    5. 碰撞误报修复：时间戳判断
    6. AirSim断连自动重连
    """

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(self):
        super().__init__()

        # AirSim连接
        self.client = airsim.MultirotorClient(ip="172.31.240.1")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # ---- 飞行范围 ----
        self.xy_range = 5.0     # 水平范围 ±5m
        self.z_min    = -6.0    # 最低高度（AirSim z轴向下为正，这里用负数表示上方）
        self.z_max    = -1.0    # 最高高度

        # ---- 路径点参数 ----
        self.num_waypoints = 5
        self.waypoints     = []
        self.current_idx   = 0
        self.reach_dist    = 0.3    # 到达阈值（米）

        # ---- 课程学习 ----
        # 等级0~4，每级对应 [水平最大间距, z最大间距]
        # z间距明显小于水平间距，防止z轴失控
        self.curriculum_level  = 0
        self.success_count     = 0   # 连续成功次数（宽容模式：失败-1而不是清零）
        self.episode_count     = 0
        self.promote_threshold = 10  # 连续成功N次升级
        self.level_params = [
            # [xy_max_dist, z_max_dist]
            [2.0, 0.8],   # 等级0：水平2m，高度0.8m
            [3.0, 1.2],   # 等级1：水平3m，高度1.2m
            [4.5, 1.8],   # 等级2：水平4.5m，高度1.8m
            [6.0, 2.5],   # 等级3：水平6m，高度2.5m
            [8.0, 3.5],   # 等级4：水平8m，高度3.5m（全范围）
        ]

        # ---- 观测归一化参数 ----
        # pos_error归一化：除以最大可能距离
        self.pos_norm  = 10.0   # 位置误差归一化因子（米）
        # vel归一化：最大速度3m/s
        self.vel_norm  = 3.0    # 速度归一化因子（m/s）
        # next_dir已经是单位向量，不需要归一化

        # ---- 观测/动作空间 ----
        # obs: [pos_error归一化(3), vel归一化(3), next_dir(3)] = 9维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # action: [vx, vy, vz] 归一化到[-1,1]，乘以max_vel得到实际速度
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.max_vel = 3.0   # 最大速度（m/s）

        # ---- 其他状态 ----
        self.max_steps           = 1000
        self.step_count          = 0
        self.collision_reset_time = 0
        self.prev_dist           = None  # 上一步到当前路径点的距离

    # ------------------------------------------------------------------ #
    # AirSim 安全调用（自动重连）
    # ------------------------------------------------------------------ #
    def _reconnect(self, max_retries=10, wait=3.0):
        for i in range(max_retries):
            try:
                print(f"[重连] 第 {i+1}/{max_retries} 次...")
                self.client = airsim.MultirotorClient(ip="172.31.240.1")
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                print("[重连] 成功")
                return
            except Exception as e:
                print(f"[重连] 失败: {e}")
                time.sleep(wait)
        raise RuntimeError("AirSim 重连失败，请检查仿真器")

    def _safe_call(self, fn, *args, **kwargs):
        for attempt in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"[AirSim] 调用失败(第{attempt+1}次): {e}")
                if attempt < 2:
                    self._reconnect()
                else:
                    raise

    # ------------------------------------------------------------------ #
    # 路径点生成（带间距约束）
    # ------------------------------------------------------------------ #
    def _generate_waypoints(self):
        xy_max, z_max_step = self.level_params[self.curriculum_level]
        xy_min = 0.5   # 最小水平间距
        z_min_step = 0.1  # 最小高度变化

        waypoints = []

        # 第一个点：在中心区域随机生成，避免太靠边
        margin = min(1.5, self.xy_range * 0.3)
        x = float(np.random.uniform(-(self.xy_range - margin), self.xy_range - margin))
        y = float(np.random.uniform(-(self.xy_range - margin), self.xy_range - margin))
        z = float(np.random.uniform(self.z_min + 1.0, self.z_max - 0.5))
        waypoints.append(np.array([x, y, z]))

        for _ in range(self.num_waypoints - 1):
            prev = waypoints[-1]
            # 最多尝试100次生成满足约束的路径点
            for attempt in range(100):
                # 水平方向随机偏移
                dx = float(np.random.uniform(-xy_max, xy_max))
                dy = float(np.random.uniform(-xy_max, xy_max))
                # z方向：限制变化幅度
                dz = float(np.random.uniform(-z_max_step, z_max_step))

                nx = prev[0] + dx
                ny = prev[1] + dy
                nz = prev[2] + dz

                # 边界检查
                if (abs(nx) > self.xy_range or abs(ny) > self.xy_range or
                        nz < self.z_min or nz > self.z_max):
                    continue

                # 水平距离检查（不能太近，也不能太远）
                horiz_dist = np.sqrt(dx**2 + dy**2)
                if horiz_dist < xy_min or horiz_dist > xy_max:
                    continue

                waypoints.append(np.array([nx, ny, nz]))
                break
            else:
                # 实在找不到，就在安全范围内生成一个保底点
                nx = float(np.clip(prev[0] + np.random.uniform(-1.0, 1.0),
                                   -self.xy_range + 0.5, self.xy_range - 0.5))
                ny = float(np.clip(prev[1] + np.random.uniform(-1.0, 1.0),
                                   -self.xy_range + 0.5, self.xy_range - 0.5))
                nz = float(np.clip(prev[2] + np.random.uniform(-0.5, 0.5),
                                   self.z_min + 0.5, self.z_max - 0.5))
                waypoints.append(np.array([nx, ny, nz]))

        return waypoints

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #
    def _get_next_dir(self):
        """返回从当前路径点指向下一路径点的单位向量，最后一个点返回零向量"""
        if self.current_idx + 1 < len(self.waypoints):
            diff = self.waypoints[self.current_idx + 1] - self.waypoints[self.current_idx]
            norm = np.linalg.norm(diff)
            if norm > 0.01:
                return diff / norm
        return np.zeros(3)

    def _get_obs(self):
        state = self._safe_call(self.client.getMultirotorState)
        pos   = state.kinematics_estimated.position
        vel   = state.kinematics_estimated.linear_velocity

        pos_arr = np.array([pos.x_val, pos.y_val, pos.z_val])
        vel_arr = np.array([vel.x_val, vel.y_val, vel.z_val])

        current_wp = self.waypoints[self.current_idx]
        pos_error  = pos_arr - current_wp  # 当前位置 - 目标位置（正方向=过头了）
        next_dir   = self._get_next_dir()

        # 归一化观测，保持输入分布稳定
        obs = np.concatenate([
            pos_error / self.pos_norm,   # 位置误差归一化
            vel_arr   / self.vel_norm,   # 速度归一化
            next_dir,                    # 下一路径点方向（已是单位向量）
        ]).astype(np.float32)

        # 裁剪防止极端值
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    # ------------------------------------------------------------------ #
    # 奖励函数
    # ------------------------------------------------------------------ #
    def _get_reward(self, obs, action):
        # 从obs中还原真实值（乘回归一化因子）
        pos_error = obs[:3] * self.pos_norm
        vel_vec   = obs[3:6] * self.vel_norm

        dist = np.linalg.norm(pos_error)
        vel  = np.linalg.norm(vel_vec)

        # ---- 1. 势能奖励：每步距离减少给正奖励，是主要驱动力 ----
        if self.prev_dist is not None:
            # 系数3.0：让势能奖励足够显著，引导模型快速靠近
            potential = (self.prev_dist - dist) * 3.0
        else:
            potential = 0.0
        self.prev_dist = dist

        # ---- 2. 接近奖励：给模型一个"当前有多近"的感知 ----
        # exp(-dist)：dist=0时=1.0，dist=1时≈0.37，dist=3时≈0.05
        proximity = np.exp(-dist)

        # ---- 3. 横向速度惩罚：只惩罚垂直于目标方向的速度分量 ----
        # 朝目标飞不惩罚，侧飞/倒飞才惩罚
        lateral_penalty = 0.0
        if dist > 0.1 and vel > 0.01:
            target_dir = -pos_error / dist
            # 纵向速度（朝目标方向）
            longitudinal_vel = np.dot(vel_vec, target_dir)
            # 横向速度 = 总速度² - 纵向速度²
            lateral_vel_sq = max(0.0, vel**2 - longitudinal_vel**2)
            lateral_vel    = np.sqrt(lateral_vel_sq)
            lateral_penalty = -0.05 * lateral_vel

        # ---- 4. 动作平滑惩罚：轻微惩罚大动作，避免抖动 ----
        action_penalty = -0.01 * np.linalg.norm(action)

        reward = potential + proximity + lateral_penalty + action_penalty
        return float(reward)

    # ------------------------------------------------------------------ #
    # 课程升级逻辑
    # ------------------------------------------------------------------ #
    def _update_curriculum(self, success):
        self.episode_count += 1
        if success:
            self.success_count += 1
            if (self.success_count >= self.promote_threshold and
                    self.curriculum_level < len(self.level_params) - 1):
                self.curriculum_level += 1
                self.success_count = 0
                xy_max, z_max = self.level_params[self.curriculum_level]
                print(f"[课程升级] -> 等级{self.curriculum_level} "
                      f"水平间距≤{xy_max}m 高度间距≤{z_max}m")
        else:
            # 失败：宽容模式，减1而不是直接清零（防止一次偶然失败重置进度）
            self.success_count = max(0, self.success_count - 1)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._safe_call(self.client.enableApiControl, True)
        self._safe_call(self.client.armDisarm, True)

        self.waypoints   = self._generate_waypoints()
        self.current_idx = 0
        self.prev_dist   = None

        # 飞到第一个路径点附近（加小扰动，提升鲁棒性）
        first_wp = self.waypoints[0]
        x = float(first_wp[0] + np.random.uniform(-0.3, 0.3))
        y = float(first_wp[1] + np.random.uniform(-0.3, 0.3))
        z = float(first_wp[2] + np.random.uniform(-0.2, 0.2))
        # 速度5：比之前快，加速训练
        self._safe_call(lambda: self.client.moveToPositionAsync(x, y, z, velocity=5).join())

        # 清除碰撞标志
        pose = self._safe_call(self.client.simGetVehiclePose)
        self._safe_call(self.client.simSetVehiclePose, pose, ignore_collision=True)
        time.sleep(0.15)
        self.collision_reset_time = self._safe_call(self.client.simGetCollisionInfo).time_stamp

        self.step_count = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action):
        self.step_count += 1

        # 执行速度指令
        vx = float(action[0]) * self.max_vel
        vy = float(action[1]) * self.max_vel
        vz = float(action[2]) * self.max_vel
        self._safe_call(lambda: self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1).join())

        new_obs = self._get_obs()
        reward  = self._get_reward(new_obs, action)

        # ---- 到达当前路径点 ----
        dist = np.linalg.norm(new_obs[:3] * self.pos_norm)  # 还原真实距离
        if dist < self.reach_dist:
            self.current_idx += 1

            if self.current_idx >= len(self.waypoints):
                # 完成所有路径点
                self._update_curriculum(success=True)
                return new_obs, reward + 30.0, True, False, {"success": True}

            # 切换到下一个路径点：获取当前真实位置，计算到新路径点的距离，
            # 作为势能奖励的基准（避免下一步势能符号突变）
            try:
                state    = self.client.getMultirotorState()
                p        = state.kinematics_estimated.position
                curr_pos = np.array([p.x_val, p.y_val, p.z_val])
                next_wp  = self.waypoints[self.current_idx]
                self.prev_dist = float(np.linalg.norm(curr_pos - next_wp))
            except Exception:
                # 获取失败时置None（下一步势能=0，无害）
                self.prev_dist = None

            # 到达中间路径点的小奖励（激励持续飞行）
            reward += 5.0

        # ---- 碰撞检测（时间戳方式，避免误报）----
        if self.step_count <= 8:
            collision = False
        else:
            ci        = self._safe_call(self.client.simGetCollisionInfo)
            collision = ci.has_collided and ci.time_stamp > self.collision_reset_time

        # ---- 出界检测（严格边界）----
        pos_error  = new_obs[:3] * self.pos_norm
        actual_pos = pos_error + self.waypoints[self.current_idx]

        z_high = actual_pos[2] > (self.z_max + 0.3)   # 比边界宽松0.3m容错
        z_low  = actual_pos[2] < (self.z_min - 0.3)
        x_out  = abs(actual_pos[0]) > (self.xy_range + 0.5)
        y_out  = abs(actual_pos[1]) > (self.xy_range + 0.5)
        out_of_bounds = z_high or z_low or x_out or y_out

        terminated = bool(collision or out_of_bounds)
        truncated  = self.step_count >= self.max_steps

        if terminated:
            print(f"终止 collision:{collision} pos:{actual_pos.round(2)} "
                  f"z_high:{z_high} z_low:{z_low} x_out:{x_out} y_out:{y_out}")
            self._update_curriculum(success=False)
            reward -= 20.0  # 失败惩罚（比到达奖励小，保持正向激励）

        if truncated and not terminated:
            self._update_curriculum(success=False)

        return new_obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    # Close
    # ------------------------------------------------------------------ #
    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass