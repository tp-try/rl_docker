import gymnasium as gym
import numpy as np
import airsim
import time
from gymnasium import spaces


class DroneAirSimEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = airsim.MultirotorClient(ip="172.31.240.1")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.target_pos = np.array([0.0, 0.0, -2.0])

        # 问题2修复：用相对位置代替绝对位置，训练速度x10
        # obs = [pos_error(3), vel(3)] = 6维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.max_steps = 500
        self.step_count = 0
        self.prev_action = np.zeros(3)

    def _get_obs(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        pos_arr = np.array([pos.x_val, pos.y_val, pos.z_val])
        vel_arr = np.array([vel.x_val, vel.y_val, vel.z_val])

        # 问题2修复：返回相对位置误差而不是绝对位置
        pos_error = pos_arr - self.target_pos
        return np.concatenate([pos_error, vel_arr]).astype(np.float32)

    def _get_reward(self, obs, action):
        pos_error = obs[:3]
        vel = obs[3:]
        dist = np.linalg.norm(pos_error)

        # 问题1修复：去掉跳变的100分奖励，改成连续奖励
        reward = np.exp(-dist) - 0.1 * np.linalg.norm(vel)

        # 问题5修复：加入动作惩罚，抑制抖动
        action_penalty = -0.01 * np.linalg.norm(action)
        reward += action_penalty

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        x = float(np.random.uniform(-1, 1))
        y = float(np.random.uniform(-1, 1))
        z = float(-2 + np.random.uniform(-0.5, 0.5))
        self.client.moveToPositionAsync(x, y, z, velocity=2).join()
        self.step_count = 0
        self.prev_action = np.zeros(3)
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        vx = float(action[0]) * 2.0
        vy = float(action[1]) * 2.0
        vz = float(action[2]) * 2.0

        # 问题4修复：统一控制时间，duration和sleep一致
        self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1).join()

        new_obs = self._get_obs()
        reward = self._get_reward(new_obs, action)
        self.prev_action = action.copy()

        collision = self.client.simGetCollisionInfo().has_collided
        pos_error = new_obs[:3]
        actual_pos = pos_error + self.target_pos

        # 问题6修复：更紧的高度约束
        out_of_bounds = (
            abs(actual_pos[0]) > 15 or
            abs(actual_pos[1]) > 15 or
            actual_pos[2] > -0.5 or    # 太高（接近地面）
            actual_pos[2] < -5.0       # 太低
        )

        terminated = bool(collision or out_of_bounds)
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= 50.0

        return new_obs, reward, terminated, truncated, {}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)