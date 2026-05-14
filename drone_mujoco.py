import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces


class DroneMujocoEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("/app/drone.xml")
        self.data = mujoco.MjData(self.model)

        # 目标悬停位置
        self.target_pos = np.array([0.0, 0.0, 2.0])

        # 状态：位置(3) + 速度(3) + 姿态四元数(4) + 角速度(3) = 13维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # 动作：4个电机推力，每个 0-5N
        self.action_space = spaces.Box(
            low=0.0, high=5.0, shape=(4,), dtype=np.float32)

        self.max_steps = 1000
        self.step_count = 0

    def _get_obs(self):
        pos  = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel  = self.data.qvel[:3]
        angv = self.data.qvel[3:6]
        return np.concatenate([pos, vel, quat, angv]).astype(np.float32)

    def _get_reward(self, obs):
        pos  = obs[:3]
        vel  = obs[3:6]
        angv = obs[9:12]

        dist = np.linalg.norm(pos - self.target_pos)
        
        if dist < 0.1:
            return 100.0

        reward_dist = np.exp(-dist) * 10.0
        reward_vel  = -0.1 * np.linalg.norm(vel)
        reward_angv = -0.1 * np.linalg.norm(angv)

        return reward_dist + reward_vel + reward_angv

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 随机初始位置（在目标附近）
        self.data.qpos[:3] = self.target_pos + np.random.uniform(-0.5, 0.5, 3)
        self.data.qpos[2] = max(self.data.qpos[2], 0.3)  # 不低于地面
        self.data.qpos[3:7] = [1, 0, 0, 0]  # 初始姿态水平

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # 设置电机推力
        self.data.ctrl[:] = np.clip(action, 0.0, 5.0)

        # 仿真10步（相当于0.1秒）
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs     = self._get_obs()
        reward  = self._get_reward(obs)

        # 终止条件
        pos = obs[:3]
        crashed    = pos[2] < 0.05
        out_bounds = (abs(pos[0]) > 15 or abs(pos[1]) > 15 or pos[2] > 20)
        flipped    = self.data.qpos[3] < 0.5  # 四元数w分量，翻转检测

        terminated = bool(crashed or out_bounds or flipped)
        truncated  = self.step_count >= self.max_steps

        if terminated:
            reward -= 50.0

        return obs, reward, terminated, truncated, {}

    def close(self):
        pass
