import numpy as np
from drone_mujoco import DroneMujocoEnv
from stable_baselines3 import SAC

env = DroneMujocoEnv()
model = SAC.load("drone_model", env=env)

for ep in range(5):
    obs, _ = env.reset()
    total_reward = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        pos = obs[:3]
        target = env.target_pos
        dist = np.linalg.norm(pos - target)
        print(f"ep{ep} step{step:3d} pos:[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] dist:{dist:.3f} reward:{reward:.1f}")
        if terminated or truncated:
            break
    print(f"=== ep{ep} 总奖励: {total_reward:.1f} ===\n")
