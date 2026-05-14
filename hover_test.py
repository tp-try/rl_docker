import airsim
import numpy as np
from stable_baselines3 import SAC
from hover import DroneAirSimEnv
import time

env = DroneAirSimEnv()
model = SAC.load("/app/airsim_hover/airsim_model", env=env)

# 测试不同目标位置
test_positions = [
    [0.0,  0.0, -2.0],   # 原始目标
    [0.8,  0.0, -2.0],   # 右边0.8m
    [-0.8, 0.0, -2.0],   # 左边0.8m
    [0.0,  0.8, -2.0],   # 前面0.8m
    [0.5,  0.5, -2.4],   # 斜方向
    [3.0,  0.0, -3.0],   # 右边3m，高3m
]

for target in test_positions:
    print(f"\n目标位置: {target}")
    env.target_pos = np.array(target)
    obs, _ = env.reset()

    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        dist = np.linalg.norm(obs[:3])
        if i % 20 == 0:
            print(f"step{i:3d} dist:{dist:.3f} reward:{reward:.2f}")
        if terminated or truncated:
            break

    print(f"最终dist: {dist:.3f}m")

env.client.landAsync().join()
env.client.armDisarm(False)
print("测试完成！")