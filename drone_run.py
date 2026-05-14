import airsim
import numpy as np
from stable_baselines3 import SAC
from drone_mujoco import DroneMujocoEnv
import time

env = DroneMujocoEnv()
model = SAC.load("drone_model", env=env)

client = airsim.MultirotorClient(ip="172.31.240.1")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
time.sleep(2)

print("开始测试...")
obs = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)

for i in range(200):
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    obs = np.array([
    pos.x_val, pos.y_val, -pos.z_val,   # 位置
    vel.x_val, vel.y_val, vel.z_val,     # 速度
    1, 0, 0, 0,                          # 姿态四元数（简化）
    0, 0, 0                              # 角速度（简化）
    ], dtype=np.float32)
    
    action, _ = model.predict(obs, deterministic=True)
    
    cur = obs[:3]
    new_pos = cur + action
    new_pos[2] = np.clip(new_pos[2], 0.5, 5.0)
    
    client.moveToPositionAsync(
        float(new_pos[0]), float(new_pos[1]), float(-new_pos[2]),
        velocity=1.0)
    
    time.sleep(0.1)
    print(f"step{i} pos:[{pos.x_val:.2f},{pos.y_val:.2f},{pos.z_val:.2f}] action:{action}")
