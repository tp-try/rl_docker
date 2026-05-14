import numpy as np
import mujoco
import imageio
from drone_mujoco import DroneMujocoEnv
from stable_baselines3 import SAC

env = DroneMujocoEnv()
model = SAC.load("drone_model", env=env)

renderer = mujoco.Renderer(env.model, height=480, width=640)
obs, _ = env.reset()

frames = []
for i in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    renderer.update_scene(env.data)
    frames.append(renderer.render())
    if terminated or truncated:
        obs, _ = env.reset()

imageio.mimsave('/mnt/c/Users/35202/Desktop/drone_test.gif', frames, fps=30)
print("保存完成！")
