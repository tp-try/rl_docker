import os
import glob
from drone_mujoco import DroneMujocoEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

env = DroneMujocoEnv()

checkpoints = sorted(glob.glob("drone_ckpt_*_steps.zip"))
if checkpoints:
    latest = checkpoints[-1]
    print(f"加载检查点: {latest}")
    model = SAC.load(latest.replace(".zip", ""), env=env)
elif os.path.exists("drone_model.zip"):
    print("加载已有模型...")
    model = SAC.load("drone_model", env=env)
else:
    print("新建模型开始训练...")
    model = SAC("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                batch_size=256,
                buffer_size=1000000)

checkpoint = CheckpointCallback(
    save_freq=10000,
    save_path='./',
    name_prefix='drone_ckpt'
)

model.learn(total_timesteps=1000000, reset_num_timesteps=False, callback=checkpoint)
model.save("drone_model")
print("训练完成！")
