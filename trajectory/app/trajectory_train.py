import os
import signal
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from trajectory import DroneTrajectoryEnv

# -----------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------
MODEL_DIR       = "./model"
CKPT_DIR        = "./model/checkpoints"
MODEL_PATH      = f"{MODEL_DIR}/trajectory_model"
BUFFER_PATH     = f"{MODEL_DIR}/replay_buffer"
CURRICULUM_PATH = f"{MODEL_DIR}/curriculum.npy"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

TOTAL_STEPS = 1_000_000
SAVE_FREQ   = 100_000       # 每10万步保存一次

# -----------------------------------------------------------------------
# 回调：每N步同时保存模型 + buffer + 课程状态
# -----------------------------------------------------------------------
class SaveAllCallback(BaseCallback):
    def __init__(self, save_freq, ckpt_dir, raw_env, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.ckpt_dir  = ckpt_dir
        self.raw_env   = raw_env

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            tag = f"step{self.num_timesteps}"
            self._save(tag)
        return True

    def _save(self, tag):
        self.model.save(f"{self.ckpt_dir}/trajectory_{tag}")
        self.model.save_replay_buffer(f"{self.ckpt_dir}/buffer_{tag}")
        np.save(
            f"{self.ckpt_dir}/curriculum_{tag}.npy",
            np.array([self.raw_env.curriculum_level,
                      self.raw_env.success_count,
                      self.raw_env.episode_count])
        )
        if self.verbose:
            print(f"\n[保存] {tag} | 课程等级:{self.raw_env.curriculum_level} "
                  f"总步数:{self.num_timesteps:,}")


# -----------------------------------------------------------------------
# 回调：Ctrl+C 中断时保存当前状态
# -----------------------------------------------------------------------
interrupted = False

def handle_interrupt(sig, frame):
    global interrupted
    print("\n[中断] 检测到 Ctrl+C，正在保存...")
    interrupted = True

signal.signal(signal.SIGINT, handle_interrupt)


class InterruptCallback(BaseCallback):
    def __init__(self, model_dir, raw_env):
        super().__init__()
        self.model_dir = model_dir
        self.raw_env   = raw_env

    def _on_step(self) -> bool:
        if interrupted:
            self.model.save(f"{self.model_dir}/trajectory_model_interrupted")
            self.model.save_replay_buffer(f"{self.model_dir}/replay_buffer_interrupted")
            np.save(
                f"{self.model_dir}/curriculum_interrupted.npy",
                np.array([self.raw_env.curriculum_level,
                          self.raw_env.success_count,
                          self.raw_env.episode_count])
            )
            print(f"[保存完成] 模型/buffer/课程 -> {self.model_dir}/*_interrupted.*")
            sys.exit(0)
        return True


# -----------------------------------------------------------------------
# 环境
# -----------------------------------------------------------------------
raw_env = DroneTrajectoryEnv()
env     = Monitor(raw_env)

# -----------------------------------------------------------------------
# 断点续训：自动找最新存档
# -----------------------------------------------------------------------
def find_latest_checkpoint():
    # 优先找中断存档
    p = f"{MODEL_DIR}/trajectory_model_interrupted.zip"
    if os.path.exists(p):
        print(f"[续训] 发现中断存档")
        return (
            f"{MODEL_DIR}/trajectory_model_interrupted",
            f"{MODEL_DIR}/replay_buffer_interrupted",
            f"{MODEL_DIR}/curriculum_interrupted.npy",
        )

    # 找 checkpoints 里步数最大的
    if os.path.exists(CKPT_DIR):
        zips = [f for f in os.listdir(CKPT_DIR)
                if f.startswith("trajectory_step") and f.endswith(".zip")]
        if zips:
            zips.sort(key=lambda x: int(x.replace("trajectory_step","").replace(".zip","")))
            latest = zips[-1].replace(".zip", "")
            step_tag = latest.replace("trajectory_", "")
            print(f"[续训] 发现最新 checkpoint: {latest}")
            return (
                f"{CKPT_DIR}/{latest}",
                f"{CKPT_DIR}/buffer_{step_tag}",
                f"{CKPT_DIR}/curriculum_{step_tag}.npy",
            )

    return None, None, None


model_ckpt, buffer_ckpt, curriculum_ckpt = find_latest_checkpoint()

if model_ckpt and os.path.exists(model_ckpt + ".zip"):
    print(f"[续训] 加载模型: {model_ckpt}")
    model = SAC.load(model_ckpt, env=env)

    if buffer_ckpt and os.path.exists(buffer_ckpt + ".pkl"):
        print(f"[续训] 加载 replay buffer: {buffer_ckpt}")
        model.load_replay_buffer(buffer_ckpt)
    else:
        print("[续训] 未找到 buffer，从空 buffer 开始")

    if curriculum_ckpt and os.path.exists(curriculum_ckpt):
        state = np.load(curriculum_ckpt)
        raw_env.curriculum_level = int(state[0])
        raw_env.success_count    = int(state[1])
        raw_env.episode_count    = int(state[2])
        print(f"[续训] 课程状态: 等级={raw_env.curriculum_level} "
              f"连续成功={raw_env.success_count} episode={raw_env.episode_count}")

    steps_done      = model.num_timesteps
    steps_remaining = max(TOTAL_STEPS - steps_done, 0)
    print(f"[续训] 已训练 {steps_done:,} 步，剩余 {steps_remaining:,} 步")

else:
    print("[新训练] 未发现存档，从头开始")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1,
        tensorboard_log="./tb_logs/",
    )
    steps_remaining = TOTAL_STEPS

# -----------------------------------------------------------------------
# 回调组合
# -----------------------------------------------------------------------
callbacks = [
    SaveAllCallback(save_freq=SAVE_FREQ, ckpt_dir=CKPT_DIR, raw_env=raw_env),
    InterruptCallback(model_dir=MODEL_DIR, raw_env=raw_env),
]

# -----------------------------------------------------------------------
# 训练
# -----------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"训练目标:   {TOTAL_STEPS:,} 步")
print(f"本次训练:   {steps_remaining:,} 步")
print(f"保存频率:   每 {SAVE_FREQ:,} 步")
print(f"课程等级:   {raw_env.curriculum_level} / {len(raw_env.level_params)-1}")
print(f"按 Ctrl+C 可随时中断并保存")
print(f"{'='*50}\n")

if steps_remaining > 0:
    model.learn(
        total_timesteps=steps_remaining,
        callback=callbacks,
        reset_num_timesteps=False,  # 续训时保持步数计数
        progress_bar=True,
    )

# 训练结束，保存最终结果
model.save(MODEL_PATH)
model.save_replay_buffer(BUFFER_PATH)
np.save(CURRICULUM_PATH, np.array([
    raw_env.curriculum_level,
    raw_env.success_count,
    raw_env.episode_count,
]))

print(f"\n训练完成！")
print(f"模型:   {MODEL_PATH}.zip")
print(f"Buffer: {BUFFER_PATH}.pkl")
print(f"最终课程等级: {raw_env.curriculum_level}")