"""
Continue training from checkpoint for Super Mario Bros PPO agent
"""

import os
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mario import make_mario_env

if __name__ == "__main__":
    checkpoint_path = r"results\ppo\exp1\models\checkpoints\mario_PPO_400000_steps.zip"

    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")

    # Random environment with all 32 stages (default: World 1-8, all 4 stages per world)
    train_env = make_mario_env(
        "SuperMarioBros-1-1-v3",
        n_envs=4,
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": True,
            "noop_max": 80,
        },
        vec_normalize_kwargs={
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 50.0,
            "gamma": 0.99,
        },
        monitor_dir=f"{log_dir}/train",
    )
    vec_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_path):
        train_env = VecNormalize.load(vec_path, train_env)
    else:
        print(f"Warning: {vec_path} not found. Using new VecNormalize.")
    train_env.training = True
    model = PPO.load(checkpoint_path, env=train_env)


    model.batch_size = 512

    model.tensorboard_log = os.path.join(log_dir, "tensorboard")

    checkpoint_callback = CheckpointCallback(
        save_freq=12500,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    model.learn(
        total_timesteps=1e7,
        callback=[checkpoint_callback],
        tb_log_name="mario_PPO",
        progress_bar=True,
        reset_num_timesteps=False,
    )

    train_env.close()
