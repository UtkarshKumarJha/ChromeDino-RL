import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from dino_env import DinoEnv


SEED = 42
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


train_env = DummyVecEnv([lambda: DinoEnv(render_mode=None)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)


eval_env = DummyVecEnv([lambda: DinoEnv(render_mode=None)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_env.training = False


checkpoint_callback = CheckpointCallback(
    save_freq=100_000,            # Save every 100k steps
    save_path="./checkpoints",
    name_prefix="ppo_dino"
)

eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="./best_model",
    log_path="./logs",
    eval_freq=20_000,
    deterministic=True,
    render=False,
)


model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./tensorboard",
    verbose=1,
    seed=SEED
)


model.learn(
    total_timesteps=2_000_000,
    callback=[checkpoint_callback, eval_callback]
)


model.save("dino_agent")
train_env.save("dino_vec_normalize.pkl")
