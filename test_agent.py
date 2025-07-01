from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from dino_env import DinoEnv
import numpy as np


dummy_env = DummyVecEnv([lambda: DinoEnv(render_mode="human")])


env = VecNormalize.load("dino_vec_normalize.pkl", dummy_env)
env.training = False
env.norm_reward = False  


model = PPO.load("dino_agent", env=env)


obs = env.reset()
done = False  

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    env.envs[0].clock.tick(30)
    env.render()

print("Episode finished.")
