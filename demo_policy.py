
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# load learner from ppo .hopper_backflip.zip
learner = PPO.load("rlhf_hopper")

# Create the environment
env = gym.make("Hopper-v4", render_mode='rgb_array', max_episode_steps=2000, terminate_when_unhealthy=False)
env.model.opt.gravity[2] = -3
env = RecordVideo(env, './evaluation_videos', name_prefix="hopper", episode_trigger=lambda x: x % 1 == 0) 
# Run the model in the environment
obs, info = env.reset()
for _ in range(10000):
        action, _states = learner.predict(obs, deterministic=True)
        obs, reward, _ ,done, info = env.step(action)
        if done:
            obs, info = env.reset()
            

env.close()