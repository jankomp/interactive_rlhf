import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.imitation.util.custom_envs import hopper_v4_1

from imitation.rewards.reward_wrapper import RewardVecEnvWrapper, WrappedRewardCallback

def evaluate_policy(policy, venv, n_eval_episodes=10):
    rewards = []
    for _ in range(n_eval_episodes):
        obs = venv.reset()
        done = [False] * venv.num_envs
        episode_reward = 0
        has_been_unhealthy = np.zeros(venv.num_envs, dtype=bool)
        while not all(done):
            action, _ = policy.predict(obs)
            obs, reward, done, info = venv.step(action)
            is_healthy = np.array([i.get('is_healthy', True) for i in info])
            has_been_unhealthy |= ~is_healthy
            healthy_reward = reward * (not has_been_unhealthy)
            episode_reward += healthy_reward
        rewards.append(episode_reward)
    return np.mean(rewards), np.std(rewards)

# Define a simple reward function
def reward_fn(obs, act, next_obs, dones):
    return np.ones((len(obs),))

# Create the environment
env = gym.make('Hopper-v4.1', render_mode='rgb_array', terminate_when_unhealthy=False, max_episode_steps=1000)

# Vectorize the environment
venv = DummyVecEnv([lambda: env])

# Wrap the environment
wrapped_venv = RewardVecEnvWrapper(venv, reward_fn)

# Create the callback
#callback = wrapped_venv.make_log_callback()

# Load the pre-trained model
model_path = 'feedback_logs/pilot_study/pairwise_comparison_02_policy_model_Hopper.zip'
agent = PPO.load(model_path, env=wrapped_venv)


# Evaluate the policy
n_eval_episodes = 10
reward_mean, reward_std = evaluate_policy(agent, venv, n_eval_episodes)
print(f"Reward: {reward_mean:.0f} +/- {reward_std:.0f}")