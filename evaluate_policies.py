from stable_baselines3 import PPO
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet, RewardEnsemble
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.regularization.regularizers import LpRegularizer
from imitation.regularization.updaters import IntervalParamScaler
from imitation.util.video_wrapper import VideoWrapper
import gymnasium as gym
import numpy as np
import stable_baselines3.common.logger as sb_logger
from imitation.util import logger
from stable_baselines3.common.evaluation import evaluate_policy
from src.imitation.util.custom_envs import hopper_v4_1, walker2d_v4_1, swimmer_v4_1, half_cheetah_v4_1, ant_v4_1, reacher_v4_1, inverted_pendulum_v4_1, inverted_double_pendulum_v4_1

def evaluate_policies(name_prefix, range_end):
    for i in range(range_end):
        name = name_prefix + str(i).zfill(2)
        print(f"Evaluating policy {name}")

        # BEGIN: PARAMETERS
        logs_folder = 'user_study'
        n_eval_episodes = 100
        environment_number = 1 # integer from 0 to 7
        # END: PARAMETERS

        rng = np.random.default_rng(0)

        environments = ['Walker2d-v4.1', 'Hopper-v4.1', 'Swimmer-v4.1', 'HalfCheetah-v4.1', 'Ant-v4.1', 'Reacher-v4.1', 'InvertedPendulum-v4.1', 'InvertedDoublePendulum-v4.1']
        chosen_environment = environments[environment_number]
        chosen_environment_short_name = chosen_environment.split('-v')[0]
        print(f"Chosen environment: {chosen_environment_short_name}")
        env_make_kwargs = {'terminate_when_unhealthy': False}

        # Set up the environment
        venv = make_vec_env(
            chosen_environment,
            rng=rng,
            n_envs=8,
            max_episode_steps=1000,
            gravity=-9.81,
            env_make_kwargs=env_make_kwargs,
        )

        # Load the policy model
        agent = PPO.load(logs_folder + '/' + name + '_policy_model_' + chosen_environment_short_name)
        
    reward_mean, reward_std = evaluate_policies(agent.policy, venv, n_eval_episodes)
    reward_stderr = reward_std / np.sqrt(n_eval_episodes)
    print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")



evaluate_policies('pairwise_comparison_', 5, 2_000_000)
evaluate_policies('groupwise_comparison_', 5, 2_000_000)