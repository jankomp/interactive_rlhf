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
from src.imitation.util.custom_envs import hopper_v4_1, walker2d_v4_1, swimmer_v4_1, half_cheetah_v4_1, ant_v4_1, reacher_v4_1, inverted_pendulum_v4_1, inverted_double_pendulum_v4_1
from gymnasium.wrappers import RecordVideo

def continue_training(name_prefix, range_end, additional_timesteps):
    for i in range(range_end):
        name = name_prefix + str(i).zfill(2)
        print(f"Training model {name}")

        # BEGIN: PARAMETERS
        logs_folder = 'case_study'
        tb_log_dir = logs_folder + '/tb_logs'
        tb_log_name = name
        environment_number = 1 # integer from 0 to 7
        gravity = -5
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
            gravity=gravity,
            env_make_kwargs=env_make_kwargs,
        )

        # Load the policy model
        agent = PPO.load(logs_folder + '/' + name + '_policy_model_' + chosen_environment_short_name)
        agent.tensorboard_log = tb_log_dir

        # Load the preference model
        reward_net_members = [BasicRewardNet(venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm) for _ in range(3)]
        reward_net = RewardEnsemble(venv.observation_space, venv.action_space, reward_net_members)
        preference_model = preference_comparisons.PreferenceModel.load_model(logs_folder + "/" + name + "_preference_model_" + chosen_environment_short_name, reward_net)

        # Set up the trajectory generator
        trajectory_generator = preference_comparisons.AgentTrainer(
            algorithm=agent,
            reward_fn=reward_net,
            venv=venv,
            rng=np.random.default_rng(0),
            exploration_frac=0.25,
        )

        # Train the model for additional timesteps
        trajectory_generator.train(additional_timesteps, tb_log_name=tb_log_name)

        print(f"Training completed for an additional {additional_timesteps} timesteps")

        # Save the trained models
        agent.save(logs_folder + '/' + name + '_policy_model_' + chosen_environment_short_name)
        print("Model saved as " + tb_log_name + f"_policy_model_{chosen_environment}")

        preference_model.save_model(logs_folder + "/" + name + "_preference_model_" + chosen_environment_short_name)
        print("Model saved as " + tb_log_name + f"_preference_model_{chosen_environment}")


        # Create the environment
        env = gym.make(chosen_environment, render_mode='rgb_array', max_episode_steps=1000, terminate_when_unhealthy=False)
        env.model.opt.gravity[2] = gravity
        env = RecordVideo(env, logs_folder + '/videos', name_prefix=tb_log_name + '_' + chosen_environment_short_name, episode_trigger=lambda x: x % 1 == 0) 
        # Run the model in the environment
        obs, info = env.reset()
        for _ in range(1000):
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, _ ,done, info = env.step(action)
                if done:
                    obs, info = env.reset()


continue_training('pairwise_', 10, 2_000_000)
continue_training('groupwise_', 10, 2_000_000)