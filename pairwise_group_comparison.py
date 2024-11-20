import random
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet, RewardEnsemble
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.regularization.regularizers import LpRegularizer
from imitation.regularization.updaters import IntervalParamScaler
from stable_baselines3.common.logger import Logger
from imitation.util.logger import HierarchicalLogger
from imitation.util.video_wrapper import VideoWrapper
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import torch.optim as optim
from src.imitation.util.custom_envs import hopper_v4_1, walker2d_v4_1, swimmer_v4_1, half_cheetah_v4_1, ant_v4_1, reacher_v4_1, inverted_pendulum_v4_1, inverted_double_pendulum_v4_1
from imitation.util import logger
import stable_baselines3.common.logger as sb_logger

# BEGIN: PARAMETERS
total_timesteps = 35_000
total_comparisons = 200
rounds = 7
initial_comparison_frac = 1 / (rounds + 1) # We want to keep all the comparison rounds constant
max_episode_steps = 300 # make sure that max_episode_steps is divisible by fragment_length
fragment_length = 50 # make sure that max_episode_steps is divisible by fragment_length
every_n_frames = 3 # when to record a frame
gravity = -5
environment_number = 1 # integer from 0 to 7
#final_training_timesteps = 100_000
logs_folder = 'case_study'
tb_log_name = 'groupwise_comparison_00'
# END: PARAMETERS

environments = ['Walker2d-v4.1', 'Hopper-v4.1', 'Swimmer-v4.1', 'HalfCheetah-v4.1', 'Ant-v4.1', 'Reacher-v4.1', 'InvertedPendulum-v4.1', 'InvertedDoublePendulum-v4.1']
chosen_environment = environments[environment_number]
chosen_environment_short_name = chosen_environment.split('-v')[0]
print(f"Chosen environment: {chosen_environment_short_name}")
env_make_kwargs = {'terminate_when_unhealthy': False}

# some environments need a higher framerate
if chosen_environment in [3, 4, 6, 7]:
     every_n_frames = 1

rng = np.random.default_rng(0)

# we need a videoWrapper only for the first environment, since the VideoRecorder can't be used in parallel environments
def video_recorder_wrapper(env: gym.Env, i: int) -> gym.Env:
    if i == 0:
        return VideoWrapper(
            env,
            directory='videos',
            record_video_trigger = lambda step: step % fragment_length == 0,
            video_length=fragment_length,
            name_prefix=f'rl-video-env-{i}',
            timeline=True,
            every_nth_timestep=every_n_frames,
        )
    else:
        return env

venv = make_vec_env(
    chosen_environment,
    rng=rng,
    render_mode='rgb_array',
    n_envs=8,
    max_episode_steps=max_episode_steps,
    env_make_kwargs=env_make_kwargs,
    gravity=gravity,
    post_wrappers=[video_recorder_wrapper],
)

reward_net_members = [BasicRewardNet(venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm) for _ in range(3)]
reward_net = RewardEnsemble(venv.observation_space, venv.action_space, reward_net_members)

preference_model = preference_comparisons.PreferenceModel(reward_net)
#preference_model = preference_comparisons.PreferenceModel.load_model(logs_folder + "/" + tb_log_name + f"_preference_model_{chosen_environment_short_name}", reward_net)


# Create a lambda updater
scaling_factor = 0.1
tolerable_interval = (0.8, 1.0) 
lambda_updater = IntervalParamScaler(scaling_factor, tolerable_interval)
# Create a RegularizerFactory
regularizer_factory = LpRegularizer.create(initial_lambda=0.2, lambda_updater=lambda_updater, p=2, val_split=0.1)

reward_trainer = preference_comparisons.EnsembleTrainer(
    preference_model,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    rng=rng,
    epochs=5,
    batch_size = 4,
    minibatch_size = 2,
    # lr: float = 1e-3,
    # custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    regularizer_factory = regularizer_factory,
)

#fragmenter = preference_comparisons.JsonFragmenter(directory='fragmenter_data')
fragmenter = preference_comparisons.AbsoluteUncertaintyFragmenter(
        preference_model,
        2.0,
        rng=rng,
)

#gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
gatherer = preference_comparisons.HumanGathererForGroupComparisonsAPI(rng=rng, augment_to_group_size=1, preference_model=preference_model, timed = False)

# Several hyperparameters (reward_epochs, ppo_clip_range, ppo_ent_coef,
# ppo_gae_lambda, ppo_n_epochs, discount_factor, use_sde, sde_sample_freq,
# ppo_lr, exploration_frac, num_iterations, initial_comparison_frac,
# initial_epoch_multiplier, query_schedule) used in this example have been
# approximately fine-tuned to reach a reasonable level of performance.
agent = PPO(
    policy=FeedForward32Policy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=venv,
    seed=0,
    n_steps=2048 // venv.num_envs,
    batch_size=64,
    ent_coef=0.01,
    learning_rate=2e-3,
    clip_range=0.1,
    gae_lambda=0.95,
    gamma=0.97,
    n_epochs=10,
    tensorboard_log=logs_folder + "/tb_logs",
)
#agent = PPO.load(logs_folder + '/' + tb_log_name + '_policy_model_' + chosen_environment_short_name)

trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=agent,
    reward_fn=reward_net,
    venv=venv,
    rng=rng,
    exploration_frac=0.4,
)


default_logger = sb_logger.Logger(folder=logs_folder, output_formats='stdout,log,csv,tensorboard')
custom_logger = logger.HierarchicalLogger(default_logger=default_logger)

feedback_logger = preference_comparisons.FeedbackLogger(logs_folder, tb_log_name + '.csv')

pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=rounds,
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=fragment_length,
    transition_oversampling=1.5,
    initial_comparison_frac=initial_comparison_frac,
    allow_variable_horizon=False,
    initial_epoch_multiplier=4,
    query_schedule="constant",
    #custom_logger=custom_logger,
    feedback_logger=feedback_logger,
    #preference_dataset_name=logs_folder + '/' + tb_log_name + '_preference_dataset.pkl',
)

pref_comparisons.train(
    total_timesteps=total_timesteps,
    total_comparisons=total_comparisons,
    tb_log_name=tb_log_name,
)

from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)


#print(f"Training the learner for {final_training_timesteps} timesteps")
#trajectory_generator.train(final_training_timesteps, tb_log_name=tb_log_name)

from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_policy_healthy_reward(policy, venv, n_eval_episodes=10):
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
            healthy_reward = reward * np.logical_not(has_been_unhealthy)
            episode_reward += healthy_reward
        rewards.append(episode_reward)
    return np.mean(rewards), np.std(rewards)

learner = trajectory_generator.algorithm
n_eval_episodes = 100
reward_mean, reward_std = evaluate_policy_healthy_reward(learner.policy, venv, n_eval_episodes)
reward_stderr = reward_std / np.sqrt(n_eval_episodes)
print(f"Only healthy reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")

reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes)
reward_stderr = reward_std / np.sqrt(n_eval_episodes)
print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")

learner.save(logs_folder + '/' + tb_log_name + '_policy_model_' + chosen_environment_short_name)
print("Model saved as " + tb_log_name + f" policy_model_{chosen_environment_short_name}")

preference_model.save_model(logs_folder + "/" + tb_log_name + f"_preference_model_{chosen_environment_short_name}")
print("Model saved as " + tb_log_name + f"_preference_model_{chosen_environment_short_name}")

from gymnasium.wrappers import RecordVideo

# Create the environment
env = gym.make(chosen_environment, render_mode='rgb_array', max_episode_steps=1000, terminate_when_unhealthy=False)
env.model.opt.gravity[2] = gravity
env = RecordVideo(env, logs_folder + '/videos', name_prefix=tb_log_name + '_' + chosen_environment_short_name, episode_trigger=lambda x: x % 1 == 0) 
# Run the model in the environment
obs, info = env.reset()
for _ in range(1000):
        action, _states = learner.predict(obs, deterministic=True)
        obs, reward, _ ,done, info = env.step(action)
        if done:
            obs, info = env.reset()
            

env.close()
#gatherer.close()