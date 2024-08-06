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

# BEGIN: PARAMETERS
total_timesteps = 100_000
total_comparisons = 500
max_episode_steps = 200 # make sure that max_episode_steps is divisible by fragment_length
fragment_length = 25 # make sure that max_episode_steps is divisible by fragment_length
every_n_frames = 3 # when to record a frame
gravity = -9.81
environment_number = 1 # integer from 0 to 7
final_training_timesteps = 100_000
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
# reward_trainer = preference_comparisons.BasicRewardTrainer(
#     preference_model=preference_model,
#     loss=preference_comparisons.CrossEntropyRewardLoss(),
#     epochs=3,
#     rng=rng,
# )


# Create a lambda updater
scaling_factor = 0.1
tolerable_interval = (0.9, 1.1) 
lambda_updater = IntervalParamScaler(scaling_factor, tolerable_interval)
# Create a RegularizerFactory
regularizer_factory = LpRegularizer.create(initial_lambda=0.1, lambda_updater=lambda_updater, p=2, val_split=0.1)

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
gatherer = preference_comparisons.HumanGathererForGroupComparisonsAPI(total_feedbacks=total_comparisons, rng=rng, augment_to_group_size=1, preference_model=preference_model,)

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
)

# Create the logger
default_logger = Logger('logs', output_formats=['stdout'])
hierarchical_logger = HierarchicalLogger(default_logger)


trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=agent,
    reward_fn=reward_net,
    venv=venv,
    rng=rng,
    exploration_frac=0.2,
    #video_folder='videos',
    #video_length=fragment_length,
    #name_prefix='rl-video',
    #timeline=True,
    custom_logger=hierarchical_logger,
)


pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=10,  # Set to 60 for better performance
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=fragment_length,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=4,
    query_schedule="constant",
    tb_log_name='groupwise_comparison',
)

pref_comparisons.train(
    total_timesteps=total_timesteps,
    total_comparisons=total_comparisons,
)

from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)

learner = PPO(
    seed=0,
    policy=FeedForward32Policy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=learned_reward_venv,
    batch_size=64,
    ent_coef=0.01,
    n_epochs=10,
    n_steps=2048 // learned_reward_venv.num_envs,
    clip_range=0.1,
    gae_lambda=0.95,
    gamma=0.97,
    learning_rate=2e-3,
)
print(f"Training the learner for {final_training_timesteps} timesteps")
learner.learn(final_training_timesteps)  # Note: set to 100_000 to train a proficient expert

from stable_baselines3.common.evaluation import evaluate_policy

n_eval_episodes = 100
reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes)
reward_stderr = reward_std / np.sqrt(n_eval_episodes)
print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")

learner.save('rlhf_group_wise' + chosen_environment_short_name)

from gymnasium.wrappers import RecordVideo

# Create the environment
env = gym.make(chosen_environment, render_mode='rgb_array', max_episode_steps=1000, terminate_when_unhealthy=False)
env.model.opt.gravity[2] = gravity
env = RecordVideo(env, './evaluation_videos', name_prefix=chosen_environment_short_name, episode_trigger=lambda x: x % 1 == 0) 
# Run the model in the environment
obs, info = env.reset()
for _ in range(1000):
        action, _states = learner.predict(obs, deterministic=True)
        obs, reward, _ ,done, info = env.step(action)
        if done:
            obs, info = env.reset()
            

env.close()
gatherer.close()