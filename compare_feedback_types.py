import random
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet, RewardEnsemble
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.regularization.regularizers import LpRegularizer
from imitation.regularization.updaters import IntervalParamScaler
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from imitation.util import logger
import stable_baselines3.common.logger as sb_logger


rng = np.random.default_rng(0)
def intantiate_and_train(pairwise, tb_log_name):
    # make sure that max_episode_steps is divisible by fragment_length
    max_episode_steps = 2000
    fragment_length = 25
    gravity = -9.81
    std_dev = 0.0 # irrationality
    final_training_timesteps = 400_000

    venv = make_vec_env("Hopper-v4", rng=rng, render_mode='rgb_array', n_envs=8, max_episode_steps=max_episode_steps, env_make_kwargs={'terminate_when_unhealthy': False}, gravity=gravity)

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
    if pairwise:
        base_fragmenter = preference_comparisons.RandomFragmenter(
            warning_threshold=0,
            rng=rng,
        )
        fragmenter = preference_comparisons.ActiveSelectionFragmenter(
                preference_model,
                base_fragmenter,
                2.0,
        )
        gatherer = preference_comparisons.SyntheticGatherer(rng=rng, std_dev=std_dev)
    else:
        fragmenter = preference_comparisons.AbsoluteUncertaintyFragmenter(
            preference_model,
            2.0,
            rng=rng,
        )
        gatherer = preference_comparisons.SyntheticGathererForGroupComparisons(rng=rng, augment_to_group_size=1, use_active_learning=True, std_dev=std_dev, preference_model=preference_model)
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
        tensorboard_log="./compare_feedback_types_perfect/",
    )

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        rng=rng,
        exploration_frac=0.05,
    )

    default_logger = sb_logger.Logger(folder='/logs', output_formats='stdout,log,csv,tensorboard')
    custom_logger = logger.HierarchicalLogger(default_logger=default_logger)

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=9,  # Set to 60 for better performance
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=fragment_length,
        transition_oversampling=1,
        initial_comparison_frac=0.1,
        allow_variable_horizon=False,
        initial_epoch_multiplier=4,
        query_schedule="constant",
        custom_logger=custom_logger,
    )

    pref_comparisons.train(
        total_timesteps=800_000,
        total_comparisons=1000,
        tb_log_name=tb_log_name,
    )

    print(f"Training the learner for {final_training_timesteps} timesteps")
    trajectory_generator.train(final_training_timesteps, tb_log_name=tb_log_name)  # Note: set to 100_000 to train a proficient expert


for i in range(10):
    print(f"Group comparison {i}")
    intantiate_and_train(False, f"groupwise_{i}")

for i in range(10):
    print(f"Pairwise comparison {i}")
    intantiate_and_train(True, f"pairwise_{i}")