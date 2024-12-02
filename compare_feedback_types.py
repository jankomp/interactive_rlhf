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
from src.imitation.util.custom_envs import hopper_v4_1, walker2d_v4_1, swimmer_v4_1, half_cheetah_v4_1, ant_v4_1, reacher_v4_1, inverted_pendulum_v4_1, inverted_double_pendulum_v4_1, grid_world


rng = np.random.default_rng(0)
def intantiate_and_train(pairwise, logs_folder_top, tb_log_name, total_comparisons, rounds, std_dev, environment_number):
    # make sure that max_episode_steps is divisible by fragment_length
    total_timesteps = 45_000
    max_episode_steps = 1000
    fragment_length = 25
    gravity = -9.81
    if environment_number == 0:
        gravity = None
    final_training_timesteps = 100_000
    environments = ['GridWorld-v0.1', 'HalfCheetah-v4.1', 'Reacher-v4.1', 'Walker2d-v4.1', 'MountainCarContinuous-v0']
    chosen_environment = environments[environment_number]
    chosen_environment_short_name = chosen_environment.split('-v')[0]
    tb_log_name = tb_log_name + '_' + chosen_environment_short_name
    print(f"Chosen environment: {chosen_environment_short_name}")
    env_make_kwargs = {'terminate_when_unhealthy': False}
    if environment_number == 0 or environment_number == 2 or environment_number == 3 or environment_number == 4:
        env_make_kwargs = {}

    logs_folder = logs_folder_top + '/' + chosen_environment_short_name

    venv = make_vec_env(
        chosen_environment,
        rng=rng,
        render_mode='rgb_array',
        n_envs=8,
        max_episode_steps=max_episode_steps,
        env_make_kwargs=env_make_kwargs,
        gravity=gravity,
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
        clustering_levels = 4
        if chosen_environment_short_name == 'HalfCheetah' or chosen_environment_short_name == 'Pusher':
            print('Clustering the levels differently because HalfCheetah is more complex')
            gatherer = preference_comparisons.SyntheticGathererForGroupComparisons(rng=rng, augment_to_group_size=1, use_active_learning=True, std_dev=std_dev, preference_model=preference_model, clustering_levels=clustering_levels, constant_tree_level_size=False)
        else:        
            gatherer = preference_comparisons.SyntheticGathererForGroupComparisons(rng=rng, augment_to_group_size=1, use_active_learning=True, std_dev=std_dev, preference_model=preference_model, clustering_levels=clustering_levels)
     
    # Several hyperparameters (reward_epochs, ppo_clip_range, ppo_ent_coef,
    # ppo_gae_lambda, ppo_n_epochs, discount_factor, use_sde, sde_sample_freq,
    # ppo_lr, exploration_frac, num_iterations, initial_comparison_frac,
    # initial_epoch_multiplier, query_schedule) used in this example have been
    # approximately fine-tuned to reach a reasonable level of performance.
    ent_coef = 0.01
    if chosen_environment_short_name == 'Swimmer':
        ent_coef = 0.001
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
        ent_coef=ent_coef,
        learning_rate=2e-3,
        clip_range=0.1,
        gae_lambda=0.95,
        gamma=0.97,
        n_epochs=10,
        tensorboard_log= logs_folder + "/tb_logs/",
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

    feedback_logger = preference_comparisons.FeedbackLogger(logs_folder, tb_log_name + '.csv')

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=rounds,  # Set to 60 for better performance
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=fragment_length,
        transition_oversampling=1,
        initial_comparison_frac=0.25,
        allow_variable_horizon=False,
        initial_epoch_multiplier=4,
        query_schedule="hyperbolic",
        custom_logger=custom_logger,
        feedback_logger=feedback_logger,
    )

    pref_comparisons.train(
        total_timesteps=total_timesteps,
        total_comparisons=total_comparisons,
        tb_log_name=tb_log_name,
    )

    print(f"Training the learner for {final_training_timesteps} timesteps")
    trajectory_generator.train(final_training_timesteps, tb_log_name=tb_log_name)  # Note: set to 100_000 to train a proficient expert

    policy_model = trajectory_generator.algorithm
    policy_model.save(logs_folder + '/' + tb_log_name + '_policy_model_' + chosen_environment_short_name)
    print("Model saved as " + tb_log_name + f" policy_model_{chosen_environment_short_name}")

    preference_model.save_model(logs_folder + "/" + tb_log_name + f"_preference_model_{chosen_environment_short_name}")
    print("Model saved as " + tb_log_name + f"_preference_model_{chosen_environment_short_name}")


for environment_no in [4, 1]:
    for i in range(5):
        print(f"Pairwise comparison {i}")
        intantiate_and_train(True, 'New_envs', f"pairwise_{i}", 500, 9, 0.25, environment_no)
        print(f"Group comparison {i}")
        intantiate_and_train(False, 'New_envs', f"groupwise_{i}", 500, 9, 0.25, environment_no)