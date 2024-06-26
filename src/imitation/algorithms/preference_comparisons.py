"""Learning reward models using preference comparisons.

Trains a reward model and optionally a policy based on preferences
between trajectory fragments.
"""
import os
import random
import string
import pygame
import cv2
import abc
import math
import pickle
import re
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from queue import Queue
import numpy as np
import torch as th
from scipy import special
from stable_baselines3.common import base_class, type_aliases, utils, vec_env
from torch import nn
from torch.utils import data as data_th
from tqdm.auto import tqdm

from imitation.algorithms import base
from imitation.data import rollout, types, wrappers
from imitation.data.types import (
    AnyPath,
    TrajectoryPair,
    TrajectoryWithRew,
    TrajectoryWithRewPair,
    Transitions,
    find_video_file,
)
from imitation.policies import exploration_wrapper
from imitation.regularization import regularizers
from imitation.rewards import reward_function, reward_nets, reward_wrapper
from imitation.util import logger as imit_logger
from imitation.util import networks, util
from imitation.util import vec_video_recorder

from IPython.display import HTML, display, clear_output
import ipywidgets as widgets
from functools import wraps
from flask import Flask, Response, request, jsonify, send_file
import time
import json
from flask_cors import CORS
from threading import Thread

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from scipy.spatial.distance import pdist, squareform

from scipy.spatial.distance import euclidean

class TrajectoryGenerator(abc.ABC):
    """Generator of trajectories with optional training logic."""

    _logger: imit_logger.HierarchicalLogger
    """Object to log statistics and natural language messages to."""

    def __init__(self, custom_logger: Optional[imit_logger.HierarchicalLogger] = None):
        """Builds TrajectoryGenerator.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:
        """Sample a batch of trajectories.

        Args:
            steps: All trajectories taken together should
                have at least this many steps.

        Returns:
            A list of sampled trajectories with rewards (which should
            be the environment rewards, not ones from a reward model).
        """  # noqa: DAR202

    def train(self, steps: int, **kwargs: Any) -> None:
        """Train an agent if the trajectory generator uses one.

        By default, this method does nothing and doesn't need
        to be overridden in subclasses that don't require training.

        Args:
            steps: number of environment steps to train for.
            **kwargs: additional keyword arguments to pass on to
                the training procedure.
        """

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return self._logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        self._logger = value


class TrajectoryDataset(TrajectoryGenerator):
    """A fixed dataset of trajectories."""

    def __init__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        rng: np.random.Generator,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Creates a dataset loaded from `path`.

        Args:
            trajectories: the dataset of rollouts.
            rng: RNG used for shuffling dataset.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)
        self._trajectories = trajectories
        self.rng = rng

    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:
        # make a copy before shuffling
        trajectories = list(self._trajectories)
        # NumPy's annotation here is overly-conservative, but this works at runtime
        self.rng.shuffle(trajectories)  # type: ignore[arg-type]
        return _get_trajectories(trajectories, steps)


class AgentTrainer(TrajectoryGenerator):
    """Wrapper for training an SB3 algorithm on an arbitrary reward function."""

    def __init__(
        self,
        algorithm: base_class.BaseAlgorithm,
        reward_fn: Union[reward_function.RewardFn, reward_nets.RewardNet],
        venv: vec_env.VecEnv,
        rng: np.random.Generator,
        exploration_frac: float = 0.0,
        switch_prob: float = 0.5,
        random_prob: float = 0.5,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initialize the agent trainer.

        Args:
            algorithm: the stable-baselines algorithm to use for training.
            reward_fn: either a RewardFn or a RewardNet instance that will supply
                the rewards used for training the agent.
            venv: vectorized environment to train in.
            rng: random number generator used for exploration and for sampling.
            exploration_frac: fraction of the trajectories that will be generated
                partially randomly rather than only by the agent when sampling.
            switch_prob: the probability of switching the current policy at each
                step for the exploratory samples.
            random_prob: the probability of picking the random policy when switching
                during exploration.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.algorithm = algorithm
        # NOTE: this has to come after setting self.algorithm because super().__init__
        # will set self.logger, which also sets the logger for the algorithm
        super().__init__(custom_logger)
        if isinstance(reward_fn, reward_nets.RewardNet):
            utils.check_for_correct_spaces(
                venv,
                reward_fn.observation_space,
                reward_fn.action_space,
            )
            reward_fn = reward_fn.predict_processed
        self.reward_fn = reward_fn
        self.exploration_frac = exploration_frac
        self.rng = rng

        # The BufferingWrapper records all trajectories, so we can return
        # them after training. This should come first (before the wrapper that
        # changes the reward function), so that we return the original environment
        # rewards.
        # When applying BufferingWrapper and RewardVecEnvWrapper, we should use `venv`
        # instead of `algorithm.get_env()` because SB3 may apply some wrappers to
        # `algorithm`'s env under the hood. In particular, in image-based environments,
        # SB3 may move the image-channel dimension in the observation space, making
        # `algorithm.get_env()` not match with `reward_fn`.
        self.buffering_wrapper = wrappers.BufferingWrapper(venv)
        self.venv = self.reward_venv_wrapper = reward_wrapper.RewardVecEnvWrapper(
            self.buffering_wrapper,
            reward_fn=self.reward_fn,
        )

        self.log_callback = self.reward_venv_wrapper.make_log_callback()

        self.algorithm.set_env(self.venv)
        # Unlike with BufferingWrapper, we should use `algorithm.get_env()` instead
        # of `venv` when interacting with `algorithm`.
        algo_venv = self.algorithm.get_env()
        assert algo_venv is not None
        # This wrapper will be used to ensure that rollouts are collected from a mixture
        # of `self.algorithm` and a policy that acts randomly. The samples from
        # `self.algorithm` are themselves stochastic if `self.algorithm` is stochastic.
        # Otherwise, they are deterministic, and action selection is only stochastic
        # when sampling from the random policy.
        self.exploration_wrapper = exploration_wrapper.ExplorationWrapper(
            policy=self.algorithm,
            venv=algo_venv,
            random_prob=random_prob,
            switch_prob=switch_prob,
            rng=self.rng,
        )


    def train(self, steps: int, **kwargs) -> None:
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Raises:
            RuntimeError: Transitions left in `self.buffering_wrapper`; call
                `self.sample` first to clear them.
        """
        n_transitions = self.buffering_wrapper.n_transitions
        if n_transitions:
            raise RuntimeError(
                f"There are {n_transitions} transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )
        self.algorithm.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=self.log_callback,
            **kwargs,
        )

    def sample(self, steps: int) -> Sequence[types.TrajectoryWithRew]:
        agent_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
        # We typically have more trajectories than are needed.
        # In that case, we use the final trajectories because
        # they are the ones with the most relevant version of
        # the agent.
        # The easiest way to do this will be to first invert the
        # list and then later just take the first trajectories:
        agent_trajs = agent_trajs[::-1]
        avail_steps = sum(len(traj) for traj in agent_trajs)

        exploration_steps = int(self.exploration_frac * steps)
        if self.exploration_frac > 0 and exploration_steps == 0:
            self.logger.warn(
                "No exploration steps included: exploration_frac = "
                f"{self.exploration_frac} > 0 but steps={steps} is too small.",
            )
        agent_steps = steps - exploration_steps

        if avail_steps < agent_steps:
            self.logger.log(
                f"Requested {agent_steps} transitions but only {avail_steps} in buffer."
                f" Sampling {agent_steps - avail_steps} additional transitions.",
            )
            sample_until = rollout.make_sample_until(
                min_timesteps=agent_steps - avail_steps,
                min_episodes=None,
            )
            # Important note: we don't want to use the trajectories returned
            # here because 1) they might miss initial timesteps taken by the RL agent
            # and 2) their rewards are the ones provided by the reward model!
            # Instead, we collect the trajectories using the BufferingWrapper.
            algo_venv = self.algorithm.get_env()
            assert algo_venv is not None
            rollout.generate_trajectories(
                self.algorithm,
                algo_venv,
                sample_until=sample_until,
                # By setting deterministic_policy to False, we ensure that the rollouts
                # are collected from a deterministic policy only if self.algorithm is
                # deterministic. If self.algorithm is stochastic, then policy_callable
                # will also be stochastic.
                deterministic_policy=False,
                rng=self.rng,
            )
            additional_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
            agent_trajs = list(agent_trajs) + list(additional_trajs)

        agent_trajs = _get_trajectories(agent_trajs, agent_steps)

        trajectories = list(agent_trajs)

        if exploration_steps > 0:
            self.logger.log(f"Sampling {exploration_steps} exploratory transitions.")
            sample_until = rollout.make_sample_until(
                min_timesteps=exploration_steps,
                min_episodes=None,
            )
            algo_venv = self.algorithm.get_env()
            assert algo_venv is not None
            rollout.generate_trajectories(
                policy=self.exploration_wrapper,
                venv=algo_venv,
                sample_until=sample_until,
                # buffering_wrapper collects rollouts from a non-deterministic policy,
                # so we do that here as well for consistency.
                deterministic_policy=False,
                rng=self.rng,
            )
            exploration_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
            exploration_trajs = _get_trajectories(exploration_trajs, exploration_steps)
            # We call _get_trajectories separately on agent_trajs and exploration_trajs
            # and then just concatenate. This could mean we return slightly too many
            # transitions, but it gets the proportion of exploratory and agent
            # transitions roughly right.
            trajectories.extend(list(exploration_trajs))
        return trajectories

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return super().logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        self._logger = value
        #commented, because we want to save tensorboard logs
        #self.algorithm.set_logger(self.logger)


def _get_trajectories(
    trajectories: Sequence[TrajectoryWithRew],
    steps: int,
) -> Sequence[TrajectoryWithRew]:
    """Get enough trajectories to have at least `steps` transitions in total."""
    if steps == 0:
        return []

    available_steps = sum(len(traj) for traj in trajectories)
    if available_steps < steps:
        raise RuntimeError(
            f"Asked for {steps} transitions but only {available_steps} available",
        )
    # We need the cumulative sum of trajectory lengths
    # to determine how many trajectories to return:
    steps_cumsum = np.cumsum([len(traj) for traj in trajectories])
    # Now we find the first index that gives us enough
    # total steps:
    idx = int((steps_cumsum >= steps).argmax())
    # we need to include the element at position idx
    trajectories = trajectories[: idx + 1]
    # sanity check
    assert sum(len(traj) for traj in trajectories) >= steps
    return trajectories


class AgentTrainerWithVideoBuffering(AgentTrainer):
    def __init__(
        self,
        algorithm: base_class.BaseAlgorithm,
        reward_fn: Union[reward_function.RewardFn, reward_nets.RewardNet],
        venv: vec_env.VecEnv,
        rng: np.random.Generator,
        exploration_frac: float = 0.0,
        switch_prob: float = 0.5,
        random_prob: float = 0.5,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        video_folder: str = 'videos',
        video_length: int = 200,
        record_video_trigger: Optional[Callable[[int], bool]] = None,
        name_prefix: str = 'rl-video',
        timeline: Optional[bool] = False,
    ) -> None:
        """Initialize the agent trainer with video buffering.

        Args:
            algorithm: the stable-baselines algorithm to use for training.
            reward_fn: either a RewardFn or a RewardNet instance that will supply
                the rewards used for training the agent.
            venv: vectorized environment to train in.
            rng: random number generator used for exploration and for sampling.
            exploration_frac: fraction of the trajectories that will be generated
                partially randomly rather than only by the agent when sampling.
            switch_prob: the probability of switching the current policy at each
                step for the exploratory samples.
            random_prob: the probability of picking the random policy when switching
                during exploration.
            custom_logger: Where to log to; if None (default), creates a new logger.
            video_folder: Folder where videos will be saved.
            video_length: Length of each video in steps.
            record_video_trigger: Function to determine when to start recording.
            name_prefix: Prefix for video file names.
        """
        if record_video_trigger is None:
            record_video_trigger = lambda step: step % video_length == 0
        
        self.video_folder = video_folder
        self.video_length = video_length
        self.name_prefix = name_prefix
        
        # Create video folder if it doesn't exist
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        # Wrapping the environment with VecVideoRecorder
        venv = vec_video_recorder.VecVideoRecorder(
            venv,
            video_folder=video_folder,
            record_video_trigger=record_video_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            timeline=timeline,
        )

        super().__init__(
            algorithm=algorithm,
            reward_fn=reward_fn,
            venv=venv,
            rng=rng,
            exploration_frac=exploration_frac,
            switch_prob=switch_prob,
            random_prob=random_prob,
            custom_logger=custom_logger
        )


class PreferenceModel(nn.Module):
    """Class to convert two fragments' rewards into preference probability."""

    def __init__(
        self,
        model: reward_nets.RewardNet,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
    ) -> None:
        """Create Preference Prediction Model.

        Args:
            model: base model to compute reward.
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss).
            discount_factor: the model of preference generation uses a softmax
                of returns as the probability that a fragment is preferred.
                This is the discount factor used to calculate those returns.
                Default is 1, i.e. undiscounted sums of rewards (which is what
                the DRLHP paper uses).
            threshold: the preference model used to compute the loss contains
                a softmax of returns. To avoid overflows, we clip differences
                in returns that are above this threshold. This threshold
                is therefore in logspace. The default value of 50 means
                that probabilities below 2e-22 are rounded up to 2e-22.

        Raises:
            ValueError: if `RewardEnsemble` is wrapped around a class
                other than `AddSTDRewardWrapper`.
        """
        super().__init__()
        self.model = model
        self.noise_prob = noise_prob
        self.discount_factor = discount_factor
        self.threshold = threshold
        base_model = get_base_model(model)
        self.ensemble_model = None
        # if the base model is an ensemble model, then keep the base model as
        # model to get rewards from all networks
        if isinstance(base_model, reward_nets.RewardEnsemble):
            # reward_model may include an AddSTDRewardWrapper for RL training; but we
            # must train directly on the base model for reward model training.
            is_base = model is base_model
            is_std_wrapper = (
                isinstance(model, reward_nets.AddSTDRewardWrapper)
                and model.base is base_model
            )

            if not (is_base or is_std_wrapper):
                raise ValueError(
                    "RewardEnsemble can only be wrapped"
                    f" by AddSTDRewardWrapper but found {type(model).__name__}.",
                )
            self.ensemble_model = base_model
            self.member_pref_models = []
            for member in self.ensemble_model.members:
                member_pref_model = PreferenceModel(
                    cast(reward_nets.RewardNet, member),  # nn.ModuleList is not generic
                    self.noise_prob,
                    self.discount_factor,
                    self.threshold,
                )
                self.member_pref_models.append(member_pref_model)

    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """Computes the preference probability of the first fragment for all pairs.

        Note: This function passes the gradient through for non-ensemble models.
              For an ensemble model, this function should not be used for loss
              calculation. It can be used in case where passing the gradient is not
              required such as during active selection or inference time.
              Therefore, the EnsembleTrainer passes each member network through this
              function instead of passing the EnsembleNetwork object with the use of
              `ensemble_member_index`.

        Args:
            fragment_pairs: batch of pair of fragments.

        Returns:
            A tuple with the first element as the preference probabilities for the
            first fragment for all fragment pairs given by the network(s).
            If the ground truth rewards are available, it also returns gt preference
            probabilities in the second element of the tuple (else None).
            Reward probability shape - (num_fragment_pairs, ) for non-ensemble reward
            network and (num_fragment_pairs, num_networks) for an ensemble of networks.

        """
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        gt_reward_available = _trajectory_pair_includes_reward(fragment_pairs[0])
        if gt_reward_available:
            gt_probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = self.rewards(trans1)
            rews2 = self.rewards(trans2)
            probs[i] = self.probability(rews1, rews2)
            if gt_reward_available:
                frag1 = cast(TrajectoryWithRew, frag1)
                frag2 = cast(TrajectoryWithRew, frag2)
                gt_rews_1 = th.from_numpy(frag1.rews)
                gt_rews_2 = th.from_numpy(frag2.rews)
                gt_probs[i] = self.probability(gt_rews_1, gt_rews_2)

        return probs, (gt_probs if gt_reward_available else None)

    def rewards(self, transitions: Transitions) -> th.Tensor:
        """Computes the reward for all transitions.

        Args:
            transitions: batch of obs-act-obs-done for a fragment of a trajectory.

        Returns:
            The reward given by the network(s) for all the transitions.
            Shape - (num_transitions, ) for Single reward network and
            (num_transitions, num_networks) for ensemble of networks.
        """
        state = types.assert_not_dictobs(transitions.obs)
        action = transitions.acts
        next_state = types.assert_not_dictobs(transitions.next_obs)
        done = transitions.dones
        if self.ensemble_model is not None:
            rews_np = self.ensemble_model.predict_processed_all(
                state,
                action,
                next_state,
                done,
            )
            assert rews_np.shape == (len(state), self.ensemble_model.num_members)
            rews = util.safe_to_tensor(rews_np).to(self.ensemble_model.device)
        else:
            preprocessed = self.model.preprocess(state, action, next_state, done)
            rews = self.model(*preprocessed)
            assert rews.shape == (len(state),)
        return rews

    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        """Computes the Boltzmann rational probability the first trajectory is best.

        Args:
            rews1: array/matrix of rewards for the first trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.
            rews2: array/matrix of rewards for the second trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.

        Returns:
            The softmax of the difference between the (discounted) return of the
            first and second trajectory.
            Shape - (num_ensemble_members, ) for ensemble model and
            () for non-ensemble model which is a torch scalar.
        """
        # check rews has correct shape based on the model
        expected_dims = 2 if self.ensemble_model is not None else 1
        assert rews1.ndim == rews2.ndim == expected_dims
        # First, we compute the difference of the returns of
        # the two fragments. We have a special case for a discount
        # factor of 1 to avoid unnecessary computation (especially
        # since this is the default setting).
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum(axis=0)  # type: ignore[call-overload]
        else:
            device = rews1.device
            assert device == rews2.device
            discounts = self.discount_factor ** th.arange(len(rews1), device=device)
            if self.ensemble_model is not None:
                discounts = discounts.reshape(-1, 1)
            returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        model_probability = 1 / (1 + returns_diff.exp())
        probability = self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        if self.ensemble_model is not None:
            assert probability.shape == (self.model.num_members,)
        else:
            assert probability.shape == ()
        return probability


class Fragmenter(abc.ABC):
    """Class for creating pairs of trajectory fragments from a set of trajectories."""

    def __init__(self, custom_logger: Optional[imit_logger.HierarchicalLogger] = None):
        """Initialize the fragmenter.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        """Create fragment pairs out of a sequence of trajectories.

        Args:
            trajectories: collection of trajectories that will be split up into
                fragments
            fragment_length: the length of each sampled fragment
            num_pairs: the number of fragment pairs to sample

        Returns:
            a sequence of fragment pairs
        """  # noqa: DAR202

class RandomFragmenter(Fragmenter):
    """Sample fragments of trajectories uniformly at random with replacement.

    Note that each fragment is part of a single episode and has a fixed
    length. This leads to a bias: transitions at the beginning and at the
    end of episodes are less likely to occur as part of fragments (this affects
    the first and last fragment_length transitions).

    An additional bias is that trajectories shorter than the desired fragment
    length are never used.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        warning_threshold: int = 10,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initialize the fragmenter.

        Args:
            rng: the random number generator
            warning_threshold: give a warning if the number of available
                transitions is less than this many times the number of
                required samples. Set to 0 to disable this warning.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger)
        self.rng = rng
        self.warning_threshold = warning_threshold

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        fragments: List[TrajectoryWithRew] = []

        prev_num_trajectories = len(trajectories)
        # filter out all trajectories that are too short
        trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
        if len(trajectories) == 0:
            raise ValueError(
                "No trajectories are long enough for the desired fragment length "
                f"of {fragment_length}.",
            )
        num_discarded = prev_num_trajectories - len(trajectories)
        if num_discarded:
            self.logger.log(
                f"Discarded {num_discarded} out of {prev_num_trajectories} "
                "trajectories because they are shorter than the desired length "
                f"of {fragment_length}.",
            )

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times.",
            )
        elif (
            self.warning_threshold
            and sum(weights) < self.warning_threshold * num_transitions
        ):
            # If the number of available transitions is not much larger
            # than the number of requires ones, we already give a warning.
            # But only if self.warning_threshold is non-zero.
            self.logger.warn(
                f"Samples will contain {num_transitions} transitions in total "
                f"and only {sum(weights)} are available. "
                f"Because we sample with replacement, a significant number "
                "of transitions are likely to appear multiple times.",
            )

        # we need two fragments for each comparison
        for _ in range(2 * num_pairs):
            # NumPy's annotation here is overly-conservative, but this works at runtime
            traj = self.rng.choice(
                trajectories,  # type: ignore[arg-type]
                p=np.array(weights) / sum(weights),
            )
            n = len(traj)
            start = self.rng.integers(0, n - fragment_length, endpoint=True)
            end = start + fragment_length
            terminal = (end == n) and traj.terminal
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=terminal,
            )
            fragments.append(fragment)
        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        return list(zip(iterator, iterator))
    
    def get_fragment_pair(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
    ) -> Tuple[TrajectoryWithRew, TrajectoryWithRew]:
        """Get a pair of fragments randomly.

        This method should return a pair of fragments that can be used for preference comparison.
        The specific implementation can vary depending on the fragmenter. For example, in the
        RandomFragmenter, this returns a pair of fragments selected randomly.

        Args:
            trajectories: A sequence of trajectories to sample from.
            fragment_length: The length of each fragment.

        Returns:
            A pair of fragments.
        """
        fragments: List[TrajectoryWithRew] = []

        # filter out all trajectories that are too short
        trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
        if len(trajectories) == 0:
            raise ValueError(
                "No trajectories are long enough for the desired fragment length "
                f"of {fragment_length}.",
            )

        weights = [len(traj) for traj in trajectories]

        # we need two fragments for each comparison
        for _ in range(2):
            traj = self.rng.choice(
                trajectories,  # type: ignore[arg-type]
                p=np.array(weights) / sum(weights),
            )
            n = len(traj)
            start = self.rng.integers(0, n - fragment_length, endpoint=True)
            end = start + fragment_length
            terminal = (end == n) and traj.terminal
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=terminal,
            )
            fragments.append(fragment)
        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        return next(zip(iterator, iterator))    


class ActiveSelectionFragmenter(Fragmenter):
    """Sample fragments of trajectories based on active selection.

    Actively picks the fragment pairs with the highest uncertainty (variance)
    of rewards/probabilties/predictions from ensemble model.
    """

    def __init__(
        self,
        preference_model: PreferenceModel,
        base_fragmenter: Fragmenter,
        fragment_sample_factor: float,
        uncertainty_on: str = "logit",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initialize the active selection fragmenter.

        Args:
            preference_model: an ensemble model that predicts the
                preference of the first fragment over the other.
            base_fragmenter: fragmenter instance to get
                fragment pairs from trajectories
            fragment_sample_factor: the factor of the number of
                fragment pairs to sample from the base_fragmenter
            uncertainty_on: the variable to calculate the variance on.
                Can be logit|probability|label.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: Preference model not wrapped over an ensemble of networks.
        """
        super().__init__(custom_logger=custom_logger)
        if preference_model.ensemble_model is None:
            raise ValueError(
                "PreferenceModel not wrapped over an ensemble of networks.",
            )
        self.preference_model = preference_model
        self.base_fragmenter = base_fragmenter
        self.fragment_sample_factor = fragment_sample_factor
        self._uncertainty_on = uncertainty_on
        if not (uncertainty_on in ["logit", "probability", "label"]):
            self.raise_uncertainty_on_not_supported()

    @property
    def uncertainty_on(self) -> str:
        return self._uncertainty_on

    def raise_uncertainty_on_not_supported(self) -> NoReturn:
        raise ValueError(
            f"""{self.uncertainty_on} not supported.
            `uncertainty_on` should be from `logit`, `probability`, or `label`""",
        )

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        # sample a large number (self.fragment_sample_factor*num_pairs)
        # of fragments from all the trajectories
        fragments_to_sample = int(self.fragment_sample_factor * num_pairs)
        fragment_pairs = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=fragments_to_sample,
        )
        var_estimates = np.zeros(len(fragment_pairs))
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            with th.no_grad():
                rews1 = self.preference_model.rewards(trans1)
                rews2 = self.preference_model.rewards(trans2)
            var_estimate = self.variance_estimate(rews1, rews2)
            var_estimates[i] = var_estimate
        fragment_idxs = np.argsort(var_estimates)[::-1]  # sort in descending order
        # return fragment pairs that have the highest uncertainty
        return [fragment_pairs[idx] for idx in fragment_idxs[:num_pairs]]

    def variance_estimate(self, rews1: th.Tensor, rews2: th.Tensor) -> float:
        """Gets the variance estimate from the rewards of a fragment pair.

        Args:
            rews1: rewards obtained by all the ensemble models for the first fragment.
                Shape - (fragment_length, num_ensemble_members)
            rews2: rewards obtained by all the ensemble models for the second fragment.
                Shape - (fragment_length, num_ensemble_members)

        Returns:
            the variance estimate based on the `uncertainty_on` flag.
        """
        if self.uncertainty_on == "logit":
            returns1, returns2 = rews1.sum(0), rews2.sum(0)
            var_estimate = (returns1 - returns2).var().item()
        else:  # uncertainty_on is probability or label
            probs = self.preference_model.probability(rews1, rews2)
            probs_np = probs.cpu().numpy()
            assert probs_np.shape == (self.preference_model.model.num_members,)
            if self.uncertainty_on == "probability":
                var_estimate = probs_np.var()
            elif self.uncertainty_on == "label":  # uncertainty_on is label
                preds = (probs_np > 0.5).astype(np.float32)
                # probability estimate of Bernoulli random variable
                prob_estimate = preds.mean()
                # variance estimate of Bernoulli random variable
                var_estimate = prob_estimate * (1 - prob_estimate)
            else:
                self.raise_uncertainty_on_not_supported()
        return var_estimate

    def get_fragment_pair(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
    ) -> Tuple[TrajectoryWithRew, TrajectoryWithRew]:
        """Get a pair of fragments with the highest uncertainty.

        This method should return a pair of fragments that can be used for preference comparison.
        The specific implementation can vary depending on the fragmenter. For example, in the
        ActiveSelectionFragmenter, this returns the pair of fragments with the highest
        uncertainty.

        Args:
            trajectories: A sequence of trajectories to sample from.
            fragment_length: The length of each fragment.

        Returns:
            A pair of fragments.
        """
        # Sample a large number of fragments from all the trajectories
        fragments_to_sample = int(self.fragment_sample_factor * 2)  # We need two fragments
        fragment_pairs = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=fragments_to_sample,
        )
        var_estimates = np.zeros(len(fragment_pairs))
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            with th.no_grad():
                rews1 = self.preference_model.rewards(trans1)
                rews2 = self.preference_model.rewards(trans2)
            var_estimate = self.variance_estimate(rews1, rews2)
            var_estimates[i] = var_estimate
        fragment_idxs = np.argsort(var_estimates)[::-1]  # sort in descending order
        # Return the fragment pair that has the highest uncertainty
        return fragment_pairs[fragment_idxs[0]]
    
class AbsoluteUncertaintyFragmenter(Fragmenter):
    """Sample fragments of trajectories based on absolute uncertainty.

    Actively picks the fragments with the highest absolute uncertainty
    of rewards from ensemble model.
    """

    def __init__(
        self,
        preference_model: PreferenceModel,
        fragment_sample_factor: float,
        rng: np.random.Generator,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initialize the absolute uncertainty fragmenter.

        Args:
            preference_model: an ensemble model that predicts the
                preference of the first fragment over the other.
            base_fragmenter: fragmenter instance to get
                fragment pairs from trajectories
            fragment_sample_factor: the factor of the number of
                fragment pairs to sample from the base_fragmenter
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: Preference model not wrapped over an ensemble of networks.
        """
        super().__init__(custom_logger=custom_logger)
        if preference_model.ensemble_model is None:
            raise ValueError(
                "PreferenceModel not wrapped over an ensemble of networks.",
            )
        self.preference_model = preference_model
        self.fragment_sample_factor = fragment_sample_factor
        self.rng = rng

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_fragments: int,
    ) -> Sequence[TrajectoryWithRew]:
        fragments = self.sample_fragments(trajectories, num_fragments, fragment_length)

        var_estimates = np.zeros(len(fragments))
        for i, fragment in enumerate(fragments):
            trans = rollout.flatten_trajectories([fragment])
            with th.no_grad():
                rews = self.preference_model.rewards(trans)
            var_estimate = self.variance_estimate(rews)
            var_estimates[i] = var_estimate
        fragment_idxs = np.argsort(np.abs(var_estimates))[::-1]  # sort in descending order
        # return fragments that have the highest absolute uncertainty
        return [fragments[idx] for idx in fragment_idxs[:num_fragments]]
    
    def sample_fragments(self,
        trajectories: Sequence[TrajectoryWithRew],
        framgents_to_sample: int,
        fragment_length: int
    ) -> Sequence[TrajectoryWithRew]:
        if sum([len(traj) - (len(traj) % fragment_length) for traj in trajectories]) < framgents_to_sample * fragment_length:
            raise ValueError("Not enough data to sample the required number of fragments")
        
        fragments = []
        for traj in trajectories:
            if len(traj) < fragment_length:
                continue
            
            #start = self.rng.integers(0, fragment_length, endpoint=True)
            start = 0 # TODO: fix the saving of the videoclips (make start and end of videoclip coincide with fragment's), then uncomment the above line and remove this line

            #sample the fragments from the trajectory starting from start until the remaining size is smaller than fragment_length
            for i in range(start, len(traj) - start, fragment_length):
                fragment = TrajectoryWithRew(
                    obs=traj.obs[i : i + fragment_length + 1],
                    acts=traj.acts[i : i + fragment_length],
                    infos=traj.infos[i : i + fragment_length] if traj.infos is not None else None,
                    rews=traj.rews[i : i + fragment_length],
                    terminal=traj.terminal,
                )
                fragments.append(fragment)

        return fragments


    def variance_estimate(self, rews: th.Tensor) -> float:
        """Gets the variance estimate from the rewards of a fragment.

        Args:
            rews: rewards obtained by all the ensemble models for the fragment.
                Shape - (fragment_length, num_ensemble_members)

        Returns:
            the variance estimate.
        """
        return rews.var().item()


class JsonFragmenter(Fragmenter):
    def __init__(self, directory: str, custom_logger: Optional[imit_logger.HierarchicalLogger] = None):
        super().__init__(custom_logger)
        self.directory = directory

    def __call__(self, trajectories: Sequence[TrajectoryWithRew], fragment_length: int, num_pairs: int) -> Sequence[TrajectoryWithRew]:
        fragments = trajectories
        return fragments
    
    def _convert_infos(self, infos):
        converted_infos = []
        for info in infos:
            if isinstance(info, dict):
                converted_info = {k: self._convert_value(v) for k, v in info.items()}
            else:
                converted_info = self._convert_value(info)
            converted_infos.append(converted_info)
        return converted_infos

    def _convert_value(self, value):
        if isinstance(value, np.float32):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return [self._convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        else:
            return value

    def save(self, fragments: Sequence[TrajectoryWithRew], filename: str):
        # Convert fragments to a serializable format
        serializable_fragments = [
            {
                "obs": fragment.obs.tolist(),
                "acts": fragment.acts.tolist(),
                "infos": self._convert_infos(fragment.infos),
                "rews": fragment.rews.tolist(),
                "terminal": fragment.terminal
            }
            for fragment in fragments
        ]

        path = os.path.join(self.directory, filename)

        with open(path, 'w') as f:
            json.dump(serializable_fragments, f)

    def _revert_infos(self, infos):
        reverted_infos = []
        for info in infos:
            if isinstance(info, dict):
                reverted_info = {k: self._revert_value(v) for k, v in info.items()}
            else:
                reverted_info = self._revert_value(info)
            reverted_infos.append(reverted_info)
        return reverted_infos

    def _revert_value(self, value):
        if isinstance(value, list):
            return np.array(value, dtype=np.float32) if all(isinstance(v, float) for v in value) else [self._revert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._revert_value(v) for k, v in value.items()}
        else:
            return value

    def load(self, filename: str) -> Sequence[TrajectoryWithRew]:
        path = os.path.join(self.directory, filename)

        with open(path, 'r') as f:
            serializable_fragments = json.load(f)

        # Convert fragments back to the original format
        fragments = [
            TrajectoryWithRew(
                obs=np.array(fragment["obs"]),
                acts=np.array(fragment["acts"]),
                infos=self._revert_infos(fragment["infos"]),
                rews=np.array(fragment["rews"]),
                terminal=self._revert_value(fragment["terminal"])
            )
            for fragment in serializable_fragments
        ]

        return fragments

class PreferenceGatherer(abc.ABC):
    """Base class for gathering preference comparisons between trajectory fragments."""

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initializes the preference gatherer.

        Args:
            rng: random number generator, if applicable.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        # The random seed isn't used here, but it's useful to have this
        # as an argument nevertheless because that means we can always
        # pass in a seed in training scripts (without worrying about whether
        # the PreferenceGatherer we use needs one).
        del rng
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Gathers the probabilities that fragment 1 is preferred in `fragment_pairs`.

        Args:
            fragment_pairs: sequence of pairs of trajectory fragments

        Returns:
            A numpy array with shape (b, ), where b is the length of the input
            (i.e. batch size). Each item in the array is the probability that
            fragment 1 is preferred over fragment 2 for the corresponding
            pair of fragments.

            Note that for human feedback, these probabilities are simply 0 or 1
            (or 0.5 in case of indifference), but synthetic models may yield other
            probabilities.
        """  # noqa: DAR202


class SyntheticGatherer(PreferenceGatherer):
    """Computes synthetic preferences using ground-truth environment rewards."""

    def __init__(
        self,
        temperature: float = 1,
        discount_factor: float = 1,
        sample: bool = True,
        rng: Optional[np.random.Generator] = None,
        threshold: float = 50,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initialize the synthetic preference gatherer.

        Args:
            temperature: the preferences are sampled from a softmax, this is
                the temperature used for sampling. temperature=0 leads to deterministic
                results (for equal rewards, 0.5 will be returned).
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            sample: if True (default), the preferences are 0 or 1, sampled from
                a Bernoulli distribution (or 0.5 in the case of ties with zero
                temperature). If False, then the underlying Bernoulli probabilities
                are returned instead.
            rng: random number generator, only used if
                ``temperature > 0`` and ``sample=True``
            threshold: preferences are sampled from a softmax of returns.
                To avoid overflows, we clip differences in returns that are
                above this threshold (after multiplying with temperature).
                This threshold is therefore in logspace. The default value
                of 50 means that probabilities below 2e-22 are rounded up to 2e-22.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: if `sample` is true and no random state is provided.
        """
        super().__init__(custom_logger=custom_logger)
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = rng
        self.threshold = threshold

        if self.sample and self.rng is None:
            raise ValueError("If `sample` is True, then `rng` must be provided.")

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Computes probability fragment 1 is preferred over fragment 2."""
        returns1, returns2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            return (np.sign(returns1 - returns2) + 1) / 2

        returns1 /= self.temperature
        returns2 /= self.temperature

        # clip the returns to avoid overflows in the softmax below
        returns_diff = np.clip(returns2 - returns1, -self.threshold, self.threshold)
        # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
        # we divide enumerator and denominator by exp(rews1) to prevent overflows:
        model_probs = 1 / (1 + np.exp(returns_diff))
        # Compute the mean binary entropy. This metric helps estimate
        # how good we can expect the performance of the learned reward
        # model to be at predicting preferences.
        entropy = -(
            special.xlogy(model_probs, model_probs)
            + special.xlogy(1 - model_probs, 1 - model_probs)
        ).mean()
        self.logger.record("entropy", entropy)

        if self.sample:
            assert self.rng is not None
            return self.rng.binomial(n=1, p=model_probs).astype(np.float32)
        return model_probs

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        rews1, rews2 = zip(
            *[
                (
                    rollout.discounted_sum(f1.rews, self.discount_factor),
                    rollout.discounted_sum(f2.rews, self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)

class HumanGathererAPI(PreferenceGatherer):
    """Collects human feedback by displaying videos and capturing keyboard inputs."""

    def __init__(
        self,
        total_feedbacks: int,
        fragmenter: Fragmenter,
        rng: Optional[np.random.Generator] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        super().__init__(rng=rng, custom_logger=custom_logger)
        self.window_name = "Trajectory Comparison"
        self.app = Flask(__name__)
        CORS(self.app)
        self.videos = ['','']
        self.queue = Queue()
        self.feedback_count = 1
        self.total_feedbacks = total_feedbacks
        self.fragmenter = fragmenter
        self.app.route('/key_press', methods=['POST'])(self.key_press)
        self.app.route('/stream')(self.stream)
        self.app.route('/videos/<path:filename>')(self.serve_video)
        self.app.route('/total_feedbacks')(self.get_total_feedbacks)
        print('Starting server in a new thread')
        Thread(target=self.app.run, kwargs={'host': '0.0.0.0', 'debug': True, 'use_reloader': False, 'threaded': True}).start()

    def send_videos(self):
        yield 'data: {}\n\n'.format(json.dumps(self.videos))

    def get_total_feedbacks(self):
        return jsonify({'total_feedbacks': self.total_feedbacks})

    def key_press(self):
        key = request.json.get('key')
        if self.queue.empty():
            self.queue.put(key)
        return jsonify({'message': 'Success', 'feedback_count': self.feedback_count})
    
    def stream(self):
        return Response(self.send_videos(), mimetype='text/event-stream')

    def serve_video(self, filename):
        if not os.path.isabs(filename):
            filename = os.path.join('/', filename)
        return send_file(filename)

    def display_videos(self, video_path1: str, video_path2: str) -> None:
        self.videos = [video_path1, video_path2]
        self.send_videos()        

    def get_human_feedback(self) -> float:
        key = self.queue.get()
        if key == 'ArrowLeft':
            return 1.0
        elif key == 'ArrowRight':
            return 0.0
        elif key == 'ArrowUp':
            return 0.5
        elif key == 'ArrowDown':
            return None
        else:
            print("Invalid input. Please press 'Left' for left video, 'Right' for right video, 'Up' for tie, 'Down' to skip.")

    def __call__(self, trajectories: Sequence[TrajectoryWithRew], fragment_length: int, num_pairs: int) -> Tuple[List[TrajectoryWithRewPair], np.ndarray]:
        """Gather human preferences for the given fragment pairs."""
        preferences = []
        fragment_pairs = []
        while len(preferences) < num_pairs:
            fragment_pair = self.fragmenter.get_fragment_pair(trajectories, fragment_length)
            video_path1 = find_video_file(fragment_pair[0].infos)
            video_path2 = find_video_file(fragment_pair[1].infos)
            if video_path1 is None or video_path2 is None:
                self.logger.log(f"Skipping this pair {len(preferences)} because one of the video_paths is None. Frag1 path: {video_path1}, Frag2 path: {video_path2}")
                continue
            self.display_videos(video_path1, video_path2)
            feedback = self.get_human_feedback()
            clear_output(wait=True)
            if feedback is None:
                self.logger.log(f"Skipping this comparison. Frag1 path: {video_path1}, Frag2 path: {video_path2}")
                continue
            self.feedback_count += 1
            preferences.append(feedback)
            fragment_pairs.append(fragment_pair)
        self.display_videos('','')

        return fragment_pairs, np.array(preferences, dtype=np.float32)
    
class SyntheticGathererForGroupComparisons(PreferenceGatherer):
    """Collects synthetic feedback by displaying the fragments in a dimensionally reduced scatterplot and receiving preferences over groups."""

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        augment_to_group_size = 10,
    ) -> None:
        super().__init__(rng=rng, custom_logger=custom_logger)
        self.augment_to_group_size = augment_to_group_size
    
    def convert_to_low_dimensional_data(self, high_dimensional_data, fragment_length, n_trajectory_components, reduce_to):
        def dtw_distance(t1, t2):
            # https://pypi.org/project/fastdtw/
            # FastDTW: Toward accurate dynamic time warping in linear time and space.” Intelligent Data Analysis 
            # x = np.array(x)
            # y = np.array(y)
            t1 = t1.reshape((fragment_length, n_trajectory_components))
            t2 = t2.reshape((fragment_length, n_trajectory_components))
            distance, _ = fastdtw(t1, t2, dist=euclidean)
            return distance

        tsne = TSNE(n_components=reduce_to, perplexity=4, metric=dtw_distance)
        low_dimensional_data = tsne.fit_transform(high_dimensional_data)
        return low_dimensional_data
    
    def dimensional_reduction(self, fragments: Sequence[TrajectoryWithRew], fragment_length) -> np.ndarray:
        """Reduce the dimensionality of the fragments. Return id, x, y, video_path of the fragments"""
        n_components = 2
        n_trajectory_components = len(fragments[0].obs[0]) + len(fragments[0].acts[0])
        # Convert fragments to fragments_data
        fragments_data = []
        for fragment in fragments:
            fragment_data = np.concatenate([np.array(list(fragment.obs[i]) + list(fragment.acts[i])) for i in range(fragment_length)])
            fragments_data.append(fragment_data)
        fragments_data = np.array(fragments_data)

        low_dimensional_data = self.convert_to_low_dimensional_data(fragments_data, fragment_length, n_trajectory_components, n_components)

        fragments_with_id = [{'id': id, 'x': float(x), 'y': float(y), 'reward': fragment.rews.sum()} 
                         for id, (x, y), fragment in zip(range(len(fragments)), low_dimensional_data, fragments)]

        return fragments_with_id
    


    def get_group_preferences(self, dimensionally_reduced_fragments, num_pairs: int):
        
        group_preferences = []
        trajectories_in_comparisons = 0
        X = np.array([[fragment['x'], fragment['y']] for fragment in dimensionally_reduced_fragments])

        # Normalize the data
        X = StandardScaler().fit_transform(X)

        eps = 0.15
        min_samples = 3
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Split the fragments into groups based on the cluster labels
        groups = [[] for _ in range(n_clusters_)]
        for i, label in enumerate(labels):
            if label != -1:  # Ignore noise
                groups[label].append(dimensionally_reduced_fragments[i])

        print(f"DBSCAN found {n_clusters_} clusters. {len(groups)} groups were created.")

        #TODO: for now the noise is ignored. should we treat each noise fragment as a separate group instead?
        
        completed_group_comparisons = 0
        # Generate group comparisons
        while trajectories_in_comparisons < num_pairs:
            # Pick two random groups
            group1 = random.choice(groups)
            group2 = random.choice(groups)
            while group1 == group2:
                group2 = random.choice(groups)

            print(f"Before cleaning: Group 1 has {len(group1)} fragments, Group 2 has {len(group2)} fragments")
            # Calculate the average true rewards for each group
            true_reward_group1 = sum(fragment['reward'] for fragment in group1) / len(group1)
            true_reward_group2 = sum(fragment['reward'] for fragment in group2) / len(group2)
            print(f"Group 1 has an average reward of {true_reward_group1}, Group 2 has an average reward of {true_reward_group2}")

            groups_reward_border = (true_reward_group1 + true_reward_group2) / 2
            print(f"Groups reward border is {groups_reward_border}")

            # discard the members of the unpreferred group with rewards that are larger than groups_reward_border, and members of the preferred group with rewards that are smaller than groups_reward_border
            if true_reward_group1 < true_reward_group2:
                group1 = [fragment['id'] for fragment in group1 if fragment['reward'] < groups_reward_border]
                group2 = [fragment['id'] for fragment in group2 if fragment['reward'] > groups_reward_border]
            else:
                group1 = [fragment['id'] for fragment in group1 if fragment['reward'] > groups_reward_border]
                group2 = [fragment['id'] for fragment in group2 if fragment['reward'] < groups_reward_border]

            print(f"Group 1 has {len(group1)} fragments, Group 2 has {len(group2)} fragments")

            # this should never actually happen, if it does there is a bug in the code
            if len(group1) == 0 or len(group2) == 0:
                continue

            # Determine the preference based on the true rewards
            if true_reward_group1 > true_reward_group2:
                preference = 1.0
            elif true_reward_group1 < true_reward_group2:
                preference = 0.0
            else:
                preference = 0.5

            group_preferences.append({
                'group1': group1,
                'group2': group2,
                'preference': preference
            })
            completed_group_comparisons += 1
            trajectories_in_comparisons += len(group1) + len(group2)

        print(f"The system did {completed_group_comparisons} group comparisons.")
        print(f"This amounts to a total of {trajectories_in_comparisons} trajectories in comparisons")

        return group_preferences

    def __call__(self, fragments: Sequence[TrajectoryWithRew], fragment_length: int, num_pairs: int) -> Tuple[List[TrajectoryWithRewPair], np.ndarray]:
        """Gather human preferences for the given fragment pairs."""
        fragment_pairs = []
        dimensionally_reduced_fragments = self.dimensional_reduction(fragments, fragment_length=fragment_length)
        
        comparisons_goal = self.augment_to_group_size * self.augment_to_group_size
        preferences = []
        group_preferences = self.get_group_preferences(dimensionally_reduced_fragments, num_pairs)
        for group_preference in group_preferences:
            group1 = group_preference['group1']
            group2 = group_preference['group2']
            preference = group_preference['preference']

            # If the product of the group sizes is less than 100, augment the smaller group
            while len(group1) * len(group2) < comparisons_goal:
                if len(group1) < len(group2):
                    group1 += random.choices(group1, k=1)
                else:
                    group2 += random.choices(group2, k=1)

            for i in group1:
                for j in group2:
                    fragment_pair = (fragments[i], fragments[j])
                    fragment_pairs.append(fragment_pair)
                    preferences.append(preference)
            
        print(f"Generated {len(preferences)} preferences")
        return fragment_pairs, np.array(preferences, dtype=np.float32)
    
class HumanGathererForGroupComparisonsAPI(PreferenceGatherer):
    """Collects human feedback by displaying the fragments in a dimensionally reduced scatterplot, displaying videos and receiving preferences over groups."""

    def __init__(
        self,
        total_feedbacks: int,
        rng: Optional[np.random.Generator] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        augment_to_group_size = 10,
    ) -> None:
        super().__init__(rng=rng, custom_logger=custom_logger)
        self.augment_to_group_size = augment_to_group_size
        self.app = Flask(__name__)
        CORS(self.app)
        self.feedback_count = 0
        self.total_feedbacks = total_feedbacks
        self.current_fragments_hash = None
        self.fragments_with_id = []
        self.fragments_for_frontend = []
        self.queue = Queue() 
        self.app.route('/stream')(self.stream)
        self.app.route('/videos/<path:filename>')(self.serve_video)
        self.app.route('/total_feedbacks')(self.get_total_feedbacks)
        self.app.route('/fragments')(self.get_fragments)
        self.app.route('/preference', methods=['POST'])(self.post_preference_pairs)
        print('Starting server in a new thread')
        Thread(target=self.app.run, kwargs={'host': '0.0.0.0', 'debug': True, 'use_reloader': False, 'threaded': True}).start()
    
    def get_total_feedbacks(self):
        return jsonify({'given_feedbacks': self.feedback_count, 'total_feedbacks': self.total_feedbacks})
    
    def send_fragment_hash(self):
        yield 'data: {}\n\n'.format(json.dumps(self.current_fragments_hash))

    def stream(self):
        return Response(self.send_fragment_hash(), mimetype='text/event-stream')

    def serve_video(self, filename):
        if not os.path.isabs(filename):
            filename = os.path.join('/', filename)
        return send_file(filename)
    
    def get_fragments(self):
        return jsonify(self.fragments_for_frontend)
    
    def post_preference_pairs(self):
        data = request.json
        self.queue.put(data)
        self.feedback_count += (10 if len(data['group1']) < 10 else len(data['group1'])) + (10 if len(data['group2']) < 10 else len(data['group2'])) # replace + with * if we want to count the number of pairs instead of the number of fragments
        return jsonify(self.feedback_count)

    def hierarchical_clustering(self, fragments: Sequence[TrajectoryWithRew], fragment_length) -> np.ndarray:
        n_trajectory_components = len(fragments[0].obs[0]) + len(fragments[0].acts[0])
        fragments_data = []
        for fragment in fragments:
            fragment_data = np.concatenate([np.array(list(fragment.obs[i]) + list(fragment.acts[i])) for i in range(fragment_length)])
            fragments_data.append(fragment_data)
        fragments_data = np.array(fragments_data)

        def dtw_distance(t1, t2):
            t1 = t1.reshape((fragment_length, n_trajectory_components))
            t2 = t2.reshape((fragment_length, n_trajectory_components))
            distance, _ = fastdtw(t1, t2, dist=euclidean)
            return distance

        # Compute the distance matrix using DTW
        dist_matrix = squareform(pdist(fragments_data, dtw_distance))

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='complete', compute_distances=True).fit(dist_matrix)

        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        self.plot_dendrogram(clustering, truncate_mode=None, p=3)        
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig('dendrogram.png')

        # Convert the children_ attribute to a tree structure
        tree = self.children_to_tree(clustering.children_, fragments)

        return tree


    def children_to_tree(self, children, fragments):
        # Create a node for each fragment
        nodes = [{"id": i, "video_path": find_video_file(fragment.infos)} for i, fragment in enumerate(fragments)]

        # Create a node for each non-leaf node
        for i, (child1, child2) in enumerate(children):
            node = {"id": len(nodes), "children": [nodes[child1], nodes[child2]]}
            nodes.append(node)

        # The root of the tree is the last node
        root = nodes[-1]

        return root
    
    def plot_dendrogram(self, model, **kwargs):
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    def __call__(self, fragments: Sequence[TrajectoryWithRew], fragment_length: int, num_pairs: int) -> Tuple[List[TrajectoryWithRewPair], np.ndarray]:
        """Gather human preferences for the given fragment pairs."""
        fragment_pairs = []
        preferences = []
        self.fragments_for_frontend = self.hierarchical_clustering(fragments, fragment_length=fragment_length)
        #save fragments_for_frontend to a json_file
        print('Saving fragments_for_frontend to a json file')
        with open('fragments_for_frontend.json', 'w') as f:
            json.dump(self.fragments_for_frontend, f)
        print('saved to json')
        self.current_fragments_hash = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        self.feedback_count = 0

        comparisons_goal = self.augment_to_group_size * self.augment_to_group_size

        while self.feedback_count < num_pairs:
            feedback = self.queue.get()
            # the feedback contains two sequences of indices and a preference (1.0, 0.5, or 0.0)
            # we create one preference for each possible pair of fragments across the two groups with the value of preference
            preference = 1.0 if feedback['preference'] == 'ArrowLeft' else 0.0 if feedback['preference'] == 'ArrowRight' else 0.5 if feedback['preference'] == 'ArrowUp' else None
            if preference is not None:
                group1 = feedback['group1']
                group2 = feedback['group2']

                # If the product of the group sizes is less than comparison_goal, augment the smaller group
                while len(group1) * len(group2) < comparisons_goal:
                    if len(group1) < len(group2):
                        group1 += random.choices(group1, k=1)
                    else:
                        group2 += random.choices(group2, k=1)

                for i in group1:
                    for j in group2:
                        fragment_pair = (fragments[i], fragments[j])
                        fragment_pairs.append(fragment_pair)
                        preferences.append(preference)
            print(f'Feedback count: {self.feedback_count}/{num_pairs}')
            print(f'Preferences: {len(preferences)}')
            print(f'Fragment pairs: {len(fragment_pairs)}')

        self.current_fragments_hash = None
        return fragment_pairs, np.array(preferences, dtype=np.float32)


class PreferenceDataset(data_th.Dataset):
    """A PyTorch Dataset for preference comparisons.

    Each item is a tuple consisting of two trajectory fragments
    and a probability that fragment 1 is preferred over fragment 2.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.
    """

    def __init__(self, max_size: Optional[int] = None) -> None:
        """Builds an empty PreferenceDataset.

        Args:
            max_size: Maximum number of preference comparisons to store in the dataset.
                If None (default), the dataset can grow indefinitely. Otherwise, the
                dataset acts as a FIFO queue, and the oldest comparisons are evicted
                when `push()` is called and the dataset is at max capacity.
        """
        self.fragments1: List[TrajectoryWithRew] = []
        self.fragments2: List[TrajectoryWithRew] = []
        self.max_size = max_size
        self.preferences: np.ndarray = np.array([])

    def push(
        self,
        fragments: Sequence[TrajectoryWithRewPair],
        preferences: np.ndarray,
    ) -> None:
        """Add more samples to the dataset.

        Args:
            fragments: list of pairs of trajectory fragments to add
            preferences: corresponding preference probabilities (probability
                that fragment 1 is preferred over fragment 2)

        Raises:
            ValueError: `preferences` shape does not match `fragments` or
                has non-float32 dtype.
        """
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments),):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments),)}",
            )
        if preferences.dtype != np.float32:
            raise ValueError("preferences should have dtype float32")

        self.fragments1.extend(fragments1)
        self.fragments2.extend(fragments2)
        self.preferences = np.concatenate((self.preferences, preferences))

        # Evict old samples if the dataset is at max capacity
        if self.max_size is not None:
            extra = len(self.preferences) - self.max_size
            if extra > 0:
                self.fragments1 = self.fragments1[extra:]
                self.fragments2 = self.fragments2[extra:]
                self.preferences = self.preferences[extra:]

    @overload
    def __getitem__(self, key: int) -> Tuple[TrajectoryWithRewPair, float]:
        pass

    @overload
    def __getitem__(
        self,
        key: slice,
    ) -> Tuple[types.Pair[Sequence[TrajectoryWithRew]], Sequence[float]]:
        pass

    def __getitem__(self, key):
        return (self.fragments1[key], self.fragments2[key]), self.preferences[key]

    def __len__(self) -> int:
        assert len(self.fragments1) == len(self.fragments2) == len(self.preferences)
        return len(self.fragments1)

    def save(self, path: AnyPath) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: AnyPath) -> "PreferenceDataset":
        with open(path, "rb") as file:
            return pickle.load(file)


def preference_collate_fn(
    batch: Sequence[Tuple[TrajectoryWithRewPair, float]],
) -> Tuple[Sequence[TrajectoryWithRewPair], np.ndarray]:
    fragment_pairs, preferences = zip(*batch)
    return list(fragment_pairs), np.array(preferences)


class LossAndMetrics(NamedTuple):
    """Loss and auxiliary metrics for reward network training."""

    loss: th.Tensor
    metrics: Mapping[str, th.Tensor]


class RewardLoss(nn.Module, abc.ABC):
    """A loss function over preferences."""

    @abc.abstractmethod
    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
        preference_model: PreferenceModel,
    ) -> LossAndMetrics:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).
            preference_model: model to predict the preferred fragment from a pair.

        Returns: # noqa: DAR202
            loss: the loss
            metrics: a dictionary of metrics that can be logged
        """


def _trajectory_pair_includes_reward(fragment_pair: TrajectoryPair) -> bool:
    """Return true if and only if both fragments in the pair include rewards."""
    frag1, frag2 = fragment_pair
    return isinstance(frag1, TrajectoryWithRew) and isinstance(frag2, TrajectoryWithRew)


class CrossEntropyRewardLoss(RewardLoss):
    """Compute the cross entropy reward loss."""

    def __init__(self) -> None:
        """Create cross entropy reward loss."""
        super().__init__()

    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
        preference_model: PreferenceModel,
    ) -> LossAndMetrics:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).
            preference_model: model to predict the preferred fragment from a pair.

        Returns:
            The cross-entropy loss between the probability predicted by the
                reward model and the target probabilities in `preferences`. Metrics
                are accuracy, and gt_reward_loss, if the ground truth reward is
                available.
        """
        probs, gt_probs = preference_model(fragment_pairs)
        # TODO(ejnnr): Here and below, > 0.5 is problematic
        #  because getting exactly 0.5 is actually somewhat
        #  common in some environments (as long as sample=False or temperature=0).
        #  In a sense that "only" creates class imbalance
        #  but it's still misleading.
        predictions = probs > 0.5
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = preferences_th > 0.5
        metrics = {}
        metrics["accuracy"] = (predictions == ground_truth).float().mean()
        if gt_probs is not None:
            metrics["gt_reward_loss"] = th.nn.functional.binary_cross_entropy(
                gt_probs,
                preferences_th,
            )
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}
        return LossAndMetrics(
            loss=th.nn.functional.binary_cross_entropy(probs, preferences_th),
            metrics=metrics,
        )


class RewardTrainer(abc.ABC):
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see :class: `PreferenceComparisons` for that).
    """

    def __init__(
        self,
        preference_model: PreferenceModel,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initialize the reward trainer.

        Args:
            preference_model: the preference model to train the reward network.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self._preference_model = preference_model
        self._logger = custom_logger or imit_logger.configure()

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return self._logger

    @logger.setter
    def logger(self, custom_logger: imit_logger.HierarchicalLogger) -> None:
        self._logger = custom_logger

    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Train the reward model on a batch of fragment pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        with networks.training(self._preference_model.model):
            self._train(dataset, epoch_multiplier)

    @abc.abstractmethod
    def _train(self, dataset: PreferenceDataset, epoch_multiplier: float) -> None:
        """Train the reward model; see ``train`` for details."""


class BasicRewardTrainer(RewardTrainer):
    """Train a basic reward model."""

    regularizer: Optional[regularizers.Regularizer]

    def __init__(
        self,
        preference_model: PreferenceModel,
        loss: RewardLoss,
        rng: np.random.Generator,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        epochs: int = 1,
        lr: float = 1e-3,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        regularizer_factory: Optional[regularizers.RegularizerFactory] = None,
    ) -> None:
        """Initialize the reward model trainer.

        Args:
            preference_model: the preference model to train the reward network.
            loss: the loss to use
            rng: the random number generator to use for splitting the dataset into
                training and validation.
            batch_size: number of fragment pairs per batch
            minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until `batch_size` examples
                are processed before making an optimization step. This
                is useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `batch_size`.
                Optional, defaults to `batch_size`.
            epochs: number of epochs in each training iteration (can be adjusted
                on the fly by specifying an `epoch_multiplier` in `self.train()`
                if longer training is desired in specific cases).
            lr: the learning rate
            custom_logger: Where to log to; if None (default), creates a new logger.
            regularizer_factory: if you would like to apply regularization during
                training, specify a regularizer factory here. The factory will be
                used to construct a regularizer. See
                ``imitation.regularization.RegularizerFactory`` for more details.

        Raises:
            ValueError: if the batch size is not a multiple of the minibatch size.
        """
        super().__init__(preference_model, custom_logger)
        self.loss = loss
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        self.epochs = epochs
        self.optim = th.optim.AdamW(self._preference_model.parameters(), lr=lr)
        self.rng = rng
        self.regularizer = (
            regularizer_factory(optimizer=self.optim, logger=self.logger)
            if regularizer_factory is not None
            else None
        )

    def _make_data_loader(self, dataset: data_th.Dataset) -> data_th.DataLoader:
        """Make a dataloader."""
        return data_th.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            collate_fn=preference_collate_fn,
        )

    @property
    def requires_regularizer_update(self) -> bool:
        """Whether the regularizer requires updating.

        Returns:
            If true, this means that a validation dataset will be used.
        """
        return self.regularizer is not None and self.regularizer.val_split is not None

    def _train(
        self,
        dataset: PreferenceDataset,
        epoch_multiplier: float = 1.0,
    ) -> None:
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        if self.regularizer is not None and self.regularizer.val_split is not None:
            val_length = int(len(dataset) * self.regularizer.val_split)
            train_length = len(dataset) - val_length
            if val_length < 1 or train_length < 1:
                raise ValueError(
                    "Not enough data samples to split into training and validation, "
                    "or the validation split is too large/small. "
                    "Make sure you've generated enough initial preference data. "
                    "You can adjust this through initial_comparison_frac in "
                    "PreferenceComparisons.",
                )
            train_dataset, val_dataset = data_th.random_split(
                dataset,
                lengths=[train_length, val_length],
                # we convert the numpy generator to the pytorch generator.
                generator=th.Generator().manual_seed(util.make_seeds(self.rng)),
            )
            dataloader = self._make_data_loader(train_dataset)
            val_dataloader = self._make_data_loader(val_dataset)
        else:
            dataloader = self._make_data_loader(dataset)
            val_dataloader = None

        epochs = round(self.epochs * epoch_multiplier)

        assert epochs > 0, "Must train for at least one epoch."
        with self.logger.accumulate_means("reward"):
            for epoch_num in tqdm(range(epochs), desc="Training reward model"):
                with self.logger.add_key_prefix(f"epoch-{epoch_num}"):
                    train_loss = 0.0
                    accumulated_size = 0
                    self.optim.zero_grad()
                    for fragment_pairs, preferences in dataloader:
                        with self.logger.add_key_prefix("train"):
                            loss = self._training_inner_loop(
                                fragment_pairs,
                                preferences,
                            )

                            # Renormalise the loss to be averaged over
                            # the whole batch size instead of the
                            # minibatch size. If there is an incomplete
                            # batch, its gradients will be smaller,
                            # which may be helpful for stability.
                            loss *= len(fragment_pairs) / self.batch_size

                        train_loss += loss.item()
                        if self.regularizer:
                            self.regularizer.regularize_and_backward(loss)
                        else:
                            loss.backward()

                        accumulated_size += len(fragment_pairs)
                        if accumulated_size >= self.batch_size:
                            self.optim.step()
                            self.optim.zero_grad()
                            accumulated_size = 0
                    if accumulated_size != 0:
                        self.optim.step()  # if there remains an incomplete batch

                    if not self.requires_regularizer_update:
                        continue
                    assert val_dataloader is not None
                    assert self.regularizer is not None

                    val_loss = 0.0
                    for fragment_pairs, preferences in val_dataloader:
                        with self.logger.add_key_prefix("val"):
                            val_loss += self._training_inner_loop(
                                fragment_pairs,
                                preferences,
                            ).item()
                    self.regularizer.update_params(train_loss, val_loss)

        # after training all the epochs,
        # record also the final value in a separate key for easy access.
        keys = list(self.logger.name_to_value.keys())
        outer_prefix = self.logger.get_accumulate_prefixes()
        for key in keys:
            base_path = f"{outer_prefix}reward/"  # existing prefix + accum_means ctx
            epoch_path = f"mean/{base_path}epoch-{epoch_num}/"  # mean for last epoch
            final_path = f"{base_path}final/"  # path to record last epoch
            pattern = rf"{epoch_path}(.+)"
            if regex_match := re.match(pattern, key):
                (key_name,) = regex_match.groups()
                val = self.logger.name_to_value[key]
                new_key = f"{final_path}{key_name}"
                self.logger.record(new_key, val)

    def _training_inner_loop(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> th.Tensor:
        output = self.loss.forward(fragment_pairs, preferences, self._preference_model)
        loss = output.loss
        self.logger.record("loss", loss.item())
        for name, value in output.metrics.items():
            self.logger.record(name, value.item())
        return loss


class EnsembleTrainer(BasicRewardTrainer):
    """Train a reward ensemble."""

    def __init__(
        self,
        preference_model: PreferenceModel,
        loss: RewardLoss,
        rng: np.random.Generator,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        epochs: int = 1,
        lr: float = 1e-3,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        regularizer_factory: Optional[regularizers.RegularizerFactory] = None,
    ) -> None:
        """Initialize the reward model trainer.

        Args:
            preference_model: the preference model to train the reward network.
            loss: the loss to use
            rng: random state for the internal RNG used in bagging
            batch_size: number of fragment pairs per batch
            minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until `batch_size` examples
                are processed before making an optimization step. This
                is useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `batch_size`.
                Optional, defaults to `batch_size`.
            epochs: number of epochs in each training iteration (can be adjusted
                on the fly by specifying an `epoch_multiplier` in `self.train()`
                if longer training is desired in specific cases).
            lr: the learning rate
            custom_logger: Where to log to; if None (default), creates a new logger.
            regularizer_factory: A factory for creating a regularizer. If None,
                no regularization is used.

        Raises:
            TypeError: if model is not a RewardEnsemble.
        """
        if preference_model.ensemble_model is None:
            raise TypeError(
                "PreferenceModel of a RewardEnsemble expected by EnsembleTrainer.",
            )

        super().__init__(
            preference_model,
            loss=loss,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            epochs=epochs,
            lr=lr,
            custom_logger=custom_logger,
            rng=rng,
            regularizer_factory=regularizer_factory,
        )
        self.member_trainers = []
        for member_pref_model in self._preference_model.member_pref_models:
            reward_trainer = BasicRewardTrainer(
                member_pref_model,
                loss=loss,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                epochs=epochs,
                lr=lr,
                custom_logger=self.logger,
                regularizer_factory=regularizer_factory,
                rng=self.rng,
            )
            self.member_trainers.append(reward_trainer)

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return super().logger

    @logger.setter
    def logger(self, custom_logger: imit_logger.HierarchicalLogger) -> None:
        self._logger = custom_logger
        for member_trainer in self.member_trainers:
            member_trainer.logger = custom_logger

    def _train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        sampler = data_th.RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset),
            # we convert the numpy generator to the pytorch generator.
            generator=th.Generator().manual_seed(util.make_seeds(self.rng)),
        )
        for member_idx in range(len(self.member_trainers)):
            # sampler gives new indexes on every call
            bagging_dataset = data_th.Subset(dataset, list(sampler))
            with self.logger.add_accumulate_prefix(f"member-{member_idx}"):
                self.member_trainers[member_idx].train(
                    bagging_dataset,
                    epoch_multiplier=epoch_multiplier,
                )

        # average the metrics across the member models
        metrics = defaultdict(list)
        keys = list(self.logger.name_to_value.keys())
        for key in keys:
            if re.match(r"member-(\d+)/reward/(.+)", key) and "final" in key:
                val = self.logger.name_to_value[key]
                key_list = key.split("/")
                key_list.pop(0)
                metrics["/".join(key_list)].append(val)

        for k, v in metrics.items():
            self.logger.record(k, np.mean(v))
            self.logger.record(k + "_std", np.std(v))


def get_base_model(reward_model: reward_nets.RewardNet) -> reward_nets.RewardNet:
    base_model = reward_model
    while hasattr(base_model, "base"):
        base_model = cast(reward_nets.RewardNet, base_model.base)

    return base_model


def _make_reward_trainer(
    preference_model: PreferenceModel,
    loss: RewardLoss,
    rng: np.random.Generator,
    reward_trainer_kwargs: Optional[Mapping[str, Any]] = None,
) -> RewardTrainer:
    """Construct the correct type of reward trainer for this reward function."""
    if reward_trainer_kwargs is None:
        reward_trainer_kwargs = {}

    if preference_model.ensemble_model is not None:
        return EnsembleTrainer(
            preference_model,
            loss,
            rng=rng,
            **reward_trainer_kwargs,
        )
    else:
        return BasicRewardTrainer(
            preference_model,
            loss=loss,
            rng=rng,
            **reward_trainer_kwargs,
        )


QUERY_SCHEDULES: Dict[str, type_aliases.Schedule] = {
    "constant": lambda t: 1.0,
    "hyperbolic": lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


class PreferenceComparisons(base.BaseImitationAlgorithm):
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        trajectory_generator: TrajectoryGenerator,
        reward_model: reward_nets.RewardNet,
        num_iterations: int,
        fragmenter: Optional[Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
        comparison_queue_size: Optional[int] = None,
        fragment_length: int = 100,
        transition_oversampling: float = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        rng: Optional[np.random.Generator] = None,
        query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
        json_fragmenter: Optional[JsonFragmenter] = None,
    ) -> None:
        """Initialize the preference comparison trainer.

        The loggers of all subcomponents are overridden with the logger used
        by this class.

        Args:
            trajectory_generator: generates trajectories while optionally training
                an RL agent on the learned reward function (can also be a sampler
                from a static dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
            num_iterations: number of times to train the agent against the reward model
                and then train the reward model against newly gathered preferences.
            fragmenter: takes in a set of trajectories and returns pairs of fragments
                for which preferences will be gathered. These fragments could be random,
                or they could be selected more deliberately (active learning).
                Default is a random fragmenter.
            preference_gatherer: how to get preferences between trajectory fragments.
                Default (and currently the only option) is to use synthetic preferences
                based on ground-truth rewards. Human preferences could be implemented
                here in the future.
            reward_trainer: trains the reward model based on pairs of fragments and
                associated preferences. Default is to use the preference model
                and loss function from DRLHP.
            comparison_queue_size: the maximum number of comparisons to keep in the
                queue for training the reward model. If None, the queue will grow
                without bound as new comparisons are added.
            fragment_length: number of timesteps per fragment that is used to elicit
                preferences
            transition_oversampling: factor by which to oversample transitions before
                creating fragments. Since fragments are sampled with replacement,
                this is usually chosen > 1 to avoid having the same transition
                in too many fragments.
            initial_comparison_frac: fraction of the total_comparisons argument
                to train() that will be sampled before the rest of training begins
                (using a randomly initialized agent). This can be used to pretrain the
                reward model before the agent is trained on the learned reward, to
                help avoid irreversibly learning a bad policy from an untrained reward.
                Note that there will often be some additional pretraining comparisons
                since `comparisons_per_iteration` won't exactly divide the total number
                of comparisons. How many such comparisons there are depends
                discontinuously on `total_comparisons` and `comparisons_per_iteration`.
            initial_epoch_multiplier: before agent training begins, train the reward
                model for this many more epochs than usual (on fragments sampled from a
                random agent).
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
            rng: random number generator to use for initializing subcomponents such as
                fragmenter.
                Only used when default components are used; if you instantiate your
                own fragmenter, preference gatherer, etc., you are responsible for
                seeding them!
            query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic"), or
                a function that takes in a float between 0 and 1 inclusive,
                representing a fraction of the total number of timesteps elapsed up to
                some time T, and returns a potentially unnormalized probability
                indicating the fraction of `total_comparisons` that should be queried
                at that iteration. This function will be called `num_iterations` times
                in `__init__()` with values from `np.linspace(0, 1, num_iterations)`
                as input. The outputs will be normalized to sum to 1 and then used to
                apportion the comparisons among the `num_iterations` iterations.

        Raises:
            ValueError: if `query_schedule` is not a valid string or callable.
        """
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        # for keeping track of the global iteration, in case train() is called
        # multiple times
        self._iteration = 0

        self.model = reward_model
        self.rng = rng

        # are any of the optional args that require a rng None?
        has_any_rng_args_none = None in (
            preference_gatherer,
            fragmenter,
            reward_trainer,
        )

        if self.rng is None and has_any_rng_args_none:
            raise ValueError(
                "If you don't provide a random state, you must provide your own "
                "seeded fragmenter, preference gatherer, and reward_trainer. "
                "You can initialize a random state with `np.random.default_rng(seed)`.",
            )
        elif self.rng is not None and not has_any_rng_args_none:
            raise ValueError(
                "If you provide your own fragmenter, preference gatherer, "
                "and reward trainer, you don't need to provide a random state.",
            )

        if reward_trainer is None:
            assert self.rng is not None
            preference_model = PreferenceModel(reward_model)
            loss = CrossEntropyRewardLoss()
            self.reward_trainer = _make_reward_trainer(
                preference_model,
                loss,
                rng=self.rng,
            )
        else:
            self.reward_trainer = reward_trainer

        # If the reward trainer was created in the previous line, we've already passed
        # the correct logger. But if the user created a RewardTrainer themselves and
        # didn't manually set a logger, it would be annoying if a separate one was used.
        self.reward_trainer.logger = self.logger
        self.trajectory_generator = trajectory_generator
        self.trajectory_generator.logger = self.logger
        if fragmenter:
            self.fragmenter = fragmenter
        else:
            assert self.rng is not None
            self.fragmenter = RandomFragmenter(
                custom_logger=self.logger,
                rng=self.rng,
            )
        self.fragmenter.logger = self.logger
        if preference_gatherer:
            self.preference_gatherer = preference_gatherer
        else:
            assert self.rng is not None
            self.preference_gatherer = SyntheticGatherer(
                custom_logger=self.logger,
                rng=self.rng,
            )

        if isinstance(self.preference_gatherer, HumanGathererForGroupComparisonsAPI):
            assert isinstance(self.fragmenter, AbsoluteUncertaintyFragmenter) or isinstance(self.fragmenter, JsonFragmenter)

        if isinstance(self.preference_gatherer, SyntheticGathererForGroupComparisons):
            assert isinstance(self.fragmenter, AbsoluteUncertaintyFragmenter)

        if isinstance(self.preference_gatherer, HumanGathererAPI):
            assert isinstance(self.fragmenter, ActiveSelectionFragmenter)

        self.preference_gatherer.logger = self.logger

        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.num_iterations = num_iterations
        self.transition_oversampling = transition_oversampling
        if callable(query_schedule):
            self.query_schedule = query_schedule
        elif query_schedule in QUERY_SCHEDULES:
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(f"Unknown query schedule: {query_schedule}")

        self.dataset = PreferenceDataset(max_size=comparison_queue_size)
        self.json_fragmenter = json_fragmenter

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: Optional[Callable[[int], None]] = None,
        tb_log_name: Optional[str] = None
    ) -> Mapping[str, Any]:
        """Train the reward model and the policy if applicable.

        Args:
            total_timesteps: number of environment interaction steps
            total_comparisons: number of preferences to gather in total
            callback: callback functions called at the end of each iteration

        Returns:
            A dictionary with final metrics such as loss and accuracy
            of the reward model.
        """
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons -= initial_comparisons

        # Compute the number of comparisons to request at each iteration in advance.
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_comparisons)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"Query schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.num_iterations,
        )
        reward_loss = None
        reward_accuracy = None

        for i, num_pairs in enumerate(schedule):
            ##########################
            # Gather new preferences #
            ##########################
            num_steps = math.ceil(
                self.transition_oversampling * 2 * (num_pairs + 1) * self.fragment_length,
            )
            self.logger.log(
                f"Collecting {2 * num_pairs} fragments ({num_steps} transitions)",
            )
            if not isinstance(self.fragmenter, JsonFragmenter): 
                trajectories = self.trajectory_generator.sample(num_steps)
                # pop the last trajectory (since the video could not be saved correctly)
                trajectories.pop()

                # This assumes there are no fragments missing initial timesteps
                # (but allows for fragments missing terminal timesteps).
                horizons = (len(traj) for traj in trajectories if traj.terminal)
                self._check_fixed_horizon(horizons)
            else:
                trajectories = self.fragmenter.load(f'fragments_{self._iteration}.json')

            #for the pairwise comparison of HumanGathererAPI we need to create one trajectory pair for every user query
            if isinstance(self.preference_gatherer, HumanGathererAPI):
                with self.logger.accumulate_means("preferences"):
                    self.logger.log("Gathering preferences")
                    fragments, preferences = self.preference_gatherer(trajectories, self.fragment_length, num_pairs)
            elif isinstance(self.fragmenter, AbsoluteUncertaintyFragmenter):
                self.logger.log("Creating fragment pairs")
                num_fragments = sum([math.floor(len(traj) / self.fragment_length) for traj in trajectories])
                fragments = self.fragmenter(trajectories, self.fragment_length, num_fragments=num_fragments)
                if self.json_fragmenter is not None:
                    self.json_fragmenter.save(fragments, f'fragments_{self._iteration}.json')
                with self.logger.accumulate_means("preferences"):
                    self.logger.log("Gathering preferences")
                    fragments, preferences = self.preference_gatherer(fragments, fragment_length=self.fragment_length, num_pairs=num_pairs)
            else:            
                self.logger.log("Creating fragment pairs")
                fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)
                with self.logger.accumulate_means("preferences"):
                    self.logger.log("Gathering preferences")
                    fragments, preferences = self.preference_gatherer(fragments, fragment_length=self.fragment_length, num_pairs=num_pairs)

            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} comparisons")

            ##########################
            # Train the reward model #
            ##########################

            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier
            if not isinstance(self.fragmenter, JsonFragmenter):
                self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multiplier)
                base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
                assert f"{base_key}/loss" in self.logger.name_to_value
                assert f"{base_key}/accuracy" in self.logger.name_to_value
                reward_loss = self.logger.name_to_value[f"{base_key}/loss"]
                reward_accuracy = self.logger.name_to_value[f"{base_key}/accuracy"]

            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration
            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps
            with self.logger.accumulate_means("agent"):
                if not isinstance(self.fragmenter, JsonFragmenter):
                    self.logger.log(f"Training agent for {num_steps} timesteps")
                    self.trajectory_generator.train(steps=num_steps, tb_log_name=tb_log_name)

            self.logger.dump(self._iteration)

            ########################
            # Additional Callbacks #
            ########################
            if callback:
                callback(self._iteration)
            self._iteration += 1

        return {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}
