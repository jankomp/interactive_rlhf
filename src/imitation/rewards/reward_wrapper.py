"""Common wrapper for adding custom reward values to an environment."""

import collections
from typing import Deque

import numpy as np
from stable_baselines3.common import callbacks
from stable_baselines3.common import logger as sb_logger
from stable_baselines3.common import vec_env

from imitation.data import types
from imitation.rewards import reward_function


class WrappedRewardCallback(callbacks.BaseCallback):
    """Logs mean wrapped reward as part of RL (or other) training."""

    def __init__(self, episode_rewards: Deque[float], episode_healthy_rewards: Deque[float],*args, **kwargs):
        """Builds WrappedRewardCallback.

        Args:
            episode_rewards: A queue that episode rewards will be placed into.
            *args: Passed through to `callbacks.BaseCallback`.
            **kwargs: Passed through to `callbacks.BaseCallback`.
        """
        self.episode_rewards = episode_rewards
        self.episode_healthy_rewards = episode_healthy_rewards
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if len(self.episode_rewards) == 0:
            return
        mean = sum(self.episode_rewards) / len(self.episode_rewards)
        healthy_mean = sum(self.episode_healthy_rewards) / len(self.episode_healthy_rewards)
        assert isinstance(self.logger, sb_logger.Logger)
        self.logger.record("rollout/ep_rew_wrapped_mean", mean)
        self.logger.record("rollout/ep_healthy_rew_wrapped_mean", healthy_mean) 


class RewardVecEnvWrapper(vec_env.VecEnvWrapper):
    """Uses a provided reward_fn to replace the reward function returned by `step()`.

    Automatically resets the inner VecEnv upon initialization. A tricky part
    about this class is keeping track of the most recent observation from each
    environment.

    Will also include the previous reward given by the inner VecEnv in the
    returned info dict under the `original_env_rew` key.
    """

    def __init__(
        self,
        venv: vec_env.VecEnv,
        reward_fn: reward_function.RewardFn,
        ep_history: int = 100,
    ):
        """Builds RewardVecEnvWrapper.

        Args:
            venv: The VecEnv to wrap.
            reward_fn: A function that wraps takes in vectorized transitions
                (obs, act, next_obs) a vector of episode timesteps, and returns a
                vector of rewards.
            ep_history: The number of episode rewards to retain for computing
                mean reward.
        """
        assert not isinstance(venv, RewardVecEnvWrapper)
        super().__init__(venv)
        self._cumulative_healthy_rew = np.zeros((venv.num_envs,))  
        self.episode_rewards: Deque = collections.deque(maxlen=ep_history)
        self.episode_healthy_rewards: Deque = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self.reward_fn = reward_fn
        self._old_obs = None
        self._actions = None
        self._has_been_unhealthy = np.zeros((venv.num_envs,), dtype=bool)
        self.reset()

    def make_log_callback(self) -> WrappedRewardCallback:
        """Creates `WrappedRewardCallback` connected to this `RewardVecEnvWrapper`."""
        return WrappedRewardCallback(self.episode_rewards, self.episode_healthy_rewards)

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._old_obs = self.venv.reset()
        return self._old_obs

    def step_async(self, actions):
        self._actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, old_rews, dones, infos = self.venv.step_wait()

        # Try to get the 'is_healthy' value from the 'info' dictionary for each environment
        is_healthy = np.array([info.get('is_healthy', True) for info in infos])

        # Update _has_been_unhealthy
        self._has_been_unhealthy |= ~is_healthy

        # Calculate and log the "healthy" reward
        healthy_rews = old_rews * ~self._has_been_unhealthy

        # The vecenvs automatically reset the underlying environments once they
        # encounter a `done`, in which case the last observation corresponding to
        # the `done` is dropped. We're going to pull it back out of the info dict!
        obs_fixed = []
        obs = types.maybe_wrap_in_dictobs(obs)
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                single_obs = single_infos["terminal_observation"]

            obs_fixed.append(types.maybe_wrap_in_dictobs(single_obs))
        obs_fixed = (
            types.DictObs.stack(obs_fixed)
            if isinstance(obs, types.DictObs)
            else np.stack(obs_fixed)
        )
        rews = self.reward_fn(
            self._old_obs,
            self._actions,
            types.maybe_unwrap_dictobs(obs_fixed),
            np.array(dones),
        )
        assert len(rews) == len(obs), "must return one rew for each env"
        done_mask = np.asarray(dones, dtype="bool").reshape((len(dones),))

        # Reset the _has_been_unhealthy mask for environments that have just been reset
        self._has_been_unhealthy[dones] = False

        # Update statistics
        self._cumulative_rew += rews
        self._cumulative_healthy_rew += healthy_rews  # Add this line

        for single_done, single_ep_rew, single_healthy_rew in zip(dones, self._cumulative_rew, self._cumulative_healthy_rew):  # Modify this line
            if single_done:
                self.episode_rewards.append(single_ep_rew)
                self.episode_healthy_rewards.append(single_healthy_rew)  # Add this line
        self._cumulative_rew[done_mask] = 0
        self._cumulative_healthy_rew[done_mask] = 0

        # we can just use obs instead of obs_fixed because on the next iteration
        # after a reset we DO want to access the first observation of the new
        # trajectory, not the last observation of the old trajectory
        obs = types.maybe_unwrap_dictobs(obs)
        self._old_obs = obs
        for info_dict, old_rew in zip(infos, old_rews):
            info_dict["original_env_rew"] = old_rew
        return obs, rews, dones, infos
