"""Wrapper to record rendered video frames from an environment."""

import pathlib
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.wrappers.monitoring import video_recorder
import os


class VideoWrapper(gym.Wrapper):
    """Creates videos from wrapped environment by calling render after each timestep."""

    episode_id: int
    video_recorder: Optional[video_recorder.VideoRecorder]
    directory: pathlib.Path
    record_video_trigger: Any
    video_length: int
    name_prefix: str
    timeline: bool

    def __init__(
        self,
        env: gym.Env,
        directory: pathlib.Path,
        record_video_trigger,
        video_length=200,
        name_prefix='rl-video',
        timeline=False,
    ):
        """Builds a VideoWrapper.

        Args:
            env: the wrapped environment.
            directory: the output directory.
            single_video: if True, generates a single video file, with episodes
                concatenated. If False, a new video file is created for each episode.
                Usually a single video file is what is desired. However, if one is
                searching for an interesting episode (perhaps by looking at the
                metadata), then saving to different files can be useful.
        """
        super().__init__(env)
        self.episode_id = 0
        self.video_recorder = None
        self.record_video_trigger = record_video_trigger 

        self.directory = os.path.abspath(directory)
        # Create output folder if needed
        os.makedirs(self.directory, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recorded_frames = 0

        self.episode_length = 0
        self.fragment_paths = []

        self.timeline = timeline
        self.active = False

    def deactivate(self):
        self.active = False

    def activate(self):
        self.active = True

    def _reset_video_recorder(self) -> None:
        """Creates a video recorder if one does not already exist.

        Called at the start of each episode (by `reset`). When a video recorder is
        already present, it will only create a new one if `self.single_video == False`.
        """
        self.close_video_recorder()

        video_name = '{}-step-{}-to-step-{}'.format(self.name_prefix, self.step_id,
                                                    self.step_id + self.video_length)
        base_path = os.path.join(self.directory, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"episode_id": self.episode_id},
            disable_logger=True,
        )

        self.recorded_frames = 0

    def _video_enabled(self):
        return self.record_video_trigger(self.step_id)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        if self.active:
            self.episode_length = 0
            self.fragment_paths = []
            self._reset_video_recorder()
            self.episode_id += 1
        return super().reset(seed=seed, options=options)

    def step(
        self,
        action: WrapperActType,
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, _, info = super().step(action)

        if not self.active:
            return obs, reward, done, _, info

        if self.video_recorder is None:
            self._reset_video_recorder()

        info['video_path'] = self.video_recorder.path

        if self.video_recorder is not None:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1

            self.step_id += 1
            self.episode_length += 1

            if (self.recorded_frames == self.video_length) or done:
                self.fragment_paths.append(self.video_recorder.path)
                self.close_video_recorder()
                if self.timeline:
                    if done:
                        try:
                            self.add_timelines()
                        except Exception as e:
                            print("Error caught adding timelines: ", e)
        return obs, reward, done, _, info

    def close_video_recorder(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        self.recorded_frames = 0

    def close(self) -> None:
        if self.activate:
            self.close_video_recorder()
        super().close()

    def __del__(self):
        self.close()