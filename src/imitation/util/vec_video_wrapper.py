import os

import imageio
import numpy as np

from gymnasium.wrappers.monitoring import video_recorder

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

# from stable_baselines.common.vec_env.vec_video_recorder
# https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/vec_env/vec_video_recorder.html

class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    :param venv: (VecEnv or VecEnvWrapper)
    :param video_folder: (str) Where to save videos
    :param record_video_trigger: (func) Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length: (int)  Length of recorded videos
    :param name_prefix: (str) Prefix to the video name
    """

    def __init__(self, venv, video_folder, record_video_trigger,
                 video_length=200, name_prefix='rl-video', log_folder='logs', timeline=False):

        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecNormalize) or isinstance(temp_env, VecFrameStack):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr('metadata')[0]
        else:
            metadata = temp_env.metadata

        self.env.metadata = metadata

        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

        self.episode_length = 0
        self.fragment_paths = []

        self.logger = Logger(folder=log_folder, output_formats=['.txt'])
        self.timeline = timeline

    def reset(self):
        obs = self.venv.reset()
        self.episode_length = 0
        self.fragment_paths = []
        self.start_video_recorder()
        return obs


    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = '{}-step-{}-to-step-{}'.format(self.name_prefix, self.step_id,
                                                    self.step_id + self.video_length)
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
                env=self.env,
                base_path=base_path,
                metadata={'step_id': self.step_id},
                disable_logger=True,
                )

        self.recorded_frames = 0
        self.recording = True

    def _video_enabled(self):
        return self.record_video_trigger(self.step_id)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        self.episode_length += 1
        if not self.recording and self._video_enabled():
            self.start_video_recorder()

        for info in infos:
            if self.recording:
                info['video_path'] = self.video_recorder.path

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if (self.recorded_frames == self.video_length ) or any(dones):
                self.logger.info("Saving video to ", self.video_recorder.path)
                self.fragment_paths.append(self.video_recorder.path)
                self.close_video_recorder()
                if self.timeline:
                    if any(dones):
                        try:
                            self.add_timelines()
                        except Exception as e:
                            print("Error caught adding timelines: ", e)
                            self.logger.info("Error caught adding timelines: ", e)

        return obs, rews, dones, infos


    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0

    def close(self):
        VecEnvWrapper.close(self)
        self.close_video_recorder()


    def __del__(self):
        self.close()

    def add_timelines(self):
        step = 0
        color = (255, 0, 0)
        start_point = 0

        for fragment_path in self.fragment_paths:
            # Open the video file
            reader = imageio.get_reader(fragment_path)

            # Get the size of the frames in the video
            frame_height, frame_width = reader.get_meta_data()['source_size']

            # Create a temporary VideoWriter object to write the modified frames to a new video
            temp_path = 'temp_' + os.path.basename(fragment_path)
            writer = imageio.get_writer(temp_path, fps=reader.get_meta_data()['fps'])

            for frame in reader:
                # Calculate the start and end points of the timeline
                end_point = int(step / self.episode_length * frame_width)
                
                # Draw the timeline on the frame
                frame[frame_height - 5:frame_height, start_point:end_point] = np.array(color, dtype=np.uint8)

                # Write the modified frame to the new video
                writer.append_data(frame)

                step += 1

            # Close the reader and writer
            reader.close()
            writer.close()

            # Overwrite the original video with the new video
            os.remove(fragment_path)
            os.rename(temp_path, fragment_path)