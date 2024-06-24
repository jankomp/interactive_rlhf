import os
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# The parent directory where your log files are stored
log_dir = 'compare_feedback_types'

# The names of the runs you want to process
groupwise_runs = [f'groupwise_{i}_0' for i in range(10)]
pairwise_runs = [f'pairwise_{i}_0' for i in range(10)]

# The name of the scalar you want to process
scalar_name = 'rollout/ep_rew_mean'

# Function to calculate the mean and standard deviation of a scalar across multiple runs
def process_runs(runs, scalar_name):
    scalar_events = []

    for run in runs:
        # Load the events file
        event_acc = EventAccumulator(os.path.join(log_dir, run))
        event_acc.Reload()

        # Get the scalar events
        scalars = event_acc.Scalars(scalar_name)
        scalar_events.append(scalars)

    # Calculate the mean and standard deviation
    steps = [event.step for event in scalar_events[0]]
    values = [[event.value for event in scalars] for scalars in scalar_events]
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)

    return steps, mean_values, std_values

# Process the runs
groupwise_steps, groupwise_mean, groupwise_std = process_runs(groupwise_runs, scalar_name)
pairwise_steps, pairwise_mean, pairwise_std = process_runs(pairwise_runs, scalar_name)

# Write the new summary files
with tf.summary.create_file_writer('groupwise_mean').as_default():
    for step, value in zip(groupwise_steps, groupwise_mean):
        tf.summary.scalar(scalar_name, value, step=step)

with tf.summary.create_file_writer('pairwise_mean').as_default():
    for step, value in zip(pairwise_steps, pairwise_mean):
        tf.summary.scalar(scalar_name, value, step=step)