import os
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# The parent directory where your log files are stored
log_dir = 'tb_logs'
environment_short_name = 'Hopper'
groupwise_number_of_runs = 5
pairwise_number_of_runs = 5
visualization_number_of_runs = 5

# The names of the runs you want to process
groupwise_runs = [f'groupwise_{i}_{environment_short_name}_0' for i in range(groupwise_number_of_runs)]
pairwise_runs = [f'pairwise_{i}_{environment_short_name}_0' for i in range(pairwise_number_of_runs)]
visualization_runs = [f'visualization_{i}_{environment_short_name}_0' for i in range(visualization_number_of_runs)]

# The name of the scalar you want to process
scalar_name = 'rollout/ep_rew_mean'

# Function to calculate the mean and standard deviation of a scalar across multiple runs
def process_runs(runs, scalar_name, window_size=1):
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

    
    # Apply a moving average to the values
    values = [np.convolve(val, np.ones(window_size), 'valid') / window_size for val in values]
    
    # Adjust the steps list to match the length of the values list
    steps = steps[window_size - 1:]
    
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)

    return steps, mean_values, std_values

# Process the runs
groupwise_steps, groupwise_mean, groupwise_std = process_runs(groupwise_runs, scalar_name)
pairwise_steps, pairwise_mean, pairwise_std = process_runs(pairwise_runs, scalar_name)
visualization_steps, visualization_mean, visualization_std = process_runs(visualization_runs, scalar_name)

print(f'Final true reward for groupwise: {groupwise_mean[-1]}')
print(f'Final true reward for pairwise: {pairwise_mean[-1]}')
print(f'Final true reward for visualization: {visualization_mean[-1]}')

# Create a new figure
plt.figure()

plt.title(f'{environment_short_name} - Visualization vs Groupwise vs Pairwise Feedback')

#label the axes
plt.xlabel('Steps')
plt.ylabel('True reward')

# Plot the mean and standard deviation for groupwise
plt.plot(groupwise_steps, groupwise_mean, label=f'Groupwise Mean across {groupwise_number_of_runs} runs')
plt.fill_between(groupwise_steps, groupwise_mean - groupwise_std, groupwise_mean + groupwise_std, alpha=0.1)

# Plot the mean and standard deviation for pairwise
plt.plot(pairwise_steps, pairwise_mean, label=f'Pairwise Mean accross {pairwise_number_of_runs} runs')
plt.fill_between(pairwise_steps, pairwise_mean - pairwise_std, pairwise_mean + pairwise_std, alpha=0.1)

# Plot the mean and standard deviation for visualization
plt.plot(visualization_steps, visualization_mean, label=f'Visualization Mean across {visualization_number_of_runs} runs')
plt.fill_between(visualization_steps, visualization_mean - visualization_std, visualization_mean + visualization_std, alpha=0.1)

# Add a legend
plt.legend(loc='lower right')

# Show the plot
plt.show()
