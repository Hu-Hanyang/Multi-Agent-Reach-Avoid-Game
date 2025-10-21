import numpy as np
from utils import *

robot_state = np.array([[1.0, 1.0, 0.1, 1.0], [2.0, 2.0, 0.2, 2.0], [3.0, 3.0, 0.3, np.pi/2]])
human_position = np.array([[5.5, 5.5], [10.0, 10.0]])
relative_states = compute_relative_state(human_position, robot_state)
print(f"robot_state.shape:{robot_state.shape}")
print(f"human_position.shape:{human_position.shape}")
print(f"relative_states.shape:{relative_states.shape}")
print(relative_states)
