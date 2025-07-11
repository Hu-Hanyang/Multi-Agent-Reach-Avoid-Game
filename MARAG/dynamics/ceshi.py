import numpy as np

from MARAG.dynamics.SingleIntegrator import SingleIntegrator
from MARAG.dynamics.DubinsCar import DubinsCar3D, DubinsCar3D2Ctrl

# # initiliazation
# control_freq = 10
# dt = 1.0 / control_freq
# state_upper_bound = np.array([+1.0, +1.0])
# state_lower_bound = np.array([-1.0, -1.0]) 
# action_upper_bound = np.array([+1.0, +1.0])
# action_lower_bound = np.array([-1.0, -1.0]) 
# speed = 1.0
# Single Intergrator test
# agents = SingleIntegrator(control_freq, state_lower_bound, state_upper_bound, action_lower_bound, action_upper_bound, speed)
# current_state = np.array([[0.5, 0.5], [-0.5, -0.5]])  # , [-0.5, -0.5]
# print(f"current_state.shape: {current_state.shape}")
# current_action = np.array([[0.1, 0.1], [-0.1, -0.1]]) # , [-0.1, -0.1]
# next_state = agents.forward(current_state, current_action)
# print(f"agent 1 current state: {current_state[0]}")
# print(f"agent 2 current state: {current_state[1]}")
# print(f"agent 1 next state: {next_state[0]}")
# print(f"agent 2 next state: {next_state[1]}")

# # Dubins3D test initiliazation
# control_freq = 10
# dt = 1.0 / control_freq
# state_lower_bound = np.array([-1.0, -1.0, -np.pi]) 
# state_upper_bound = np.array([+1.0, +1.0, np.pi])
# action_lower_bound = np.array([-5.0]) 
# action_upper_bound = np.array([+5.0])
# speed = 1.0
# agent = DubinsCar3D(control_freq, state_lower_bound, state_upper_bound, action_lower_bound, action_upper_bound, speed)
# current_state = np.array([[0.0, 0.0, 0.0]])
# print(f"current_state.shape: {current_state.shape}")
# current_action = np.array([[np.pi]])
# next_state = agent.forward(current_state, current_action)# print(f"agent 1 current state: {current_state[0]}")
# # print(f"agent 2 current state: {current_state[1]}")
# print(f"agent 1 next state: {next_state[0]}")
# # print(f"agent 2 next state: {next_state[1]}")


# DubinsCar3D2Ctrl
control_freq = 10
dt = 1.0 / control_freq
state_lower_bound = np.array([-1.0, -1.0, -np.pi]) 
state_upper_bound = np.array([+1.0, +1.0, np.pi])
action_lower_bound = np.array([-2.0, -5.0]) 
action_upper_bound = np.array([+2.0, +5.0])
speed = 1.0
agent = DubinsCar3D2Ctrl(control_freq, state_lower_bound, state_upper_bound, action_lower_bound, action_upper_bound)
current_state = np.array([[0.0, 0.0, 0.0], [-0.5, 0.5, 0.0]])
current_action = np.array([[1.0, np.pi], [0.5, 0.0]])
next_state = agent.forward(current_state, current_action)# print(f"agent 1 current state: {current_state[0]}")
print(f"agent 1 next state: {next_state[0]}")
print(f"agent 2 next state: {next_state[1]}")
