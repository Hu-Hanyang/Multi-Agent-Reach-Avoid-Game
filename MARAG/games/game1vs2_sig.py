import numpy as np

from MARAG.envs.ReachAvoidGame import ReachAvoidGameEnv
from MARAG.solvers import mip_solver, extend_mip_solver
from MARAG.utilities import *
from MARAG.sig_controllers import hj_controller_attackers_1vs0, hj_controller_defenders, extend_hj_controller_defenders, single_1vs2_controller_defender, hj_controller_1vs0
from MARAG.plots import animation, plot_scene, plot_value_1vs1_sig, plot_value_3agents


#### Game Settings ####
value1vs0, value1vs1, value2vs1, value1vs2, grid1vs0, grid1vs1, grid2vs1, grid1vs2  = hj_preparations_sig()
num_attackers = 1
num_defenders = 2
# # This example, captured, everything seems correct
# initial_attacker = np.array([[-0.3, 0.0]])
# initial_defender = np.array([[-0.7, 0.5], [-0.7, -0.5]])
# This example, captured, but one defender moves to a wierd position
# initial_attacker = np.array([[0.0, 0.0]])
# initial_defender = np.array([[0.5, 0.3], [0.5, -0.3]])
# This example, captured, but one defender moves out of the map
initial_attacker = np.array([[-0.3, 0.0]])
initial_defender = np.array([[0.5, 0.3], [0.5, -0.3]])
# # This example, captured, everything seems correct
# initial_attacker = np.array([[-0.5, 0.0]])
# initial_defender = np.array([[0.4, 0.3], [0.4, -0.3]])
# Random test here
# initial_attacker = np.array([[-0.1, 0.0]])
# initial_defender = np.array([[-0.5, 0.3], [-0.5, -0.3]])

assert num_attackers == initial_attacker.shape[0], "The number of attackers should be equal to the number of initial attacker states."
assert num_defenders == initial_defender.shape[0], "The number of defenders should be equal to the number of initial defender states."
T = 10.0  # time for the game
ctrl_freq = 200  # control frequency, 0.005 seconds each step
total_steps = int(T * ctrl_freq)

print(f"The shape of the value1vs2 is {value1vs2.shape}.")

#### Game Initialization ####
game = ReachAvoidGameEnv(num_attackers=num_attackers, num_defenders=num_defenders, 
                         initial_attacker=initial_attacker, initial_defender=initial_defender, 
                         uMode="max",dMode="min",ctrl_freq=ctrl_freq)

plot_value_3agents(game.attackers.state, game.defenders.state, plot_agents=[0, 1, 2], free_dim=0, value_function=value1vs2, grids=grid1vs2)

print((f"================ The initial value of the game is {check_current_value(game.attackers.state, game.defenders.state, value1vs2, grid1vs2)}. ================ \n"))

defenders_controls = []
attackers_controls = []
#### Game Loop ####
# print(f"================ The game starts now. ================")
for step in range(total_steps):
    control_defenders = single_1vs2_controller_defender(game, value1vs2, grid1vs2)
    defenders_controls.append(control_defenders.copy())
    # control_defenders = hj_controller_defenders(game, assignments, value1vs1, value2vs1, grid1vs1, grid2vs1)

    # control_attackers = np.array([0.0, 0.0])
    control_attackers = hj_controller_1vs0(uMode="min", dMode="max", uMax=1.0, a_speed=1.0, 
                                           value1vs0=value1vs0, grid1vs0=grid1vs0, 
                                           attackers=game.attackers.state, 
                                           current_status=game.attackers_status[-1])
    attackers_controls.append(control_attackers.copy())
    obs, reward, terminated, truncated, info = game.step(np.vstack((control_attackers, control_defenders)))
    
    if terminated or truncated:
        break
    
print(f"================ The game is over at the {step} step ({step / ctrl_freq} seconds). ================ \n")
current_status_check(game.attackers_status[-1], step)
# print(f"================ The number of 1 vs. 2 games happened: {counters_1vs2} ================")

#### Animation ####
animation(game.attackers_traj, game.defenders_traj, game.attackers_status)