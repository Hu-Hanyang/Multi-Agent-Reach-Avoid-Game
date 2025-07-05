from plotting_utilities import animation_2v1
from utilities import *
from odp.Grid import Grid
from compute_opt_traj import compute_opt_traj1v0
from odp.solver import HJSolver, computeSpatDerivArray
from copy import deepcopy
from AttackerDefender1v0 import AttackerDefender1v0
from AttackerDefender1v2 import AttackerDefender1v2



# Simulation: 1 attacker with 2 defenders
# preparations
print("Preparing for the simulaiton... \n")
T = 1.0 # attackers_stop_times = [0.475s (95 A1 is captured), 0.69s (138 A0 by D0)]
deltat = 0.005 # calculation time interval
times = int(T/deltat)

grid_size1v1 = 45
grid_size1v2 = 35

# load all value functions, grids and spatial derivative array
value1v0 = np.load('MARAG/values/1vs0AttackDefend_g100_speed1.0.npy')  # value1v0.shape = [100, 100, len(tau)]
# print(value1v0.shape)

# v1v1 = np.load('MARAG/1v1AttackDefend_speed15.npy')
v1v1 = np.load(f'MARAG/values/1vs1AttackDefend_g45_dspeed1.5.npy')
print(f"The shape of the 1v1 value function is {v1v1.shape}. \n")

v1v2 = np.load('MARAG/values/1vs2AttackDefend_g35_dspeed1.5.npy')
# v1v2 = np.load('MARAG/1v2AttackDefend_g35_dspeed1.0.npy')
# v1v2 = np.load('MARAG/1v2AttackDefend_g30_dspeed1.5.npy')
print(f"The shape of the 1v2 value function is {v1v2.shape}. \n")
value1v2 = v1v2[..., np.newaxis]  # value1v2.shape = [30, 30, 30, 30, 30, 30, 1]

grid1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) # original 45
# grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45
grid1v2 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, 
               np.array([grid_size1v2, grid_size1v2, grid_size1v2, grid_size1v2, grid_size1v2, grid_size1v2])) # original 45
# grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30])) # [36, 36, 36, 36, 36, 36] [30, 30, 30, 30, 30, 30]
agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")
# agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)
agents_1v2 = AttackerDefender1v2(uMode="min", dMode="max")  # 1v2 (6 dim dynamics)
# agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)
tau1v0 = np.arange(start=0, stop=2.5 + 1e-5, step=0.025)
# tau1v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)
tau1v2 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)
# tau2v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)

# Test
attackers_initials = [(-0.15, 0.0)] 
defenders_initials = [(-0.5, 0.8), (-0.5, -0.6)] 

# attackers_initials = [(-0.5, 0.5)]
# defenders_initials = [(0.5, 0.3), (0.5, -0.3)]

# Theoretically capture actually capture
# attackers_initials = [(-0.25, 0.0)] # barely captured
# defenders_initials = [(-0.5, 0.8), (-0.5, -0.6)] 
# attackers_initials = [(-0.2, 0.0)] 
# defenders_initials = [(0.0, 0.8), (-0.5, -0.3)] 

# Theoretically capture actually escape
# attackers_initials = [(0.0, 0.0)] # 
# defenders_initials = [(-0.5, 0.4), (-0.5, -0.3)]  
# attackers_initials =[(-0.05, 0.0)]  
# defenders_initials = [(-0.5, 0.5), (-0.5, -0.1)]  

# ax = attackers_initials[0][0]
# ay = attackers_initials[0][1]
# d1x = defenders_initials[0][0]
# d1y = defenders_initials[0][1]
# d2x = defenders_initials[1][0]
# d2y = defenders_initials[1][1]

# # plot 1v2 reach-avoid tube
# jointstates2v1 = (ax, ay, d1x, d1y, d2x, d2y)
# ax_slice, ay_slice, d1x_slice, d1y_slice, d2x_slice, d2y_slice = lo2slice2v1(jointstates2v1, slices=grid_size1v2)
# value_function1v2_3 = v1v2[ax_slice, ay_slice, d1x_slice, d1y_slice, d2x_slice, d2y_slice]
# print(f"************ The initial value of 1 vs. 2 value function is {value_function1v2_3}. ************ \n")


num_attacker = len(attackers_initials)
num_defender = len(defenders_initials)
attackers_trajectory  = [[] for _ in range(num_attacker)]
defenders_trajectory = [[] for _ in range(num_defender)]


# load the initial states
current_attackers = attackers_initials
current_defenders = defenders_initials
for i in range(num_attacker):
    attackers_trajectory[i].append(current_attackers[i])

for j in range(num_defender):
    defenders_trajectory[j].append(current_defenders[j])

# initialize the captured results
attackers_status_logs = []
attackers_status = [0 for _ in range(num_attacker)]
stops_index = []  # the list stores the indexes of attackers that has been captured or arrived
attackers_status_logs.append(deepcopy(attackers_status))

defenders_controls = []
attackers_controls = []

print("The simulation starts: \n")
# simulation starts
for _ in range(0, times):

    # RA1v1 = capture_1vs1(current_attackers, current_defenders, v1v1, stops_index)  # attacker will win the 1 vs. 1 game
    # RA1v2, RA1v2_ = capture_1vs2(current_attackers, current_defenders, v1v2)  # attacker will win the 1 vs. 2 game
    # RA1v1s.append(RA1v1)
    # RA1v2s.append(RA1v2)
    
    control_defenders = [[] for num in range(num_defender)]  # current controls of defenders, [(d1xc, d1yc), (d2xc, d2yc)]
    a1x, a1y = current_attackers[0]
    d1x, d1y = current_defenders[0]
    d2x, d2y = current_defenders[1]
    joint_state1v2 = (a1x, a1y, d1x, d1y, d2x, d2y)
    opt_d1, opt_d2, opt_d3, opt_d4 = defender_control1vs2_slice(agents_1v2, grid1v2, value1v2, tau1v2, joint_state1v2)

    control_defenders[0].append((opt_d1, opt_d2))
    control_defenders[1].append((opt_d3, opt_d4))  
    defenders_controls.append(deepcopy(control_defenders)) 

    newd_positions = next_positions_d(current_defenders, control_defenders, deltat)
    current_defenders = newd_positions
    
    # calculate the current controls of attackers
    control_attackers = attackers_control(agents_1v0, grid1v0, value1v0, tau1v0, current_attackers)
    attackers_controls.append(deepcopy(control_attackers))
    
    # update the next postions of attackers
    newa_positions = next_positions_a2(current_attackers, control_attackers, deltat, stops_index)  # , current_captured
    current_attackers = newa_positions

    # document the new current positions of attackers and defenders
    for i in range(num_attacker):
        attackers_trajectory[i].append(current_attackers[i])
       

    for j in range(num_defender):
        defenders_trajectory[j].append(current_defenders[j])


    # # check the attackers status: captured or not  
    selected = [[0], [0]]
    attackers_status = capture_check(current_attackers, current_defenders, selected, attackers_status)
    attackers_status_logs.append(deepcopy(attackers_status))
    attackers_arrived = arrived_check(current_attackers)
    stops_index = stoped_check(attackers_status, attackers_arrived)
    # print(f"The current status at iteration{_} of attackers is arrived:{attackers_arrived} + been captured:{attackers_status}. \n")

    if len(stops_index) == num_attacker:
        print(f"All attackers have arrived or been captured at the time t={(_+1)*deltat}. \n")
        break

print("The game is over. \n")

# print(f"The results of the selected is {capture_decisions}. \n")
# print(f"The final captured_status of all attackers is {attackers_status_logs[-1]}. \n")


# print(f"The RA1v1s is {RA1v1s}. \n")
# print(f"The RA1v2s is {RA1v2s}. \n")
# print(f"During the game, the joint state is within the BRT: {judges}. \n")
# Play the animation
animation_2v1(attackers_trajectory, defenders_trajectory, attackers_status_logs, T)
# print(f"The controls of defenders are {defenders_controls}. \n")
# print(f"The controls of attackers are {attackers_controls}. \n")