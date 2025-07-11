'''Utility functions for the reach-avoid game.

'''

import math
import time
import numpy as np

from odp.Grid import Grid
from MARAG.dynamics.SingleIntegrator_old import SingleIntegrator
from MARAG.dynamics.DubinsCar3D_old import DubinsCar


def make_agents(physics_info, numbers, initials, freqency):
    '''Make the agents with the given physics list, numbers and initials.
    
    Args:
        physics_info (dic): the physics info of the agent
        numbers (int): the number of agents
        initials (np.ndarray): the initial states of all agents
        freqency (int): the frequency of the simulation
    '''
    if physics_info['id'] == 'sig':
        return SingleIntegrator(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    elif physics_info['id'] == 'fsig':
        return SingleIntegrator(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    elif physics_info['id'] == 'dub3d':
        return DubinsCar(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    elif physics_info['id'] == 'fdub3d':
        return DubinsCar(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    else:
        raise ValueError("Invalid physics info while generating agents.")


def hj_preparations_sig():
    """ Loads all calculated HJ value functions for the single integrator agents.
    This function needs to be called before any game starts.
    
    Returns:
        value1vs0 (np.ndarray): the value function for 1 vs 0 game with all time slices
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
        value2vs1 (np.ndarray): the value function for 2 vs 1 game
        value1vs2 (np.ndarray): the value function for 1 vs 2 game
        grid1vs0 (Grid): the grid for 1 vs 0 game
        grid1vs1 (Grid): the grid for 1 vs 1 game
        grid2vs1 (Grid): the grid for 2 vs 1 game
        grid1vs2 (Grid): the grid for 1 vs 2 game
    """
    start = time.time()
    value1vs0 = np.load('MARAG/values/1vs0_SIG_g100_medium_speed1.0.npy')
    value1vs1 = np.load('MARAG/values/1vs1_SIG_g45_medium_dspeed1.5.npy')
    value2vs1 = np.load('MARAG/values/2vs1AttackDefend_g30_speed1.5.npy')
    value1vs2 = np.load('MARAG/values/1vs2_SIG_g32_medium_dspeed1.5.npy')
    end = time.time()
    print(f"============= HJ value functions loaded Successfully! (Time: {end-start :.4f} seconds) =============")
    grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
    grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
    grid2vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30]))
    # grid1vs2 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([35, 35, 35, 35, 35, 35]))
    grid1vs2 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([32, 32, 32, 32, 32, 32]))
    print(f"============= Grids created Successfully! =============")

    return value1vs0, value1vs1, value2vs1, value1vs2, grid1vs0, grid1vs1, grid2vs1, grid1vs2


def hj_preparations_dub():
    """ Loads all calculated HJ value functions for the DubinCar agents.
    This function needs to be called before any game starts.
    
    Returns:
        value1vs0 (np.ndarray): the value function for 1 vs 0 game with all time slices
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
        grid1vs0 (Grid): the grid for 1 vs 0 game
        grid1vs1 (Grid): the grid for 1 vs 1 game
    """
    start = time.time()
    # value1vs0_dub = np.load('MARAG/values/DubinCar1vs0_grid100_medium.npy')
    value1vs0_dub = np.load("MARAG/values/DubinCar1vs0_grid100_medium_1.0angularv.npy")
    value1vs1_dub = np.load('MARAG/values/DubinCar1vs1_grid28_medium_1.0angularv.npy')
    # value1vs1_dub = np.load('MARAG/values/DubinCar1vs1_grid28_medium_1.0angularv_defenderview.npy')
    end = time.time()
    print(f"============= HJ value functions loaded Successfully! (Time: {end-start :.4f} seconds) =============")
    grid_size_1vs0_0 = value1vs0_dub.shape[0]
    grid_size_1vs0_2 = value1vs0_dub.shape[2]
    grid_size_1vs1 = value1vs1_dub.shape[0]
    grid1vs0_dub = Grid(np.array([-1.0, -1.0, -math.pi]), np.array([1.0, 1.0, math.pi]), 3, 
                        np.array([grid_size_1vs0_0, grid_size_1vs0_0, grid_size_1vs0_2]), [2])
    grid1vs1_dub = Grid(np.array([-1.0, -1.0, -math.pi, -1.0, -1.0, -math.pi]), 
                    np.array([1.0, 1.0, math.pi, 1.0, 1.0, math.pi]), 6, 
                    np.array([grid_size_1vs1, grid_size_1vs1, grid_size_1vs1, grid_size_1vs1, grid_size_1vs1, grid_size_1vs1]), [2, 5])
    print(f"============= Grids created Successfully! =============")

    return value1vs0_dub, grid1vs0_dub, value1vs1_dub, grid1vs1_dub


def po2slice1vs1(attacker, defender, grid_size):
    """ Convert the position of the attacker and defender to the slice of the value function for 1 vs 1 game.

    Args:
        attacker (np.ndarray): the attacker's state
        defender (np.ndarray): the defender's state
        grid_size (int): the size of the grid
    
    Returns:
        joint_slice (tuple): the joint slice of the joint state using the grid size

    """
    joint_state = (attacker[0], attacker[1], defender[0], defender[1])  # (xA1, yA1, xD1, yD1)
    joint_slice = []
    grid_points = np.linspace(-1, +1, num=grid_size)
    for i, s in enumerate(joint_state):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            joint_slice.append(idx - 1)
        else:
            joint_slice.append(idx)

    return tuple(joint_slice)


def po2slice2vs1(attacker_i, attacker_k, defender, grid_size):
    """ Convert the position of the attackers and defender to the slice of the value function for 2 vs 1 game.

    Args:
        attackers (np.ndarray): the attackers' states
        defender (np.ndarray): the defender's state
        grid_size (int): the size of the grid
    
    Returns:
        joint_slice (tuple): the joint slice of the joint state using the grid size

    """
    joint_state = (attacker_i[0], attacker_i[1], attacker_k[0], attacker_k[1], defender[0], defender[1])  # (xA1, yA1, xA2, yA2, xD1, yD1)
    joint_slice = []
    grid_points = np.linspace(-1, +1, num=grid_size)
    for i, s in enumerate(joint_state):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            joint_slice.append(idx - 1)
        else:
            joint_slice.append(idx)

    return tuple(joint_slice)


def check_1vs1(attacker, defender, value1vs1):
    """ Check if the attacker could escape from the defender in a 1 vs 1 game.

    Args:
        attacker (np.ndarray): the attacker's state
        defender (np.ndarray): the defender's state
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
    
    Returns:
        bool: False, if the attacker could escape (the attacker will win)
    """
    joint_slice = po2slice1vs1(attacker, defender, value1vs1.shape[0])

    return value1vs1[joint_slice] > 0


def check_2vs1(attacker_i, attacker_k, defender, value2vs1):
    """ Check if the attackers could escape from the defender in a 2 vs 1 game.
    Here escape means that at least one of the attackers could escape from the defender.

    Args:
        attacker_i (np.ndarray): the attacker_i's states
        attacker_j (np.ndarray): the attacker_j's states
        defender (np.ndarray): the defender's state
        value2vs1 (np.ndarray): the value function for 2 vs 1 game
    
    Returns:
        bool: False, if the attackers could escape (the attackers will win)
    """
    joint_slice = po2slice2vs1(attacker_i, attacker_k, defender, value2vs1.shape[0])

    return value2vs1[joint_slice] > 0


def check_1vs2(attacker, defender_j, defender_k, value1vs2, epsilon=0.035):
    """ Check if the attacker could escape from the defenders in a 1 vs 2 game.

    Args:
        attacker (np.ndarray): the attacker's state
        defender_j (np.ndarray): the defender_i's state
        defender_k (np.ndarray): the defender_k's state
        value1vs2 (np.ndarray): the value function for 1 vs 2 game
        epsilon (float): the threshold for the attacker to escape
    
    Returns:
        bool: False, if the attacker could escape (the attacker will win)
    """
    joint_slice = po2slice2vs1(attacker, defender_j, defender_k, value1vs2.shape[0])

    return value1vs2[joint_slice] > epsilon


def judge_1vs1(attackers, defenders, current_attackers_status, value1vs1):
    """ Check the result of the 1 vs 1 game for those free attackers.

    Args:  
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
    
    Returns:
        EscapedAttacker1vs1 (a list of lists): the attacker that could escape from the defender in a 1 vs 1 game
    """
    num_attackers, num_defenders = len(attackers), len(defenders)
    EscapedAttacker1vs1 = [[] for _ in range(num_defenders)]

    for j in range(num_defenders):
        for i in range(num_attackers):
            if not current_attackers_status[i]:  # the attcker[i] is free now
                if not check_1vs1(attackers[i], defenders[j], value1vs1):  # the attacker could escape
                    EscapedAttacker1vs1[j].append(i)

    return EscapedAttacker1vs1
    

def judge_2vs1(attackers, defenders, current_attackers_status, value2vs1):
    """ Check the result of the 2 vs 1 game for those free attackers.
    
    Args:
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        current_attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        value2vs1 (np.ndarray): the value function for 2 vs 1 game
    
    Returns:
        EscapedPairs2vs1 (a list of lists): the pair of attackers that could escape from the defender in a 2 vs 1 game
    """
    num_attackers, num_defenders = len(attackers), len(defenders)
    EscapedPairs2vs1 = [[] for _ in range(num_defenders)]
    for j in range(num_defenders):
        for i in range(num_attackers):
            if not current_attackers_status[i]:  # the attcker[i] is free now
                for k in range(i+1, num_attackers):
                    if not current_attackers_status[k]:
                        if not check_2vs1(attackers[i], attackers[k], defenders[j], value2vs1):
                            EscapedPairs2vs1[j].append([i, k])
    
    return EscapedPairs2vs1


def judge_1vs2(attackers, defenders, current_attackers_status, value1vs2):
    """ Check the result of the 1 vs 2 game for those free attackers.

    Args:
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        current_attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        value1vs2 (np.ndarray): the value function for 1 vs 2 game

    Returns:
        EscapedAttackers1vs2 (a list of lists): the attacker that could escape from the defenders in a 1 vs 2 game
        EscapedTri1vs2 (a list of lists): the triad of the attacker and defenders that could escape from the defenders in a 1 vs 2 game
    """
    num_attackers, num_defenders = len(attackers), len(defenders)
    EscapedAttackers1vs2 = [[] for _ in range(num_defenders)]
    EscapedTri1vs2 = [[] for _ in range(num_defenders)]  #
    for j in range(num_defenders):
        for k in range(j+1, num_defenders):
            for i in range(num_attackers):
                if not current_attackers_status[i]:
                    if not check_1vs2(attackers[i], defenders[j], defenders[k], value1vs2):
                        EscapedAttackers1vs2[j].append(i)
                        EscapedAttackers1vs2[k].append(i)
                        EscapedTri1vs2[j].append([i, j, k])
                        EscapedTri1vs2[k].append([i, j, k])
                        
    return EscapedAttackers1vs2, EscapedTri1vs2


def judges(attackers, defenders, current_attackers_status, value1vs1, value2vs1, value1vs2):
    #TODO: what name is it???
    """ Check the result of 1 vs. 1, 2 vs. 1 and 1 vs. 2 games for those free attackers.

    Args:
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        current_attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
        value2vs1 (np.ndarray): the value function for 2 vs 1 game
        value1vs2 (np.ndarray): the value function for 1 vs 2 game

    Returns:
        EscapedAttacker1vs1 (a list of lists): the attacker that could escape from the defender in a 1 vs 1 game
        EscapedPairs2vs1 (a list of lists): the pair of attackers that could escape from the defender in a 2 vs 1 game
        EscapedAttackers1vs2 (a list of lists): the attacker that could escape from the defenders in a 1 vs 2 game
        EscapedTri1vs2 (a list of lists): the triad of the attacker and defenders that could escape from the defenders in a 1 vs 2 game
    """
    EscapedAttacker1vs1 = judge_1vs1(attackers, defenders, current_attackers_status, value1vs1)
    EscapedPairs2vs1 = judge_2vs1(attackers, defenders, current_attackers_status, value2vs1)
    EscapedAttackers1vs2, EscapedTri1vs2 = judge_1vs2(attackers, defenders, current_attackers_status, value1vs2)
    
    return EscapedAttacker1vs1, EscapedPairs2vs1, EscapedAttackers1vs2, EscapedTri1vs2


def current_status_check(current_attackers_status, step=None):
    """ Check the current status of the attackers.

    Args:
        current_attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        step (int): the current step of the game
    
    Returns:
        status (dic): the current status of the attackers
    """
    num_attackers = len(current_attackers_status)
    num_free, num_arrived, num_captured, num_stuck = 0, 0, 0, 0
    status = {'free': [], 'arrived': [], 'captured': [], 'stuck':[]}
    
    for i in range(num_attackers):
        if current_attackers_status[i] == 0:
            num_free += 1
            status['free'].append(i)
        elif current_attackers_status[i] == 1:
            num_arrived += 1
            status['arrived'].append(i)
        elif current_attackers_status[i] == -1:
            num_captured += 1
            status['captured'].append(i)
        elif current_attackers_status[i] == -2:
            num_stuck += 1
            status['stuck'].append(i)
        else:
            raise ValueError("Invalid status for the attackers.")
    
    print(f"================= Step {step}: {num_captured}/{num_attackers} attackers are captured \t"
      f"{num_arrived}/{num_attackers} attackers have arrived \t"
      f"{num_stuck}/{num_attackers} attackers get stuck in the obs \t"
      f"{num_free}/{num_attackers} attackers are free =================")

    print(f"================= The current status of the attackers: {status} =================")

    return status


def current_status_check_dub(current_attackers_status, step=None):
    """ Check the current status of the attackers.

    Args:
        current_attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        step (int): the current step of the game
    
    Returns:
        status (dic): the current status of the attackers
    """
    num_attackers = len(current_attackers_status)
    num_free, num_arrived, num_captured = 0, 0, 0
    status = {'free': [], 'arrived': [], 'captured': []}
    
    for i in range(num_attackers):
        if current_attackers_status[i] == 0:
            num_free += 1
            status['free'].append(i)
        elif current_attackers_status[i] == 1:
            num_arrived += 1
            status['arrived'].append(i)
        elif current_attackers_status[i] == -1:
            num_captured += 1
            status['captured'].append(i)
        else:
            raise ValueError("Invalid status for the attackers.")
    
    print(f"================= Step {step}: {num_captured}/{num_attackers} attackers are captured \t"
      f"{num_arrived}/{num_attackers} attackers have arrived \t"
      f"{num_free}/{num_attackers} attackers are free =================")

    print(f"================= The current status of the attackers: {status} =================")

    return status


def check_current_value(attackers, defenders, value_function, grids):
    """ Check the value of the current state of the attackers and defenders.

    Args:
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        value (np.ndarray): the value function for the game
        grid (Grid): the grid for the game

    Returns:
        value (float): the value of the current state of the attackers and defenders
    """
    if len(value_function.shape) == 4:  # 1vs1 game
        # joint_slice = po2slice1vs1(attackers[0], defenders[0], value_function.shape[0])
        joint_slice = grids.get_index(np.concatenate((attackers[0], defenders[0])))
    elif len(value_function.shape) == 6:  # 1vs2 or 2vs1 game
        if attackers.shape[0] == 1:  # 1vs2 game
            # joint_slice = po2slice2vs1(attackers[0], defenders[0], defenders[1], value_function.shape[0])
            joint_slice = grids.get_index(np.concatenate((attackers[0], defenders[0], defenders[1])))
        else:  # 2vs1 game
            # joint_slice = po2slice2vs1(attackers[0], attackers[1], defenders[0], value_function.shape[0])
            joint_slice = grids.get_index(np.concatenate((attackers[0], attackers[1], defenders[0])))

    value = value_function[joint_slice]

    return value


def dubin_inital_check(initial_attacker, initial_defender):
    """ Make sure the angle is in the range of [-pi, pi), if not, change it.
    
    Args:
        inital_attacker (np.ndarray, (num_attacker, 3)): the initial state of the attacker
        initial_defender (np.ndarray, (num_defender, 3)): the initial state of the defender
    
    Returns:
        initial_attacker (np.ndarray, (num_attacker, 3)): the initial state of the attacker after revision if necessary
        initial_defender (np.ndarray, (num_defender, 3)): the initial state of the defender after revision if necessary
    """
    def normalize_angle(angle):
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def normalize_states(states):
        if states is not None:
            for state in states:
                state[2] = normalize_angle(state[2])
        return states
    
    initial_attacker = normalize_states(initial_attacker)
    initial_defender = normalize_states(initial_defender)
    
    return initial_attacker, initial_defender
