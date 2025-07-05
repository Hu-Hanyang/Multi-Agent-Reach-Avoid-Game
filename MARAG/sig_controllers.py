'''Controllers for the reach-avoid game with sig dynamics.

'''
import numpy as np


def spa_deriv(slice_index, value_function, grid, periodic_dims=[]):
    """
    Calculates the spatial derivatives of V at an index for each dimension

    Args:
        slice_index: (a1x, a1y)
        value_function (ndarray): [..., neg2pos] where neg2pos is a list [scalar] or []
        grid (class): the instance of the corresponding Grid
        periodic_dims (list): the corrsponding periodical dimensions []

    Returns:
        List of left and right spatial derivatives for each dimension
    """
    spa_derivatives = []
    for dim, idx in enumerate(slice_index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(slice_index[:dim])

        if dim == len(slice_index) - 1:
            right_index = []
        else:
            right_index = list(slice_index[dim + 1:])

        next_index = tuple(
            left_index + [slice_index[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [slice_index[dim] - 1] + right_index
        )

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [value_function.shape[dim] - 1] + right_index
                )
                left_boundary = value_function[left_periodic_boundary_index]
            else:
                left_boundary = value_function[slice_index] + np.abs(value_function[next_index] - value_function[slice_index]) * np.sign(value_function[slice_index])
            left_deriv = (value_function[slice_index] - left_boundary) / grid.dx[dim]
            right_deriv = (value_function[next_index] - value_function[slice_index]) / grid.dx[dim]
        elif idx == value_function.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(
                    left_index + [0] + right_index
                )
                right_boundary = value_function[right_periodic_boundary_index]
            else:
                right_boundary = value_function[slice_index] + np.abs(value_function[slice_index] - value_function[prev_index]) * np.sign([value_function[slice_index]])
            left_deriv = (value_function[slice_index] - value_function[prev_index]) / grid.dx[dim]
            right_deriv = (right_boundary - value_function[slice_index]) / grid.dx[dim]
        else:
            left_deriv = (value_function[slice_index] - value_function[prev_index]) / grid.dx[dim]
            right_deriv = (value_function[next_index] - value_function[slice_index]) / grid.dx[dim]

        spa_derivatives.append(((left_deriv + right_deriv) / 2)[0])
        
    return spa_derivatives


def defender_control_2vs1(game, grid2vs1, value2vs1, jointstate_2vs1):
    """Return a tuple of 2-dimensional control inputs of one defender based on the value function
    
    Args:
        game (class): the corresponding ReachAvoidGameEnv instance
        grid2vs1 (class): the corresponding Grid instance
        value2vs1 (ndarray, (grid_size*dim, 1)): 1v1 HJ reachability value function with only final slice
        game (class instance): the corresponding ReachAvoidGameEnv instance
        jointstate_2vs1 (tuple): the corresponding positions of (A1, A2, D1)

    Returns:
        opt_d1, opt_d2 (tuple): the optimal control of the defender
    """
    value2vs1s = value2vs1[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid2vs1.get_index(jointstate_2vs1), value2vs1s, grid2vs1)
    opt_d1, opt_d2 = game.optDistb_2vs1(spat_deriv_vector)

    return (opt_d1, opt_d2)


def defender_control_1vs1(game, grid1vs1, value1vs1, jointstate_1vs1):
    """Return a tuple of 2-dimensional control inputs of one defender based on the value function
    
    Args:
        game (class): the corresponding ReachAvoidGameEnv instance
        grid1v1 (class): the corresponding Grid instance
        value1v1 (ndarray): 1vs1 HJ reachability value function with only final slice
        agents_1v1 (class): the corresponding AttackerDefender instance
        joint_states1v1 (tuple): the corresponding positions of (A1, D1)
    
    Returns:
        opt_d1, opt_d2 (tuple): the optimal control of the defender
    """
    value1vs1s = value1vs1[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid1vs1.get_index(jointstate_1vs1), value1vs1s, grid1vs1)
    opt_d1, opt_d2 = game.optDistb_1vs1(spat_deriv_vector)

    return (opt_d1, opt_d2)


def defender_control_1vs2(game, grid1vs2, value1vs2, jointstate_1vs2):
    """Return a tuple of 4-dimensional control inputs of one defender based on the value function
    
    Args:
        game (class): the corresponding ReachAvoidGameEnv instance
        grid1vs2 (class): the corresponding Grid instance
        value1vs2 (ndarray, (grid_size*dim, 1)): 1vs2 HJ reachability value function with only final slice
        game (class instance): the corresponding ReachAvoidGameEnv instance
        jointstate_1vs2 (tuple): the corresponding positions of (A1, D1, D2)

    Returns:
        opt_d1, opt_d2 (tuple): the optimal control of the defender
    """
    value1vs2s = value1vs2[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid1vs2.get_index(jointstate_1vs2), value1vs2s, grid1vs2)
    opt_d1, opt_d2, opt_d3, opt_d4 = game.optDistb_1vs2(spat_deriv_vector)

    return (opt_d1, opt_d2, opt_d3, opt_d4)


def attacker_control_1vs0(game, grid1vs0, value1vs0, attacker, neg2pos):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    game (class): the corresponding ReachAvoidGameEnv instance
    grid1vs0 (class): the corresponding Grid instance
    value1vs0 (ndarray): 1v1 HJ reachability value function with only final slice
    attacker (ndarray, (dim,)): the current state of one attacker
    neg2pos (list): the positions of the value function that change from negative to positive
    """
    current_value = grid1vs0.get_value(value1vs0[..., 0], list(attacker))
    if current_value > 0:
        value1vs0 = value1vs0 - current_value
    v = value1vs0[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
    spat_deriv_vector = spa_deriv(grid1vs0.get_index(attacker), v, grid1vs0)
    opt_a1, opt_a2 = game.optCtrl_1vs0(spat_deriv_vector)

    return (opt_a1, opt_a2)


def attacker_control_1vs1(game, grid1vs1, value1vs1, current_state, neg2pos):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid1vs1 (class): the corresponding Grid instance
    value1vs1 (ndarray): 1v1 HJ reachability value function with only final slice
    current_state (ndarray, (dim,)): the current state of one attacker + one defender
    neg2pos (list): the positions of the value function that change from negative to positive
    """
    current_value = grid1vs1.get_value(value1vs1[..., 0], list(current_state))
    if current_value > 0:
        value1vs1 = value1vs1 - current_value
    v = value1vs1[..., neg2pos]
    spat_deriv_vector = spa_deriv(grid1vs1.get_index(current_state), v, grid1vs1)
    opt_a1, opt_a2 = game.optCtrl_1vs1(spat_deriv_vector)

    return (opt_a1, opt_a2)


def find_sign_change1vs0(grid1vs0, value1vs0, attacker):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1vs0 (class): the instance of grid
    value1vs0 (ndarray): including all the time slices, shape = [100, 100, len(tau)]
    attacker (ndarray, (dim,)): the current state of one attacker
    """
    current_slices = grid1vs0.get_index(attacker)
    current_value = value1vs0[current_slices[0], current_slices[1], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]


def find_sign_change1vs1(grid1vs1, value1vs1, current_state):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1vs1 (class): the instance of grid
    value1vs1 (ndarray): including all the time slices, shape = [45, 45, 45, 45, len(tau)]
    current_state (ndarray, (dim,)): the current state of one attacker + one defender
    """
    current_slices = grid1vs1.get_index(current_state)
    current_value = value1vs1[current_slices[0], current_slices[1], current_slices[2], current_slices[3], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]


def hj_controller_defenders(game, assignments, 
                            value1vs1, value2vs1, 
                            grid1vs1, grid2vs1): 
    """This fuction computes the control for the defenders based on the assignments. 
       Assume dynamics are single integrator.

    Args:
        game (class): the corresponding ReachAvoidGameEnv instance
        assignments (a list of lists): the list of attackers that the defender assigned to capture
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
        value2vs1 (np.ndarray): the value function for 2 vs 1 game
        value1vs2 (np.ndarray): the value function for 1 vs 2 game
        grid1vs1 (Grid): the grid for 1 vs 1 game
        grid2vs1 (Grid): the grid for 2 vs 1 game
        grid1vs2 (Grid): the grid for 1 vs 2 game
    
    Returns:
        control_defenders ((ndarray): the control of defenders
    """
    attackers = game.attackers.state.copy()
    defenders = game.defenders.state.copy()
    num_defenders = game.NUM_DEFENDERS 
    control_defenders = np.zeros((num_defenders, 2))

    for j in range(num_defenders):
        d1x, d1y = defenders[j]
        if len(assignments[j]) == 2: # defender j capture the attacker selected[j][0] and selected[j][1]
            a1x, a1y = attackers[assignments[j][0]]
            a2x, a2y = attackers[assignments[j][1]]
            jointstate_2vs1 = (a1x, a1y, a2x, a2y, d1x, d1y)
            control_defenders[j] = defender_control_2vs1(game, grid2vs1, value2vs1, jointstate_2vs1)
        elif len(assignments[j]) == 1:
            a1x, a1y = attackers[assignments[j][0]]
            jointstate_1vs1 = (a1x, a1y, d1x, d1y)
            control_defenders[j] = defender_control_1vs1(game, grid1vs1, value1vs1, jointstate_1vs1)
        elif len(assignments[j]) == 0: # defender j could not capture any of attackers
            control_defenders[j] = (0.0, 0.0)
        else:
            raise ValueError("The number of attackers assigned to one defender should be less than 3.")
        
    return control_defenders


def extend_hj_controller_defenders(game, 
                                   assignments, weights, attacker_views,
                                   value1vs1, value2vs1, value1vs2, 
                                   grid1vs1, grid2vs1, grid1vs2):
    """This fuction computes the control for the defenders based on the assignments.
       Assume dynamics are single integrator.

    Args:
        game (class): the corresponding ReachAvoidGameEnv instance
        assignments (a list of lists): the list of attackers that the defender assigned to capture
        weights (np.ndarray, (num_free_attackers, num_defenders)): the weights for each assignment
        attacker_views (a list of lists): the list of defenders that could capture the attacker
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
        value2vs1 (np.ndarray): the value function for 2 vs 1 game
        value1vs2 (np.ndarray): the value function for 1 vs 2 game
        grid1vs1 (Grid): the grid for 1 vs 1 game
        grid2vs1 (Grid): the grid for 2 vs 1 game
        grid1vs2 (Grid): the grid for 1 vs 2 game

    Returns:
        control_defenders ((ndarray): the control of defenders
    """
    attackers = game.attackers.state.copy()
    defenders = game.defenders.state.copy()
    num_defenders = game.NUM_DEFENDERS 
    control_defenders = np.zeros((num_defenders, 2))
    calculated_defenders = []  # store the calculated defenders which should not calculate the control again
    flag_1vs2 = False

    for j in range(num_defenders):

        if j in calculated_defenders:
            continue
        d1x, d1y = defenders[j]

        if len(assignments[j]) == 2: # defender j capture the attacker selected[j][0] and selected[j][1]
            a1x, a1y = attackers[assignments[j][0]]
            a2x, a2y = attackers[assignments[j][1]]
            jointstate_2vs1 = (a1x, a1y, a2x, a2y, d1x, d1y)
            control_defenders[j] = defender_control_2vs1(game, grid2vs1, value2vs1, jointstate_2vs1)
        elif len(assignments[j]) == 1:
            a1x, a1y = attackers[assignments[j][0]]

            if weights[assignments[j][0], j] == 0.5:  # use 1 vs. 2 game based control
                collaborate_defender = attacker_views[assignments[j][0]][-1]
                d2x, d2y = defenders[collaborate_defender]
                jointstate_1vs2 = (a1x, a1y, d1x, d1y, d2x, d2y)
                opt_d1, opt_d2, opt_d3, opt_d4 = defender_control_1vs2(game, grid1vs2, value1vs2, jointstate_1vs2)
                control_defenders[j] = (opt_d1, opt_d2)
                control_defenders[collaborate_defender] = (opt_d3, opt_d4)
                calculated_defenders.append(collaborate_defender)
                flag_1vs2 = True
            else:  # use 1 vs. 1 game based control
                jointstate_1vs1 = (a1x, a1y, d1x, d1y)
                control_defenders[j] = defender_control_1vs1(game, grid1vs1, value1vs1, jointstate_1vs1)

        elif len(assignments[j]) == 0: # defender j could not capture any of attackers
            control_defenders[j] = (0.0, 0.0)
        else:
            raise ValueError("The number of attackers assigned to one defender should be less than 3.")

    return control_defenders, flag_1vs2


def single_1vs2_controller_defender(game, value1vs2, grid1vs2):
    attackers = game.attackers.state.copy()
    defenders = game.defenders.state.copy()
    assert game.NUM_ATTACKERS == 1, "The number of attacker should be 1."
    assert game.NUM_DEFENDERS == 2, "The number of defender should be 2."
    num_defenders = game.NUM_DEFENDERS 
    control_defenders = np.zeros((num_defenders, 2))
    
    a1x, a1y = attackers[0]
    d1x, d1y = defenders[0]
    d2x, d2y = defenders[1]
    jointstate_1vs2 = (a1x, a1y, d1x, d1y, d2x, d2y)
    opt_d1, opt_d2, opt_d3, opt_d4 = defender_control_1vs2(game, grid1vs2, value1vs2, jointstate_1vs2)
    
    control_defenders[0] = (opt_d1, opt_d2)
    control_defenders[1] = (opt_d3, opt_d4)

    return control_defenders


def hj_controller_attackers_1vs0(game, value1vs0, grid1vs0):
    """This function computes the control for the attackers based on the control_attackers. 
       Assume dynamics are single integrator.

    Args:
        game (class): the corresponding ReachAvoidGameEnv instance
        value1vs0 (np.ndarray): the value function for 1 vs 0 game with all time slices
        grid1vs0 (Grid): the grid for 1 vs 0 game
    
    Returns:
        control_attackers (ndarray): the control of attackers
    """
    attackers = game.attackers.state
    num_attackers = game.NUM_ATTACKERS
    current_attackers_status = game.attackers_status[-1]
    control_attackers = np.zeros((num_attackers, 2))
    for i in range(num_attackers):
        if not current_attackers_status[i]:  # the attacker is free
            neg2pos, pos2neg = find_sign_change1vs0(grid1vs0, value1vs0, attackers[i])
            if len(neg2pos):
                control_attackers[i] = attacker_control_1vs0(game, grid1vs0, value1vs0, attackers[i], neg2pos)
            else:
                control_attackers[i] = (0.0, 0.0)
        else:  # the attacker is captured or arrived
            control_attackers[i] = (0.0, 0.0)
            
    return control_attackers



def hj_contoller_attackers_1vs1(game, value1vs1, grid1vs1):
    """This function computes the control for the attackers based on the control_attackers. 
       Assume dynamics are single integrator.

    Args:
        game (class): the corresponding ReachAvoidGameEnv instance
        value1vs0 (np.ndarray): the value function for 1 vs 0 game with all time slices
        grid1vs0 (Grid): the grid for 1 vs 0 game
    
    Returns:
        control_attackers (ndarray): the control of attackers
    """
    attackers = game.attackers.state
    current_state = game.state.copy().flatten()
    num_attackers = game.NUM_ATTACKERS
    current_attackers_status = game.attackers_status[-1]
    control_attackers = np.zeros((num_attackers, 2))
    for i in range(num_attackers):
        if not current_attackers_status[i]:  # the attacker is free
            neg2pos, pos2neg = find_sign_change1vs1(grid1vs1, value1vs1, current_state)
            if len(neg2pos):
                control_attackers[i] = attacker_control_1vs1(game, grid1vs1, value1vs1, current_state, neg2pos)
            else:
                control_attackers[i] = (0.0, 0.0)
        else:  # the attacker is captured or arrived
            control_attackers[i] = (0.0, 0.0)
            
    return control_attackers


def single_1vs1_controller_defender(game, value1vs1, grid1vs1):
    attackers = game.attackers.state.copy()
    defenders = game.defenders.state.copy()
    assert game.NUM_ATTACKERS == 1, "The number of attacker should be 1."
    assert game.NUM_DEFENDERS == 1, "The number of defender should be 1."
    num_defenders = game.NUM_DEFENDERS 
    control_defenders = np.zeros((num_defenders, 2))
    
    a1x, a1y = attackers[0]
    d1x, d1y = defenders[0]
    
    jointstate_1vs1 = (a1x, a1y, d1x, d1y)
    opt_d1, opt_d2 = defender_control_1vs1(game, grid1vs1, value1vs1, jointstate_1vs1)
    
    control_defenders[0] = (opt_d1, opt_d2)

    return control_defenders


def single_1vs1_controller_defender_noise(game, value1vs1, grid1vs1, epsilon=0.3):
    attackers = game.attackers.state.copy()
    defenders = game.defenders.state.copy()
    assert game.NUM_ATTACKERS == 1, "The number of attacker should be 1."
    assert game.NUM_DEFENDERS == 1, "The number of defender should be 1."
    num_defenders = game.NUM_DEFENDERS 
    control_defenders = np.zeros((num_defenders, 2))
    
    a1x, a1y = attackers[0]
    d1x, d1y = defenders[0]
    
    jointstate_1vs1 = (a1x, a1y, d1x, d1y)
    opt_d1, opt_d2 = defender_control_1vs1(game, grid1vs1, value1vs1, jointstate_1vs1)
    
    if np.random.rand() < epsilon:
        # Generate random controls within the range [-1, 1]
        control_defenders[0] = np.random.uniform(-1, 1, 2)
    else:
        control_defenders[0] = (opt_d1, opt_d2)

    return control_defenders


########################################## game-independent controls ##########################################
def optCtrl_1vs0(spat_deriv, uMax, uMode, a_speed):
    """Computes the optimal control (disturbance) for the attacker in a 1 vs. 0 game.

    Parameters:
        spat_deriv (tuple): spatial derivative in all dimensions
        uMax (float): the maximum control value of the attacker
        uMode (str): the control mode of the attacker
        a_speed (float): the speed of the attacker
    
    Returns:
        tuple: a tuple of optimal control of the defender (disturbances)
    """
    opt_a1 = uMax
    opt_a2 = uMax
    deriv1 = spat_deriv[0]
    deriv2 = spat_deriv[1]
    ctrl_len = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
    if uMode == "min":
        if ctrl_len == 0:
            opt_a1 = 0.0
            opt_a2 = 0.0
        else:
            opt_a1 = - a_speed * deriv1 / ctrl_len
            opt_a2 = - a_speed * deriv2 / ctrl_len
    else:
        if ctrl_len == 0:
            opt_a1 = 0.0
            opt_a2 = 0.0
        else:
            opt_a1 = a_speed * deriv1 / ctrl_len
            opt_a2 = a_speed * deriv2 / ctrl_len
    return (opt_a1, opt_a2)


def hj_controller_1vs0(uMode, uMax, a_speed, value1vs0, grid1vs0, attackers, current_status):
    """This function computes the control for the attackers based on the control_attackers. 
       Assume dynamics are single integrator.   

    Args:
        uMode (str): the control mode of the attacker
        uMax (float): the maximum control value of the attacker
        a_speed (float): the speed of the attacker
        value1vs0 (np.ndarray): the value function for 1 vs 0 game with all time slices
        grid1vs0 (Grid): the grid for 1 vs 0 game
        attackers (np.ndarray, (num_players, 2)): the current states of attackers
        current_status (np.ndarray): the current status of attackers
    """
    attackers = attackers.copy()    
    num_attackers = attackers.shape[0]
    current_attackers_status = current_status
    control_attackers = np.zeros((num_attackers, 2))
    for i in range(num_attackers):
        if not current_attackers_status[i]:  # the attacker is free
            neg2pos, pos2neg = find_sign_change1vs0(grid1vs0, value1vs0, attackers[i])
            if len(neg2pos):
                current_value = grid1vs0.get_value(value1vs0[..., 0], list(attackers[i]))
                if current_value > 0:
                    value1vs0 = value1vs0 - current_value
                v = value1vs0[..., neg2pos] 
                spat_deriv_vector = spa_deriv(grid1vs0.get_index(attackers[i]), v, grid1vs0)
                control_attackers[i] = optCtrl_1vs0(spat_deriv_vector, uMax, uMode, a_speed)
            else:
                control_attackers[i] = (0.0, 0.0)
        else:  # the attacker is captured or arrived
            control_attackers[i] = (0.0, 0.0)
            
    return control_attackers


def optDistb_1vs1(spat_deriv, dMax, dMode, d_speed):
    """Computes the optimal control (disturbance) for the defender in a 1 vs. 1 game.
    
    Parameters:
        spat_deriv (tuple): spatial derivative in all dimensions
        dMax (float): the maximum control value of the defender
        dMode (str): the control mode of the defender
        d_speed (float): the speed of the defender
    
    Returns:
        tuple: a tuple of optimal control of the defender (disturbances)
    """
    opt_d1 = dMax
    opt_d2 = dMax
    deriv3 = spat_deriv[2]
    deriv4 = spat_deriv[3]
    distb_len = np.sqrt(deriv3*deriv3 + deriv4*deriv4)
    if dMode == "max":
        if distb_len == 0:
            opt_d1 = 0.0
            opt_d2 = 0.0
        else:
            opt_d1 = d_speed * deriv3 / distb_len
            opt_d2 = d_speed * deriv4 / distb_len
    else:
        if distb_len == 0:
            opt_d1 = 0.0
            opt_d2 = 0.0
        else:
            opt_d1 = -d_speed * deriv3 / distb_len
            opt_d2 = -d_speed * deriv4 / distb_len
            
    return (opt_d1, opt_d2)


def optDistb_2vs1(spat_deriv, dMax, dMode, d_speed):
    """Computes the optimal control (disturbance) for the defender in a 2 vs. 1 game.
    
    Parameters:
        spat_deriv (tuple): spatial derivative in all dimensions
        dMax (float): the maximum control value of the defender
        dMode (str): the control mode of the defender
        d_speed (float): the speed of the defender
    
    Returns:
        tuple: a tuple of optimal control of the defender (disturbances)
    """
    opt_d1 = dMax
    opt_d2 = dMax
    deriv5 = spat_deriv[4]
    deriv6 = spat_deriv[5]
    distb_len = np.sqrt(deriv5*deriv5 + deriv6*deriv6)
    if dMode == "max":
        if distb_len == 0:
            opt_d1 = 0.0
            opt_d2 = 0.0
        else:
            opt_d1 = d_speed * deriv5 / distb_len
            opt_d2 = d_speed * deriv6 / distb_len
    else:
        if distb_len == 0:
            opt_d1 = 0.0
            opt_d2 = 0.0
        else:
            opt_d1 = -d_speed * deriv5 / distb_len
            opt_d2 = -d_speed * deriv6 / distb_len

    return (opt_d1, opt_d2)


def optDistb_1vs2(spat_deriv, dMax, dMode, d_speed):
    """Computes the optimal control (disturbance) for the attacker in a 1 vs. 2 game.
    
    Parameters:
        spat_deriv (tuple): spatial derivative in all dimensions
        dMax (float): the maximum control value of the defender
        dMode (str): the control mode of the defender
        d_speed (float): the speed of the defender
    
    Returns:
        tuple: a tuple of optimal control of the defender (disturbances)
    """
    opt_d1 = dMax
    opt_d2 = dMax
    opt_d3 = dMax
    opt_d4 = dMax
    deriv3 = spat_deriv[2]
    deriv4 = spat_deriv[3]
    deriv5 = spat_deriv[4]
    deriv6 = spat_deriv[5]
    distb_len1 = np.sqrt(deriv3*deriv3 + deriv4*deriv4)
    distb_len2 = np.sqrt(deriv5*deriv5 + deriv6*deriv6)
    if dMode == "max":
        if distb_len1 == 0:
            opt_d1 = 0.0
            opt_d2 = 0.0
        else:
            opt_d1 = deriv3 / distb_len1
            opt_d2 = deriv4 / distb_len1
        if distb_len2 == 0:
            opt_d3 = 0.0
            opt_d4 = 0.0
        else:
            opt_d3 = d_speed*deriv5 / distb_len2
            opt_d4 = d_speed*deriv6 / distb_len2
    else:  # dMode == "min"
        if distb_len1 == 0:
            opt_d1 = 0.0
            opt_d2 = 0.0
        else:
            opt_d1 = -d_speed*deriv3 / distb_len1
            opt_d2 = -d_speed*deriv4 / distb_len1
        if distb_len2 == 0:
            opt_d3 = 0.0
            opt_d4 = 0.0
        else:
            opt_d3 = -d_speed*deriv5 / distb_len2
            opt_d4 = -d_speed*deriv6 / distb_len2

    return (opt_d1, opt_d2, opt_d3, opt_d4)


def hj_controller_1vs1_defender(dMode, dMax, d_speed, value1vs1, grid1vs1, jointstate_1vs1):
    """This function computes the control for the defender based on 1 vs. 1 value function. 
       Assume dynamics are single integrator.   

    Args:
        dMode (str): the control mode of the defender, either "max" or "min"
        dMax (float): the maximum control value of the defender
        d_speed (float): the speed of the defender
        value1vs1 (np.ndarray): the value function for 1 vs 1 game with the final time slices
        grid1vs1 (Grid): the grid for 1 vs 1 game
        jointstate_1vs1 (a1x, a1y, d1x, d1y): the current joint state of one attacker and one defender
    """
    value1vs1s = value1vs1[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid1vs1.get_index(jointstate_1vs1), value1vs1s, grid1vs1)
    opt_d1, opt_d2 = optDistb_1vs1(spat_deriv_vector, dMax, dMode, d_speed)

    return (opt_d1, opt_d2)


def hj_controller_2vs1_defender(dMode, dMax, d_speed, value2vs1, grid2vs1, jointstate_2vs1):
    """This function computes the control for the defender based on 2 vs. 1 value function. 
       Assume dynamics are single integrator.   

    Args:
        dMode (str): the control mode of the defender, either "max" or "min"
        dMax (float): the maximum control value of the defender
        d_speed (float): the speed of the defender
        value1vs1 (np.ndarray): the value function for 2 vs 1 game with the final time slices
        grid1vs1 (Grid): the grid for 2 vs 1 game
        jointstate_2vs1 (a1x, a1y, a2x, a2y, d1x, d1y): the current joint state of two attackers and one defender
    """
    value1vs1s = value2vs1[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid2vs1.get_index(jointstate_2vs1), value1vs1s, grid2vs1)
    opt_d1, opt_d2 = optDistb_2vs1(spat_deriv_vector, dMax, dMode, d_speed)

    return (opt_d1, opt_d2)


def hj_controller_1vs2_defender(dMode, dMax, d_speed, value1vs2, grid1vs2, jointstate_1vs2):
    """This function computes the control for the defender based on 1 vs. 2 value function. 
       Assume dynamics are single integrator.   

    Args:
        dMode (str): the control mode of the defender, either "max" or "min"
        dMax (float): the maximum control value of the defender
        d_speed (float): the speed of the defender
        value1vs2 (np.ndarray): the value function for 1 vs 2 game with the final time slices
        grid1vs2 (Grid): the grid for 1 vs 2 game
        jointstate_1vs2 (a1x, a1y, d1x, d1y, d2x, d2y): the current joint state of one attacker and two defenders
    """
    value1vs2s = value1vs2[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid1vs2.get_index(jointstate_1vs2), value1vs2s, grid1vs2)
    opt_d1, opt_d2, opt_d3, opt_d4 = optDistb_1vs2(spat_deriv_vector, dMax, dMode, d_speed)

    return (opt_d1, opt_d2, opt_d3, opt_d4)


def hj_controller_defenders_independent(attackers, defenders,
                                        assignments, weights, attacker_views,
                                        value1vs1, value2vs1, value1vs2, 
                                        grid1vs1, grid2vs1, grid1vs2):
    """This fuction computes the control for the defenders based on the assignments.
       Assume dynamics are single integrator.

    Args:
        attackers (np.ndarray, (num_attackers, 2)): the current states of attackers
        defenders (np.ndarray, (num_defenders, 2)): the current states of defenders
        assignments (a list of lists): the list of attackers that the defender assigned to capture
        weights (np.ndarray, (num_free_attackers, num_defenders)): the weights for each assignment
        attacker_views (a list of lists): the list of defenders that could capture the attacker
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
        value2vs1 (np.ndarray): the value function for 2 vs 1 game
        value1vs2 (np.ndarray): the value function for 1 vs 2 game
        grid1vs1 (Grid): the grid for 1 vs 1 game
        grid2vs1 (Grid): the grid for 2 vs 1 game
        grid1vs2 (Grid): the grid for 1 vs 2 game

    Returns:
        control_defenders ((ndarray): the control of defenders
    """
    attackers = attackers.copy()
    defenders = defenders.copy()
    num_defenders = defenders.shape[0] 
    control_defenders = np.zeros((num_defenders, 2))
    calculated_defenders = []  # store the calculated defenders which should not calculate the control again
    flag_1vs2 = False

    for j in range(num_defenders):

        if j in calculated_defenders:
            continue
        d1x, d1y = defenders[j]

        if len(assignments[j]) == 2: # defender j capture the attacker selected[j][0] and selected[j][1]
            a1x, a1y = attackers[assignments[j][0]]
            a2x, a2y = attackers[assignments[j][1]]
            jointstate_2vs1 = (a1x, a1y, a2x, a2y, d1x, d1y)
            control_defenders[j] = hj_controller_2vs1_defender('max', 1.0, 1.5, value2vs1, grid2vs1, jointstate_2vs1)
        elif len(assignments[j]) == 1:
            a1x, a1y = attackers[assignments[j][0]]

            if weights[assignments[j][0], j] == 0.5:  # use 1 vs. 2 game based control
                collaborate_defender = attacker_views[assignments[j][0]][-1]
                d2x, d2y = defenders[collaborate_defender]
                jointstate_1vs2 = (a1x, a1y, d1x, d1y, d2x, d2y)
                opt_d1, opt_d2, opt_d3, opt_d4 = hj_controller_1vs2_defender('min', 1.0, 1.5, value1vs2, grid1vs2, jointstate_1vs2)
                control_defenders[j] = (opt_d1, opt_d2)
                control_defenders[collaborate_defender] = (opt_d3, opt_d4)
                calculated_defenders.append(collaborate_defender)
                flag_1vs2 = True
            else:  # use 1 vs. 1 game based control
                jointstate_1vs1 = (a1x, a1y, d1x, d1y)
                control_defenders[j] = hj_controller_1vs1_defender('max', 1.0, 1.5, value1vs1, grid1vs1, jointstate_1vs1)

        elif len(assignments[j]) == 0: # defender j could not capture any of attackers
            control_defenders[j] = (0.0, 0.0)
        else:
            raise ValueError("The number of attackers assigned to one defender should be less than 3.")

    return control_defenders, flag_1vs2