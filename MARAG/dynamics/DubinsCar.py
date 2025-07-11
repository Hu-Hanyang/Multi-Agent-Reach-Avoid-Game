import numpy as np
from typing import Tuple
from MARAG.dynamics.BasicDynamic import BasicDynamic


class DubinsCar3D(BasicDynamic):
    '''3D DubinsCar agents dynamics with one control input omega.
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = omega  # theta: [-pi, pi) in radian
    '''
    def __init__(self,
                 control_freq: float,
                 state_lower_bound: np.ndarray,
                 state_upper_bound: np.ndarray,
                 action_lower_bound: np.ndarray,
                 action_upper_bound: np.ndarray,
                 speed):  # Hanyang: for real world experiments, change uMax from 2.84 to 1.0
        super().__init__(control_freq=control_freq,
                         state_lower_bound=state_lower_bound,
                         state_upper_bound=state_upper_bound,
                         action_lower_bound=action_lower_bound,
                         action_upper_bound=action_upper_bound,
                         )
        self.speed = speed  # constant speed 
    
    def dynamics(self, state, action) -> np.ndarray:
        """
        Args:
            state (np.ndarray): Current system state, shape (M, 3).
            action (np.ndarray): Control input for each agent, shape (M, 1).
        
        Returns:
            next_state (np.ndarray): Next state for each agent, shape (M, 3).
        """
        clipped_action = self.check_action_bounds(action)
        dx = self.speed * np.cos(state[:, 2])
        dy = self.speed * np.sin(state[:, 2])
        dtheta = clipped_action[:, 0]
        
        return np.stack([dx, dy, dtheta], axis=1)

    def forward(self, state, action) -> np.ndarray:
        """        
        Args:
            state (np.ndarray): Current system state, shape (M, 3).
            action (np.ndarray): Control input for each agent, shape (M, 1).
            
        Returns:
            next_state (np.ndarray): Next state for each agent, shape (M, 3).
        """
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, action)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, action)
        k4 = self.dynamics(state + self.dt * k3, action)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Wrap angle to [-pi, pi)
        next_state[:, 2] = (next_state[:, 2] + np.pi) % (2 * np.pi) - np.pi

        clipped_next_state = self.check_state_bounds(next_state)
        
        return clipped_next_state
            

class DubinsCar3D2Ctrl(BasicDynamic):
    '''3D DubinsCar agents dynamics with 2 control inputs [v, omega].
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = omega
    '''
    def __init__(self,
                 control_freq: float,
                 state_lower_bound: np.ndarray,
                 state_upper_bound: np.ndarray,
                 action_lower_bound: np.ndarray,
                 action_upper_bound: np.ndarray):
        super().__init__(control_freq=control_freq,
                         state_lower_bound=state_lower_bound,
                         state_upper_bound=state_upper_bound,
                         action_lower_bound=action_lower_bound,
                         action_upper_bound=action_upper_bound,
                         )
    
    def dynamics(self, state, action) -> np.ndarray:
        """
        Args:
            state (np.ndarray): Current system state, shape (M, 3).
            action (np.ndarray): Control input for each agent, shape (M, 1).
        
        Returns:
            next_state (np.ndarray): Next state for each agent, shape (M, 3).
        """
        clipped_action = self.check_action_bounds(action)
        dx = clipped_action[:, 0] * np.cos(state[:, 2])
        dy = clipped_action[:, 0] * np.cos(state[:, 2])
        dtheta = clipped_action[:, 1]
        
        return np.stack([dx, dy, dtheta], axis=1)
    
    def forward(self, state, action) -> np.ndarray:
        """        
        Args:
            state (np.ndarray): Current system state, shape (M, 3).
            action (np.ndarray): Control input for each agent, shape (M, 1).
            
        Returns:
            next_state (np.ndarray): Next state for each agent, shape (M, 3).
        """
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, action)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, action)
        k4 = self.dynamics(state + self.dt * k3, action)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Wrap angle to [-pi, pi)
        next_state[:, 2] = (next_state[:, 2] + np.pi) % (2 * np.pi) - np.pi

        clipped_next_state = self.check_state_bounds(next_state)
        
        return clipped_next_state
