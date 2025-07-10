'''Basic class for all dynamics.

Hanyang Hu, 20250709
'''
from abc import ABC, abstractmethod
import numpy as np


class BasicDynamic(ABC):    
    def __init__(self, 
                 control_freq: float, 
                 state_lower_bound: np.ndarray,
                 state_upper_bound: np.ndarray,
                 action_lower_bound: np.ndarray,
                 action_upper_bound: np.ndarray
                 ):
        """
        Abstract base class for continuous-time dynamical systems.

        Args:
            control_frequency (float): Control update frequency in Hz.
            state_lower_bound (np.ndarray): Lower bounds for each dimension of the state space (shape: (state_dim,)).
            state_upper_bound (np.ndarray): Upper bounds for each dimension of the state space (shape: (state_dim,)).
            action_lower_bound (np.ndarray): Lower bounds for each dimension of the action space (shape: (action_dim,)).
            action_upper_bound (np.ndarray): Upper bounds for each dimension of the action space (shape: (action_dim,)).
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq  # time step
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound
    
    def check_state_bounds(self, state: np.ndarray) -> np.ndarray:
        """
        Check the new state bounds after stepping forward.
        
        """
        return np.clip(state, self.state_lower_bound, self.state_upper_bound)
    
    def check_action_bounds(self, action: np.ndarray) -> np.ndarray:
        """
        Check the action bounds before executing.
        
        """
        return np.clip(action, self.action_lower_bound, self.action_upper_bound)

    @abstractmethod
    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute the time derivative of the system given the current state and action.
        This method defines the system's differential equations.
        Args:
            state (np.ndarray): Current system state (shape: (M, state_dim)).
            action (np.ndarray): Current control input (shape: (M, action_dim)).

        Returns:
            np.ndarray: Time derivative of the state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Step forward and return the next state after executing the action.

        Must be implemented in a subclass.

        """
        raise NotImplementedError