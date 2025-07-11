import numpy as np
from typing import Tuple
from MARAG.dynamics.BasicDynamic import BasicDynamic
 
 
class SingleIntegrator(BasicDynamic):
    '''2D SingleIntegrator agents dynamics.
    x_dot = v * u1
    y_dot = v * u2
    '''
    def __init__(self, 
                 control_freq: float,
                 state_lower_bound: np.ndarray,
                 state_upper_bound: np.ndarray,
                 action_lower_bound: np.ndarray,
                 action_upper_bound: np.ndarray,
                 speed=1.0):
        super().__init__(control_freq=control_freq,
                         state_lower_bound=state_lower_bound,
                         state_upper_bound=state_upper_bound,
                         action_lower_bound=action_lower_bound,
                         action_upper_bound=action_upper_bound,
                         )
        self.speed = speed

    def dynamics(self, state, action) -> np.ndarray:
        """
        Args:
            state (np.ndarray): Current system state, shape (M, 2).
            action (np.ndarray): Control input for each agent, shape (M, 2).
        Returns:
            (np.ndarray): Derivative of the state, shape (M, 2)
        """
        clipped_action = self.check_action_bounds(action)
        dx = self.speed * clipped_action[:, 0]
        dy = self.speed * clipped_action[:, 1]
        
        return np.stack([dx, dy], axis=1)

    def forward(self, state, action) -> np.ndarray:
        """        
        Args:
            state (np.ndarray): Current system state, shape (M, 2).
            action (np.ndarray): Control input for each agent, shape (M, 2).
        
        Returns:
            next_state (np.ndarray): Next state for each agent, shape (M, 2).
        """
        k1 = self.dynamics(state, action)  # shape (M, 2)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, action)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, action)
        k4 = self.dynamics(state + self.dt * k3, action)

        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        clipped_next_state = self.check_state_bounds(next_state)
        
        return clipped_next_state