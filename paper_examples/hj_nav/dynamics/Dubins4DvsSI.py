import heterocl as hcl
import numpy as np

""" 4D RELATIVE DYNAMIC between DubinsCar4D and SingleIntegrator
 x_dot = v * cos(theta) + d_1
 y_dot = v * sin(theta) + d_2
 v_dot = a
 theta_dot = w
 """


class Dubins4DvsSI():
    def __init__(self, x=[0, 0, 0, 0], 
                 uMin=[-1, -1], uMax=[1, 1], 
                 dMin=[-1.0, -0.5], dMax=[1.0, 0.5], 
                 uMode="min", dMode="max", speed_SI=1.0) -> None:
        """
        Relative dynamics between DubinsCar4D and SingleIntegratorConstantSpeed (SI).
        The SI is the pursuer, and the Dubins4D is the evader.
        We fix the origin to the SI and assume the Dubin's heading angle remains the same.

        Dynamics:
            x1_dot = ve*cos(theta) - vp*cos(alpha)  # x1 = xe - xp
            x2_dot = ve*sin(theta) - vp*sin(alpha)  # x2 = ye - yp
            x3_dot = ae                             # x3 = ve
            x4_dot = we                             # x4 = theta


        Control:
            u[0] = cos(alpha)
            u[1] = sin(alpha)

        Disturbance:
            d[0] = ae
            d[1] = we
        """
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        assert(uMode in ["min", "max"])
        self.uMode = uMode
        if uMode == "min":
            assert(dMode == "max")
        else:
            assert(dMode == "min")
        self.dMode = dMode
        self.speed_p = speed_SI
    
    def opt_ctrl(self, t: float, state, spat_deriv):
        opt_cos_alpha = hcl.scalar(0, "opt_cos_alpha")
        opt_sin_alpha = hcl.scalar(0, "opt_sin_alpha")
        u3 = hcl.scalar(0, "u3")
        u4 = hcl.scalar(0, "u4")

        denom = hcl.sqrt(spat_deriv[0] * spat_deriv[0] + spat_deriv[1] * spat_deriv[1])

        if self.uMode == "max":
            with hcl.if_(denom == 0):
                opt_cos_alpha[0] = 0.0
                opt_sin_alpha[0] = 0.0
            with hcl.else_():
                opt_cos_alpha[0] = -spat_deriv[0] / denom
                opt_sin_alpha[0] = -spat_deriv[1] / denom
        else:
            with hcl.if_(denom == 0):
                opt_cos_alpha[0] = 0.0
                opt_sin_alpha[0] = 0.0
            with hcl.else_():
                opt_cos_alpha[0] = spat_deriv[0] / denom
                opt_sin_alpha[0] = spat_deriv[1] / denom

        return (opt_cos_alpha[0], opt_sin_alpha[0], u3[0], u4[0])
        
    
    def opt_dstb(self, t, state, spat_deriv):
        opt_ae = hcl.scalar(self.dMax[0], "opt_ae")
        opt_we = hcl.scalar(self.dMax[1], "opt_we")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")

        coefficient_ae = spat_deriv[2]
        coefficient_we = spat_deriv[3]

        if self.dMode == "min":
            with hcl.if_(coefficient_ae > 0):
                opt_ae[0] = self.dMin[0]
            with hcl.if_(coefficient_we > 0):
                opt_we[0] = self.dMin[1]
        else:
            with hcl.if_(coefficient_ae < 0):
                opt_ae[0] = self.dMin[0]
            with hcl.if_(coefficient_we < 0):
                opt_we[0] = self.dMin[1]

        return (opt_ae[0], opt_we[0], d3[0], d4[0])

    def dynamics(self, t, state, u, d):
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")

        x1_dot[0] = state[2] * hcl.cos(state[3]) - self.speed_p * u[0]
        x2_dot[0] = state[2] * hcl.sin(state[3]) - self.speed_p * u[1]
        x3_dot[0] = d[0]
        x4_dot[0] = d[1]

        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0])
    
    def opt_ctrl_np(self, state: np.ndarray, spat_deriv: np.ndarray) -> np.ndarray:
        """Not implemented, not required"""
        return

    def opt_dstb_np(self, state: np.ndarray, spat_deriv: np.ndarray) -> np.ndarray:
        """Not implemented, not required"""
        return

    def dynamics_np(
        self, state: np.ndarray, t: float, ctrl: np.ndarray, dstb: np.ndarray
    ) -> np.ndarray:
        """Not implemented, not required"""
        return

    def dyn_state_to_stateaction(self, t: float, x: np.ndarray, u: np.ndarray):
        """Not implemented, not required"""
        return

    def stateaction_to_dyn_state(self, state) -> np.ndarray:
        """Not implemented, not required"""
        return