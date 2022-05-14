import numpy as np


class systemdy(object):
    # parameters initiation
    def __init__(self, x0, dt):
        self.x = [x0]
        self.u = []
        self.w = []
        # initial conditions for the 6 states
        self.x0 = x0
        # simulation step-size for the simulator
        self.dt = dt

    # apply input and dynamic bicycle model for the simulator
    def applyInput(self, ut):
        self.u.append(ut)
        xt = self.x[-1]
        # below are parameters from the Berkeley Autonomous Race Car Platform
        # Mass
        m = 1.98
        # distance from the center of gravity to front and rear axles
        lf = 0.125
        lr = 0.125
        # moment of inertia about the vertical axis passing through the center of gravity
        Iz = 0.024
        # track specific parameters for the tire force curves
        Df = 0.8 * m * 9.81 / 2.0
        Cf = 1.25
        Bf = 1.0
        Dr = 0.8 * m * 9.81 / 2.0
        Cr = 1.25
        Br = 1.0
        # slip angles for the dynamic bicycle model
        alpha_f = ut[1] - np.arctan2(xt[4] + lf * xt[2], xt[3])
        alpha_r = - np.arctan2(xt[4] - lf * xt[2], xt[3])
        # the lateral forces in the body frame for front and rear wheels
        Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
        Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
        # equations for 6 states in the dynamic bicycle model
        x_next = xt[0] + self.dt * (xt[3] * np.cos(xt[2]) - xt[4] * np.sin(xt[2]))
        y_next = xt[1] + self.dt * (xt[3] * np.sin(xt[2]) + xt[4] * np.cos(xt[2]))
        theta_next = xt[2] + self.dt * xt[5]
        vx_next = xt[3] + self.dt * (ut[0] - 1 / m * Fyf * np.sin(ut[1]) + xt[5] * xt[4])
        vy_next = xt[4] + self.dt * (1 / m * (Fyf * np.cos(ut[1]) + Fyr) - xt[5] * xt[3])
        yaw_next = xt[5] + self.dt * (1 / Iz * (lf * Fyf * np.cos(ut[1]) - Fyr * lr))
        # return updated/calculated 6 states
        state_next = np.array([x_next, y_next, theta_next, vx_next, vy_next, yaw_next])
        self.x.append(state_next)

# reset initial condition for the simulator based on the dynamic bicycle model
    def reset_IC(self):
        self.x = [self.x0]
        self.u = []
        self.w = []
