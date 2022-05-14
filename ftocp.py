from casadi import *
from numpy import *
import numpy as np
import time


# define the finite time optimal control problem used for collecting initial data
# for both the original-parameter setting and changed-parameter setting
class FTOCPNLP(object):
    # parameters initiation
    def __init__(self, N, Q, R, Qf, goal, dt, bx, bu):
        # Define MPC-related variables
        self.N = N
        self.n = Q.shape[1]
        self.d = R.shape[1]
        self.bx = bx
        self.bu = bu
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.goal = goal
        self.dt = dt
        self.buildFTOCP()
        self.solverTime = []

# define the solve method for solving the finite time optimal control problem
    def solve(self, x0, verbose=False):
        # set states and inputs box constraints
        self.lbx = x0.tolist() + (-self.bx).tolist() * (self.N) + (-self.bu).tolist() * self.N
        self.ubx = x0.tolist() + (self.bx).tolist() * (self.N) + (self.bu).tolist() * self.N
        # solve the non-linear problem
        start = time.time()
        sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
        end = time.time()
        delta = end - start
        self.solverTime.append(delta)
        # check if there exists a feasible solution
        if (self.solver.stats()['success']):
            self.feasible = 1
            x = sol["x"]
            self.qcost = sol["f"]
            self.xPred = np.array(x[0:(self.N + 1) * self.n].reshape((self.n, self.N + 1))).T
            self.uPred = np.array(
                x[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape((self.d, self.N))).T
            self.mpcInput = self.uPred[0][0]
            print("xPredicted:")
            print(self.xPred)
            print("uPredicted:")
            print(self.uPred)
            print("Cost:")
            print(self.qcost)
            print("NLP Solver Time: ", delta, " seconds.")
        else:
            self.xPred = np.zeros((self.N + 1, self.n))
            self.uPred = np.zeros((self.N, self.d))
            self.mpcInput = []
            self.feasible = 0
            print("Unfeasible")
        return self.uPred[0]

    def buildFTOCP(self):
        n = self.n
        d = self.d
        # define the CASADI solver variables
        X = SX.sym('X', n * (self.N + 1))
        U = SX.sym('U', d * self.N)
        # define dynamic constraints used by the CASADI solver
        self.constraint = []
        for i in range(0, self.N):
            X_next = self.NonLinearBicycleModel(X[n * i:n * (i + 1)], U[d * i:d * (i + 1)])
            for j in range(0, self.n):
                self.constraint = vertcat(self.constraint, X_next[j] - X[n * (i + 1) + j])
        # define cost used by the CASADI solver
        self.cost = 0
        for i in range(0, self.N):
            self.cost = self.cost + (X[n * i:n * (i + 1)] - self.goal).T @ self.Q @ (X[n * i:n * (i + 1)] - self.goal)
            self.cost = self.cost + U[d * i:d * (i + 1)].T @ self.R @ U[d * i:d * (i + 1)]
        self.cost = self.cost + (X[n * self.N:n * (self.N + 1)] - self.goal).T @ self.Qf @ (
                X[n * self.N:n * (self.N + 1)] - self.goal)
        # setup the CASADI.ipopt(interior point optimizer)
        opts = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        nlp = {'x': vertcat(X, U), 'f': self.cost, 'g': self.constraint}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        # Set lower and upper bound of inequality constraint to zero to force n*N state dynamics
        self.lbg_dyanmics = [0] * (n * self.N)
        self.ubg_dyanmics = [0] * (n * self.N)

    def NonLinearBicycleModel(self, x, u):
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
        Cf = 1.2
        Bf = 1.0
        Dr = 0.8 * m * 9.81 / 2.0
        Cr = 1.25  # 1.25
        Br = 1.0
        # 2 inputs, first one is the acceleration, second is the steering
        a = u[0]
        steer = u[1]
        # here use a tiny penalty to make the Jacobian calculation in the CASADI solver correct(force denominator not 0)
        x[3] = x[3] + 0.000001
        # slip angles for the dynamic bicycle model
        alpha_f = steer - np.arctan2(x[4] + lf * x[2], x[3])
        alpha_r = - np.arctan2(x[4] - lf * x[2], x[3])
        # the lateral forces in the body frame for front and rear wheels
        Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
        Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
        # equations for 6 states in the dynamic bicycle model
        x_next = x[0] + self.dt * (x[3] * np.cos(x[2]) - x[4] * np.sin(x[2]))
        y_next = x[1] + self.dt * (x[3] * np.sin(x[2]) + x[4] * np.cos(x[2]))
        theta_next = x[2] + self.dt * x[5]
        vx_next = x[3] + self.dt * (a - 1 / m * Fyf * np.sin(steer) + x[4] * x[5])
        vy_next = x[4] + self.dt * (1 / m * (Fyf * np.cos(steer) + Fyr) - x[3] * x[5])
        yaw_next = x[5] + self.dt * (1 / Iz * (lf * Fyf * np.cos(steer) - Fyr * lr))
        # return updated/calculated 6 states
        state_next = [x_next, y_next, theta_next, vx_next, vy_next, yaw_next]
        return state_next
