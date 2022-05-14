import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import systemdy
from gpftocp import FTOCPNLP
from params_mpc import gt
from dynmodel import Dynamic
from gps import GPModel

# safe the GP corrected data
SAVE_RESULTS = True
# create label to record which parameter is changed
LABEL = 'cf1.2'
# define MPC-related variables
# horizon
N = 25
# number of states
n = 6
# number of inputs
d = 2
# initial state
x0_dy = np.array([0, 0, 0, 3, 0, 0])
# discrete time
dt = 0.12
# dynamic bicycle model used in the simulator
sys_dy = systemdy(x0_dy, dt)
# simulation time
maxTime = 25
# goal/target states
xRef_dy = np.array([10, 10, np.pi/2, 3, 0, 0])
# for inputs
R = 1*np.eye(d)
# for states
Q_dy = 1*np.eye(n)
# for the terminal one
Qf_dy = np.diag([11.8, 2.0, 50.0, 280.0, 100.0, 1000.0])
# box constraints for states
bx_dy = np.array([15, 15, 15, 15, 15, 15])
# box constraints for inputs
bu = np.array([10, 0.5])
# load the GP corrected data for vx, vy and omega
with open('/home/ye/Downloads/GP_NMPC-main/src/vxgpcf1.2.pickle', 'rb') as f:
    (vxmodel, vxxscaler, vxyscaler) = pickle.load(f)
vxgp = GPModel('vx', vxmodel, vxyscaler)
with open('/home/ye/Downloads/GP_NMPC-main/src/vygpcf1.2.pickle', 'rb') as f:
    (vymodel, vyxscaler, vyyscaler) = pickle.load(f)
vygp = GPModel('vy', vymodel, vyyscaler)
with open('/home/ye/Downloads/GP_NMPC-main/src/omegagpcf1.2.pickle', 'rb') as f:
    (omegamodel, omegaxscaler, omegayscaler) = pickle.load(f)
omegagp = GPModel('omega', omegamodel, omegayscaler)

# GP model after correction used in the MPC
gpmodels = {
    'vx': vxgp,
    'vy': vygp,
    'omega': omegagp,
    'xscaler': vxxscaler,
    'yscaler': vxyscaler,
}
# parameters for the dynamic bicycle model used here
params = gt()
# initiate related simulation functions for the dynamic model
model = Dynamic(**params)
# initiate variables for the simulation process based on the dynamic model
states = np.zeros([n, maxTime + 1])
dstates = np.zeros([n, maxTime + 1])
inputs = np.zeros([d, maxTime + 1])
timearr = np.linspace(0, maxTime+1, maxTime + 1) * dt
Ffy = np.zeros([maxTime + 1])
Frx = np.zeros([maxTime + 1])
Fry = np.zeros([maxTime + 1])
hstates = np.zeros([n, N + 1])
hstates2 = np.zeros([n, N + 1])
# solve the GP-corrected MPC problem
nlp_dy = FTOCPNLP(N, Q_dy, R, Qf_dy, xRef_dy, dt, bx_dy, bu, gpmodels)
# solve for the input from the initial state
ut_dy = nlp_dy.solve(x0_dy)
x_init = x0_dy
# only the last 3 states, vx,vy and omega are related to the parameter changes in the dynamic model
dstates[0, 0] = x_init[3]
states[:, 0] = x_init
inputs[:, 0] = ut_dy
print('Starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))
# initial variables that used to store the MPC predicted states, inputs and cost
xPredNLP_dy = []
uPredNLP_dy = []
CostSolved_dy = []
# initiate MPC-solver loop
for t in range(0, maxTime):
    # simulation for the closed-loop simulator
    uprev = inputs[:, t - 1]
    x0 = states[:, t]
    xt_dy = sys_dy.x[-1]
    start = time.time()
    ut_dy = nlp_dy.solve(xt_dy)
    end = time.time()
    # simulation for the MPC-predicted process
    umpc = nlp_dy.uPred
    xmpc = nlp_dy.xPred
    inputs[:, t + 1] = umpc[0, :]
    print("Iteration: {}, time to solve: {:.2f}".format(t, end - start))
    xPredNLP_dy.append(nlp_dy.xPred)
    uPredNLP_dy.append(nlp_dy.uPred)
    CostSolved_dy.append(nlp_dy.qcost)
    # apply the GP-corrected states
    x_next, dxdt_next = model.sim_continuous(states[:, t], inputs[:, t].reshape(-1, 1), [0, dt])
    states[:, t + 1] = x_next[:, -1]
    dstates[:, t + 1] = dxdt_next[:, -1]
    Ffy[t + 1], Frx[t + 1], Fry[t + 1] = model.calc_forces(states[:, t], inputs[:, t])
    sys_dy.applyInput(ut_dy)
    hstates[:, 0] = x0[:n]
    hstates2[:, 0] = x0[:n]
    for j in range(N):
        x_next, dxdt_next = model.sim_continuous(hstates[:n, j], umpc[j, :].reshape(-1, 1), [0, dt])
        hstates[:, j + 1] = x_next[:, -1]
        hstates2[:, j + 1] = xmpc[j + 1, :]
# states and inputs for the simulated closed-loop trajectories
x_cl_nlp_dy = np.array(sys_dy.x)
u_cl_nlp_dy = np.array(sys_dy.u)
# save the GP-corrected MPC results
if SAVE_RESULTS:
    np.savez(
        '/home/ye/Downloads/GP_NMPC-main/src/DYN-GPNMPC-{}.npz'.format(LABEL),
        time=timearr,
        states=states,
        dstates=dstates,
        inputs=inputs,
    )

# plot coding area
arr = np.array(xPredNLP_dy)
arr_2 = arr.reshape(650, 6)
arr_1 = np.array(sys_dy.x)
arr_3 = np.zeros(26)
for i in range(26):
    arr_3[i] = arr_2[i, 3]

# comparison plot for the x position between MPC-predicted results and the closed-loop results
plt.figure()
time = np.linspace(0, 25, 26)
for t in range(0, maxTime):
    if t == 0:
        plt.plot(xPredNLP_dy[t][:, 0], '--.b', label='NLP(MPC)-predicted x position')
    else:
        time_1 = np.arange(t, 26)
        plt.plot(time_1, xPredNLP_dy[t][0:26-t, 0], '--.b')
plt.plot(time, arr_1[:, 0], '-*r', label="Close-loop simulated x position")
plt.xlabel('Time')
plt.ylabel('X-Position')
plt.legend()
plt.show()

# MPC predicted trajectory at each time step
for timeToPlot in [0, 10]:
    plt.figure()
    plt.plot(xPredNLP_dy[timeToPlot][:,0], xPredNLP_dy[timeToPlot][:,1], '--.b', label="Simulated trajectory using NLP-aided MPC at time $t = $"+str(timeToPlot))
    plt.plot(xPredNLP_dy[timeToPlot][0,0], xPredNLP_dy[timeToPlot][0,1], 'ok', label="$x_t$ at time $t = $"+str(timeToPlot))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(-1,15)
    plt.ylim(-1,15)
    plt.legend()
    plt.show()

# comparison plot for the trajectory between MPC-predicted results and the closed-loop results
plt.figure()
for t in range(0, maxTime):
    if t == 0:
        plt.plot(xPredNLP_dy[t][:,0], xPredNLP_dy[t][:,1], '--.b', label='Simulated trajectory using NLP-aided MPC')
    else:
        plt.plot(xPredNLP_dy[t][:,0], xPredNLP_dy[t][:,1], '--.b')
plt.plot(x_cl_nlp_dy[:,0], x_cl_nlp_dy[:,1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

# comparison plot for the trajectory between MPC-predicted results and the closed-loop results at first time slot
plt.figure()
plt.plot(xPredNLP_dy[0][:,0], xPredNLP_dy[0][:,1], '--.b', label='Simulated trajectory using NLP-aided MPC')
plt.plot(x_cl_nlp_dy[:,0], x_cl_nlp_dy[:,1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

# comparison plot for the acceleration between MPC-predicted results and the closed-loop results
plt.figure()
plt.plot(u_cl_nlp_dy[:,0], '-*r', label="Closed-loop input: Acceleration")
plt.plot(uPredNLP_dy[0][:,0], '-ob', label="NLP(MPC) predicted input: Acceleration")
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

# comparison plot for the steering between MPC-predicted results and the closed-loop results
plt.figure()
plt.plot(u_cl_nlp_dy[:,1], '-*r', label="Closed-loop input: Steering")
plt.plot(uPredNLP_dy[0][:,1], '-ob', label="NLP(MPC) predicted input: Steering")
plt.xlabel('Time')
plt.ylabel('Steering')
plt.legend()
plt.show()

# NLP-aided MPC prediction for the trajectory at the first time slot
plt.figure()
plt.plot(xPredNLP_dy[0][:,0], xPredNLP_dy[0][:,1], '-*r', label='Solution from the NLP(MPC) prediction')
plt.title('Simulated trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

# comparison plot for vx between MPC-predicted results and the closed-loop results
plt.figure()
plt.plot(xPredNLP_dy[0][:,3], '-*r', label='NLP-aided MPC performance')
plt.plot(x_cl_nlp_dy[:,3], 'ok', label='Closed-loop performance')
plt.xlabel('Time')
plt.ylabel('Velocity of the x-axis')
plt.legend()
plt.show()

# Plot of the NLP-aided MPC cost from the NLP CASADI solver
plt.figure()
plt.plot(CostSolved_dy, '-ob')
plt.xlabel('Time')
plt.ylabel('Iteration cost')
plt.legend()
plt.show()
