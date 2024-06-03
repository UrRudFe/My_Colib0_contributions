import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control
from scipy.io import savemat

# Global variables to ensure exchange between functions
global Mthr_instance       # ULFS stages - M threshold settings
global fthr_instance       # ULFS stages - f threshold settings
global UFLS_stages         # ULFS stages - load decrease 
global fLIM                # Frequency limit for M calculation
global f_store             # array of f values from numerical calculations
global M_store             # array of M values from numerical calculations
global T_store             # array of t values from numerical calculations
global f_M_function        # Anonymous fthr(M) tripping function
global f_rated             # rated frequency value [Hz]
global u_feedback_prev     # UFLS value

# Initialization of global variables
f_store = []
M_store = []
T_store = []
u_feedback_prev = 0

# UFLS settings
fthr_instance = [49.0, 48.95, 48.90, 48.85, 48.80, 48.75]
Mthr_instance = [ 2.0,  1.80,  1.60,  1.40,  1.20,  1.00]
UFLS_stages   = [0.10,  0.10,  0.10,  0.10,  0.10,  0.10]
fLIM          = 47.5

# Anonymous fthr(M) tripping function
# 1 = No UFLS
# 2 = Conventional UFLS
# 3 = L-shaped fthr(M) function
# 4 = ellipse-shaped fthr(M) function

UFLS_type = 4

if UFLS_type == 1:
    f_M_function = lambda M, fset, Mset: (M >= 0) * 0
elif UFLS_type == 2:
    f_M_function = lambda M, fset, Mset: (M >= 0) * fset
elif UFLS_type == 3:
    f_M_function = lambda M, fset, Mset: (M <= Mset) * fset + (M > Mset) * fLIM
elif UFLS_type == 4:
    f_M_function = lambda M, fset, Mset: fLIM + np.sqrt(np.maximum(0, (1 - (M**2) / (Mset**2)) * ((fset - fLIM)**2)))

# Simulation duration and timestep
simulation_duration = 10      # seconds
simulation_timestep = 0.001   # seconds

# Generate time vector for simulation
time = np.arange(0, simulation_duration + simulation_timestep, simulation_timestep)

# Define initial power imbalance with a step function u
u = np.zeros_like(time)  # Initialization
K = -0.70                # Step amplitude
u[time >= 1] = K         # Step at t=1
udot = np.concatenate(([0], (u[1:] - u[:-1]) / simulation_timestep))  # Time derivative of u

# Frequency Response Model (FRM) parameters
H  = 4           # Inertia constant
D  = 1           # Damping constant
FH = 0.3         # Feedback gain
R  = 0.05        # Droop
KM = 0.95        # Feedback loop gain
TR = 8           # Time constant of the feedback loop
f_rated = 50     # Rated system frequency [Hz]

# Direct FRM branch - swing equation
num_F1 = [1]                  # numerator
den_F1 = [2 * H, D]           # denominator
F1 = control.TransferFunction(num_F1, den_F1)      # create transfer function

# Feedback FRM branch - frequency control
num_FB1 = [FH * TR * KM, KM]  # numerator
den_FB1 = [TR * R, R]         # denominator
FB1 = control.TransferFunction(num_FB1, den_FB1)   # create transfer function

# Overall FRM transfer function
G = F1 / (1 + F1 * FB1)       # FRM transfer function

# Convert G transfer function to state-space
num_G = G.num
den_G = G.den
sys = control.tf(num_G, den_G)
A, B, C, D = control.ssdata(sys)

# Initial conditions
x0 = np.zeros((3, 1)).flatten()

# Simulate the step response using transfer function approach
t, Gu = control.forced_response(sys, T=time, U=u, X0=x0)

# UFLS function
def UFLS(f, dfdt, prev_u_feedback):
    global Mthr_instance
    global fthr_instance
    global UFLS_stages
    global fLIM
    global f_M_function

    # settings of UL UFLS
    f_thrs       = fthr_instance   # frequency thresholds in Hz
    thrs_sizes   = UFLS_stages     # corresponding stage sizes 
    M_thrs       = Mthr_instance   # M thresholds in sec

    # setting limits to dfdt (always negative)
    if dfdt>-1e-15:
        dfdt_lim = -1e-15
    elif dfdt<-1e+9:
        dfdt_lim = -1e+9
    else:
        dfdt_lim = dfdt

    # calculate M
    M = (fLIM-f)/dfdt_lim

    # current status of u_feedback
    u_status = prev_u_feedback 

    for idx in range(len(f_thrs)): # Iterate over all UFLS stages
        trip_threshold = f_M_function(M, f_thrs[idx], M_thrs[idx]) # Calculate M-dependent f threshold
        if f <= trip_threshold and u_status < sum(thrs_sizes[:idx + 1]): # Check if threshold is violated and stage is not activated yet
            u_status = sum(thrs_sizes[:idx + 1]) # Activate stage
        u_feedback = u_status # assign to UFLS output

    return u_feedback, M

# Function to be solved by numerical integration (solve_ivp, RK45)
def equations(t, x, time, u):
    global f_store
    global M_store
    global T_store
    global u_feedback_prev

    # setting up initial power deficit   
    u_set0 = np.interp(t, time, u)
    udot0  = np.interp(t, time, udot)

    # calculating the system output and its derivative in CURRENT state
    u_current    = u_set0 + u_feedback_prev
    xdot_current = A.dot(x) + B.dot(u_current).reshape(-1)
    y_current    = C.dot(x) + D.dot(u_current)
    ydot_current = C.dot(xdot_current) + D.dot(udot0)
 
    # calculate frequency and its derivative in CURRENT state
    f    = f_rated * y_current.reshape(-1) + f_rated
    dfdt = f_rated * ydot_current.reshape(-1)
 
    # call UFLS function
    [u_feedback, M] = UFLS(f, dfdt, u_feedback_prev)
    # update the previous value of u_feedback
    u_feedback_prev = u_feedback
    
    # add feedback to initial power deficit
    u_set = u_set0 + u_feedback
    
    # calculate system equations
    xdot = A.dot(x) + B.dot(u_set).reshape(-1)
    
    # store to global variables
    f_store.append(f)
    M_store.append(M)
    T_store.append(t)

    return xdot

# Solve the system using numerical integration (solve_ivp, RK45)
sol = solve_ivp(lambda t, x: equations(t, x, time, u), [time[0], time[-1]], x0, t_eval=time, method='RK45', max_step=simulation_timestep)

# Extract the solution
t_ss = sol.t
rez_ss = sol.y

# calculate output variable (FRM frequency) using state variables
y_ss = C.dot(sol.y) 

# Plot the results
plt.subplot(2,2,1)
plt.plot(t_ss, f_rated+f_rated*y_ss.reshape(-1), linestyle='-', linewidth=3, label='Numeric Integration')
plt.plot(t, f_rated+f_rated*Gu, linestyle='-', linewidth=3, label='Without UFLS (pure FRM Transfer Function)')
plt.xlim([0, simulation_duration])
plt.ylim([46, f_rated])
plt.grid()
plt.title('Frequency with respect to time')

plt.subplot(2,2,3)
plt.plot(T_store, M_store, linestyle='-', linewidth=3, label='Numeric Integration')
plt.xlim([0, simulation_duration])
plt.ylim([0, 10])
plt.grid()
plt.title('Frequency stability margin')

plt.subplot(2,2,(2,4))
plt.plot(M_store, f_store, linestyle='-', linewidth=3, label='Numeric Integration')

x_fm = np.arange(0,5,0.01) 
for indx in range(len(fthr_instance)):
    y_fm = f_M_function(x_fm, fthr_instance[indx], Mthr_instance[indx])
    plt.plot(x_fm, y_fm, 'k', linewidth=0.5)

plt.xlim([0, 5])
plt.ylim([fLIM, f_rated])
plt.grid()
plt.title('f-M plane')

plt.show()

##
numerical = []
numerical.append(t_ss)
numerical.append(f_rated+f_rated*y_ss.reshape(-1))
savemat('python_model.mat', {'numerical': numerical})

transferFunction = []
transferFunction.append(t)
transferFunction.append(f_rated+f_rated*Gu)
savemat('transferFunction.mat', {'transferFunction': transferFunction})

fMdata = []
fMdata.append(T_store)
fMdata.append(f_store)
fMdata.append(M_store)

savemat('fMdata.mat', {'fMdata': fMdata})

