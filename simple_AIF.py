import numpy as np
import matplotlib.pyplot as plt

def real_robot_dynamics(q1, q2, dq1, dq2, tau1, tau2):
    
    # Mechanical parameters
    I1z = 1.0209
    I2z = 1.0218
    m1 = 5.1368
    m2 = 5.2418
    L1 = 1.0446
    L2 = 1.0111
    Lg1 = 0.5451
    Lg2 = 0.5075
    g = 9.81
    D = np.array([[101.519, 0], [0, 103.8566]])

    # Variables definition
    tau = np.array([tau1, tau2])
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    dq = np.array([dq1, dq2])

    # Kinematic functions
    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s2 = np.sin(q2)
    c2 = np.cos(q2)

    # Elements of the Inertia Matrix M
    M11 = I1z + I2z + Lg1**2 * m1 + m2 * (L1**2 + Lg2**2 + 2 * L1 * Lg2 * c2)
    M12 = I2z + m2 * (Lg2**2 + L1 * Lg2 * c2)
    M22 = I2z + Lg2**2 * m2
    M = np.array([[M11, M12], [M12, M22]])

    # Coriolis and centrifugal elements
    C11 = -(L1 * dq2 * s2 * (Lg2 * m2))
    C12 = -(L1 * (dq2 + dq1) * s2 * (Lg2 * m2))
    C21 = m2 * L1 * Lg2 * s2 * dq1
    C22 = 0
    C = np.array([[C11, C12], [C21, C22]])

    # Gravity
    g1 = m1 * Lg1 * c1 + m2 * (Lg2 * c12 + L1 * c1)
    g2 = m2 * Lg2 * c12
    G = g * np.array([g1, g2])

    # Computation of acceleration
    ddq = np.linalg.inv(M).dot(tau - C.dot(dq) - D.dot(dq) - G)
    
    return ddq
    
    

# Set-up variables for simulation
t = 6              # [s] Simulation time
h = 0.001          # [s] Integration step
actionsTime = 1    # [s]
# Planar 2DOF robot parameters
a1 = 1             # [m] Length link 1
a2 = 1             # [m] Length link 2

# Initialize generative process (so the sensors' output)
q = np.zeros((2, int(t/h)))   # [rad]
dq = np.zeros((2, int(t/h)))  # [rad/s]
ddq = np.zeros((2, int(t/h))) # [rad/s^2]

# Initial state of the robot
q[:, 0] = [-np.pi/2, 0]

# Prior belief about the states of the robot arm, desired position
mu_d = np.array([-0.2, 0.3])

# Tuning parameters
P_mu0 = np.eye(2)
P_mu1 = np.eye(2)
P_y0 = np.eye(2)
P_y1 = np.eye(2)
k_mu = 20
k_a = 500

# Initialize vectors
u = np.zeros((2, int(t/h)))
u[:, 0] = [0, 0]

mu = np.zeros((2, int(t/h)))
mu_p = np.zeros((2, int(t/h)))
mu_pp = np.zeros((2, int(t/h)))

mu[:, 0] = q[:, 0] + [+0.6, +0.2]
mu_p[:, 0] = [0, 0]
mu_pp[:, 0] = [0, 0]

F = np.zeros(int(t/h) - 1)

y_q = np.zeros((2, int(t/h)))
y_dq = np.zeros((2, int(t/h)))




# Active Inference loop
for i in range(int(t/h) - 1):
    
     # Simulate noisy sensory input from encoders and tachometers
    z = np.random.normal(0, 0.001, size=q.shape[0])
    z_prime = np.random.normal(0, 0.001, size=q.shape[0])
    y_q[:, i] = q[:, i] + z
    y_dq[:, i] = dq[:, i] + z_prime

    # Compute free-energy in generalised coordinates
    F[i] = 0.5 * (y_q[:, i] - mu[:, i]).T @ P_y0 @ (y_q[:, i] - mu[:, i]) + \
           0.5 * (y_dq[:, i] - mu_p[:, i]).T @ P_y1 @ (y_dq[:, i] - mu_p[:, i]) + \
           0.5 * (mu_p[:, i] + mu[:, i] - mu_d).T @ P_mu0 @ (mu_p[:, i] + mu[:, i] - mu_d) + \
           0.5 * (mu_pp[:, i] + mu_p[:, i]).T @ P_mu1 @ (mu_pp[:, i] + mu_p[:, i])

    # Beliefs update
    mu_dot = mu_p[:, i] - k_mu * (-P_y0 @ (y_q[:, i] - mu[:, i]) + P_mu0 @ (mu_p[:, i] + mu[:, i] - mu_d))
    mu_dot_p = mu_pp[:, i] - k_mu * (-P_y1 @ (y_dq[:, i] - mu_p[:, i]) + P_mu0 @ (mu_p[:, i] + mu[:, i] - mu_d) + P_mu1 @ (mu_pp[:, i] + mu_p[:, i]))
    mu_dot_pp = -k_mu * P_mu1 @ (mu_pp[:, i] + mu_p[:, i])

    # State estimation
    mu[:, i+1] = mu[:, i] + h * mu_dot
    mu_p[:, i+1] = mu_p[:, i] + h * mu_dot_p
    mu_pp[:, i+1] = mu_pp[:, i] + h * mu_dot_pp

    # Control actions
    if i > actionsTime/h:
        # Active inference
        u[:, i+1] = u[:, i] - h * k_a * (P_y1 @ (y_dq[:, i] - mu_p[:, i]) + P_y0 @ (y_q[:, i] - mu[:, i]))
    else:
        u[:, i+1] = [0, 0]

    # Update sensory input according to the actions taken
    ddq[:, i] = real_robot_dynamics(q[0, i], q[1, i], dq[0, i], dq[1, i], u[0, i], u[1, i])
    dq[:, i+1] = dq[:, i] + h * ddq[:, i]
    q[:, i+1] = q[:, i] + h * dq[:, i]
    
    
    

# Graphics
time_vec = np.arange(0, t, h)
fontSize = 20
lineWidth = 4

plt.rcParams['legend.fontsize'] = fontSize
plt.rcParams['axes.labelsize'] = fontSize
plt.rcParams['xtick.labelsize'] = fontSize
plt.rcParams['ytick.labelsize'] = fontSize

plt.figure(figsize=(16, 12))

ind = list(range(0, len(time_vec), 400))  # 適切なステップでマーカーを表示するためのインデックスを生成

# State estimation for q1
plt.subplot(2, 2, 1)
plt.plot(time_vec, mu[0, :], '-k', linewidth=2)
plt.ylim([q[0, 0] - 0.06, mu_d[0] + 0.12])
plt.plot(time_vec, q[0, :], ':k', linewidth=2)
plt.plot([actionsTime, actionsTime], plt.ylim(), '-.k', linewidth=2)
plt.plot(time_vec, mu_d[0] * np.ones(time_vec.shape), '--k', marker='o', markersize=9, linewidth=2, markevery=ind)
plt.grid(True)
plt.grid(which='minor', linestyle='--')
plt.legend(['$\mu_1$', '$q_1$', '$t_a$', '$\mu_{d_1}$'], loc='best')
plt.title('State estimation for $q_1$')
plt.xlabel('Time $[s$]')
plt.ylabel('Internal states $\mu\ [rad]$')

# State estimation for q2
plt.subplot(2, 2, 2)
plt.plot(time_vec, mu[1, :], '-k', linewidth=2)
plt.ylim([q[1, 0] - 0.06, mu_d[1] + 0.06])
plt.plot(time_vec, q[1, :], ':k', linewidth=2)
plt.plot([actionsTime, actionsTime], plt.ylim(), '-.k', linewidth=2)
plt.plot(time_vec, mu_d[1] * np.ones(time_vec.shape), '--k', marker='o', markersize=9, linewidth=2, markevery=ind)
plt.grid(True)
plt.grid(which='minor', linestyle='--')
plt.legend(['$\mu_2$', '$q_2$', '$t_a$', '$\mu_{d_2}$'], loc='best')
plt.title('State estimation for $q_2$')
plt.xlabel('Time $[s$]')
plt.ylabel('Internal states $\mu\ [rad]$')

# Control actions
plt.subplot(2, 2, 3)
plt.plot(time_vec, u[0, :], '-k', linewidth=2)
plt.ylim([np.min(u), np.max(u) + 10])
plt.plot(time_vec, u[1, :], ':k', linewidth=2)
plt.plot([actionsTime, actionsTime], plt.ylim(), '-.k', linewidth=2)
plt.grid(True)
plt.grid(which='minor', linestyle='--')
plt.legend(['$u_1$', '$u_2$', '$t_a$'], loc='best')
plt.title('Control actions')
plt.xlabel('Time $[s$]')
plt.ylabel('Torques $u\ [rad]$')

# Free-energy
plt.subplot(2, 2, 4)
plt.plot(time_vec[:-1], F, '-k', linewidth=2)
plt.ylim([np.min(F), np.max(F) + 0.06])
plt.plot([actionsTime, actionsTime], plt.ylim(), '-.k', linewidth=1.5)
plt.grid(True)
plt.grid(which='minor', linestyle='--')
plt.legend(['$\mathcal{F}$', '$t_a$'], loc='best')
plt.title('Free-energy')
plt.xlabel('Time $[s$]')
plt.ylabel('Free-energy $[-]$')

plt.tight_layout()
plt.show()



plt.tight_layout()
plt.show()