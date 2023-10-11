from gym import wrappers
from math import cos, sin, acos, asin, atan, pi
import numpy as np
import gym

env = gym.make('Kinova-v0')

# Initial state of the robot
obs = env.reset()

action = [0,0,0]
target = [0.65,0.8]

start = 1000
end = 1000

# Set-up variables for simulation
t = 6              # [s] Simulation time
h = 0.001          # [s] Integration step
actionsTime = 1    # [s]
# Planar 2DOF robot parameters
a1 = 1             # [m] Length link 1
a2 = 1             # [m] Length link 2

# Initialize generative process (so the sensors' output)
#ここはシミュレータを使うので必要ない(observationに相当)
#q = np.zeros((2, int(t/h)))   # [rad]
#dq = np.zeros((2, int(t/h)))  # [rad/s]
#ddq = np.zeros((2, int(t/h))) # [rad/s^2]
# Initial state of the robot
#q[:, 0] = [-np.pi/2, 0]


# Prior belief about the states of the robot arm, desired position
mu_d = np.array([-0.2, 0.3, 0])

# Tuning parameters
P_mu0 = np.eye(3)
P_mu1 = np.eye(3)
P_y0 = np.eye(3)
P_y1 = np.eye(3)
k_mu = 20
k_a = 500

# Initialize vectors
u = np.zeros((3, start + 10000 + end))
u[:, 0] = [0, 0, 0]

mu = np.zeros((3, start + 10000 + end))
mu_p = np.zeros((3, start + 10000 + end))
mu_pp = np.zeros((3, start + 10000 + end))

mu[:, 0] = obs[:, 0] + [+0.6, +0.2]
mu_p[:, 0] = [0, 0, 0]
mu_pp[:, 0] = [0, 0, 0]

F = np.zeros(start + 10000 + end)

y_q = np.zeros((3, start + 10000 + end))
y_dq = np.zeros((3, start + 10000 + end))



#Actiive inference loop
for i in range(start + 10000 + end):
    
    env.render()
    z = np.random.normal(0, 0.001, size=q.shape[0])
    z_prime = np.random.normal(0, 0.001, size=q.shape[0])
    
    action = u
    obs, reward, done, info = env.step(action)

    #センサ観測はobs
    y_q[:, i] = obs[:, i] + z
    y_dq[:, i] = obs[:, i] + z_prime

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
        
    if i < start:
        u[:, i+1] = [0, 0]
        

    elif start <= i < start +10000:
        # Active inference
        u[:, i+1] = u[:, i] - h * k_a * (P_y1 @ (y_dq[:, i] - mu_p[:, i]) + P_y0 @ (y_q[:, i] - mu[:, i]))
        

        
else:
    print("finished")