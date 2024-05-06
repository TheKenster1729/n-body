import numpy as np
import scipy.integrate as integrate

def three_body_equations(r, t):
    """
    This function returns the derivatives of the positions and velocities of the three bodies.

    Parameters
    ----------
    r : array_like
        The current positions and velocities of the three bodies.
    t : float
        The current time.

    Returns
    -------
    dr : array_like
        The derivatives of the positions and velocities of the three bodies.
    """

    # Extract the positions and velocities of the three bodies.
    r1, v1, r2, v2, r3, v3 = r

    # Calculate the forces on each body.
    F1 = -G * m1 * (r2 - r1) / np.linalg.norm(r2 - r1)**3
    F2 = -G * m2 * (r1 - r2) / np.linalg.norm(r1 - r2)**3
    F3 = -G * m3 * (r1 - r3) / np.linalg.norm(r1 - r3)**3

    # Calculate the derivatives of the positions and velocities.
    dr1 = v1
    dv1 = F1 / m1
    dr2 = v2
    dv2 = F2 / m2
    dr3 = v3
    dv3 = F3 / m3

    # Return the derivatives of the positions and velocities.
    return np.array([dr1, dv1, dr2, dv2, dr3, dv3])

# Define the universal gravitational constant.
G = 6.67408e-11

# Define the masses of the three bodies.
m1 = 1.0
m2 = 0.5
m3 = 0.25

# Define the initial positions and velocities of the three bodies.
r10 = np.array([1.0, 0.0, 0.0])
v10 = np.array([0.0, 1.0, 0.0])
r20 = np.array([0.0, 1.0, 0.0])
v20 = np.array([0.0, 0.0, 1.0])
r30 = np.array([0.5, 0.0, 0.0])
v30 = np.array([0.0, 0.5, 0.0])

# Define the time steps.
t0 = 0.0
t_end = 10.0
dt = 0.01
init_condition = np.hstack((r10, v10, r20, v20, r30, v30))

# Integrate the equations of motion.
sol = integrate.odeint(three_body_equations,
                         init_condition,
                         np.arange(t0, t_end, dt))

# Plot the trajectories of the three bodies.
import matplotlib.pyplot as plt

plt.figure()
plt.plot(sol[:, 0], sol[:, 1], label="Body 1")
plt.plot(sol[:, 2], sol[:, 3], label="Body 2")
plt.plot(sol[:, 4], sol[:, 5], label="Body 3")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()