import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def n_body_equations(w, t, masses):
    n = len(masses)
    positions = w[:3 * n].reshape(n, 3)
    velocities = w[3 * n:].reshape(n, 3)

    dvdt = np.zeros((n, 3))
    G = 6.67430e-11

    for i in range(n):
        for j in range(n):
            if i != j:
                relative_position = positions[j] - positions[i]
                distance = np.linalg.norm(relative_position)
                dvdt[i] += G * masses[j] * relative_position / distance ** 3

    dwdt = np.hstack((velocities.flatten(), dvdt.flatten()))
    return dwdt

def simulate_n_body_problem(initial_positions, initial_velocities, masses, t):
    n = len(masses)
    w0 = np.hstack((initial_positions.flatten(), initial_velocities.flatten()))

    solution = odeint(n_body_equations, w0, t, args=(masses,))

    positions = solution[:, :3 * n].reshape(-1, n, 3)
    velocities = solution[:, 3 * n:].reshape(-1, n, 3)

    return positions, velocities

# Example: Earth, Moon, and Sun system
masses = [5.972e24, 7.342e22, 1.989e30]  # kg
initial_positions = np.array([
    [-147095000000, 0, 0],  # Earth
    [-147095000000 + 384400000, 0, 0],  # Moon
    [0, 0, 0],  # Sun
])  # m
initial_velocities = np.array([
    [0, -30000, 0],  # Earth
    [0, -30000 + 1022, 0],  # Moon
    [0, 0, 0],  # Sun
])  # m/s

def update(frame, positions, scatters):
    for i, scatter in enumerate(scatters):
        scatter.set_offsets(positions[frame, i, :2])
    return scatters

# Example: Earth, Moon, and Sun system (same as before)
t = np.linspace(0, 365 * 24 * 60 * 60, 1000)

positions, velocities = simulate_n_body_problem(initial_positions, initial_velocities, masses, t)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-1.5e11, 1.5e11)
ax.set_ylim(-1.5e11, 1.5e11)
labels = ["Earth", "Moon", "Sun"]
colors = ["blue", "gray", "yellow"]
scatters = [ax.scatter(*positions[0, i, :2], color=colors[i], label=labels[i]) for i in range(3)]
ax.legend()

ani = FuncAnimation(fig, update, frames=len(t), interval=30, fargs=(positions, scatters), blit=True)
plt.show()