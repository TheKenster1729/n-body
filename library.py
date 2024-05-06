import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KDTree
import pandas as pd
import plotly.express as px

class Mass:
    def __init__(self, name, mass, init_position, init_velocity) -> None:
        self.name = name
        self.mass = mass
        self.init_position = init_position
        self.init_velocity = init_velocity

class NBodySimulation:
    def __init__(self, t, masses) -> None:
        if type(masses) == int: # auto-generate masses
            self.masses = []
            for i in range(masses):
                mass = 1
                init_position = 10*np.random.random(size = 3)
                init_velocity = np.zeros(3)
                new_mass = Mass('M{}'.format(i + 1), mass, init_position, init_velocity)
                self.masses.append(new_mass)
        elif type(masses) == list:
            if len(masses) > 1:
                self.masses = masses
            else:
                raise ValueError('Invalid number of masses')
        else:
            raise TypeError('Invalid type (accepted types are list or int)')
        self.t = t # t, by default, is one year

    def n_body_equations(self, w, t):
        n = len(self.masses)
        positions = w[:3 * n].reshape(n, 3)
        velocities = w[3 * n:].reshape(n, 3)

        dvdt = np.zeros((n, 3))
        G = 6.67e-11

        for i in range(n):
            for j in range(n):
                if i != j:
                    relative_position = positions[j] - positions[i]
                    distance = np.linalg.norm(relative_position)
                    dvdt[i] += G * self.masses[j].mass * relative_position / distance ** 3

        dwdt = np.hstack((velocities.flatten(), dvdt.flatten()))
        return dwdt

    def simulate_n_body_problem(self):
        n = len(self.masses)
        initial_positions = np.array([
            mass.init_position for mass in self.masses
        ])
        initial_velocities = np.array([
            mass.init_velocity for mass in self.masses
        ])

        w0 = np.hstack((initial_positions.flatten(), initial_velocities.flatten()))

        solution, full_output = odeint(self.n_body_equations, w0, self.t, full_output = True)

        positions = solution[:, :3 * n].reshape(-1, n, 3)
        velocities = solution[:, 3 * n:].reshape(-1, n, 3)

        # could add velocities later
        n_body_df = pd.DataFrame(columns = ["time", "name", "mass", "x_pos", "y_pos", "z_pos"])
        for i, mass in enumerate(self.masses):
            temp_df = pd.DataFrame()
            temp_df["time"] = self.t
            temp_df["name"] = [mass.name for i in range(len(self.t))]
            temp_df["mass"] = [mass.mass for k in range(len(self.t))]
            temp_df["x_pos"] = positions[:, i, 0]
            temp_df["y_pos"] = positions[:, i, 1]
            temp_df["z_pos"] = positions[:, i, 2]

            n_body_df = pd.concat([n_body_df, temp_df])

        return n_body_df

# pass in position vectors as a dataframe with columns "time", "label", "x_pos", "y_pos", "z_pos"
# bounds can be inferred from position vectors
# return plotly figure
class Animation:
    def __init__(self, position_df):
        self.position_df = position_df
        if position_df["z_pos"].std() == 0:
            self.animate = "2d"
        else:
            self.animate = "3d"
        # self.position_df["size"] = 1

    def create_3d_animation(self):
        pass

    def create_2d_animation(self):
        min_x = 1.1*self.position_df["x_pos"].min()
        max_x = 1.1*self.position_df["x_pos"].max()
        min_y = 1.1*self.position_df["y_pos"].min()
        max_y = 1.1*self.position_df["y_pos"].max()

        # need to scale the size of the points on the scatterplot, but carefully, because masses are orders of magnitude
        # in difference
        # max_mass = self.position_df["mass"].max()
        # min_mass = self.position_df["mass"].min()
        # scaler = np.log(max_mass/min_mass)

        # for colors - need to find better implementation
        color_options = ["#F7B267", "#2D3047", "#E84855", "#748CAB", "#D9F7FA"]
        colors_intermediate = [color_options[i] for i in range(self.position_df["name"].nunique())]
        colors = []
        colors += [colors_intermediate[i] for i in range(len(colors_intermediate))]*self.position_df["time"].nunique()

        fig = px.scatter(self.position_df, x = "x_pos", y= "y_pos", animation_frame = "time", animation_group = "name",
                hover_name = "name", color = colors, range_x = [min_x, max_x], range_y= [min_y, max_y])

        return fig

    def export_animation(self):
        if self.animate == "2d":
            fig = self.create_2d_animation()
        else:
            self.create_3d_animation()

        fig.show()

class Display(NBodySimulation):
    def __init__(self, colors = ["blue", "gray", "yellow"], t = np.linspace(0, 365 * 24 * 60 * 60, 1000), masses = 3, preset = None, dims = 3) -> None:
        super().__init__(t, masses)
        self.fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter'}]])
        self.fig.update_layout(scene=dict(xaxis_title='x (m)', yaxis_title='y (m)'))
        self.colors = colors
        if (dims == 2) or (dims == 3):
            self.dims = dims
        else:
            raise ValueError("Number of simulated dimensions must be equal to 2 or 3")
        # if preset == "":

    def trajectory_search(self):
        # run the algorithm
        pass

    def create_matplotlib_animation(self):
        positions, velocities = self.simulate_n_body_problem()
        def update(frame, positions, scatters):
            for i, scatter in enumerate(scatters):
                scatter.set_offsets(positions[frame, i, :2])
            return scatters

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.5e11, 1.5e11)
        ax.set_ylim(-1.5e11, 1.5e11)
        labels = [x.name for x in self.masses]
        colors = self.colors
        scatters = [ax.scatter(*positions[0, i, :2], color = colors[i], label = labels[i]) for i in range(len(self.masses))]
        ax.legend()

        ani = FuncAnimation(fig, update, frames = len(self.t), interval = 30, fargs=(positions, scatters), blit = True)
        plt.show()

    def create_plotly_animation_3d(self):
        # Create a subplot with 3D scatter plot

        labels = [x.name for x in self.masses]
        positions, velocities, full_output = self.simulate_n_body_problem()

        # Add scatter plots for each celestial body
        for i in range(len(self.masses)):
            self.fig.add_trace(
                go.Scatter3d(
                    x = [positions[0, i, 0]],
                    y = [positions[0, i, 1]],
                    z = [positions[0, i, 2]],
                    mode = "markers",
                    marker = dict(size = 5),
                    name = labels[i]
                            )
            )

        # Define frames for the animation
        frames = [go.Frame(data = [go.Scatter3d(x = [positions[k, i, 0]], y = [positions[k, i, 1]], z = [positions[k, i, 2]],
                                             mode = "markers",
                                             marker = dict(size = 5),
                                             name = labels[i]
                                             ) for i in range(len(self.masses))]) for k in range(len(self.t))]

        self.fig.frames = frames

        # Set up the animation settings
        animation_settings = dict(frame = dict(duration = 25, redraw = True), fromcurrent = True)

        self.fig.update_layout(updatemenus = [dict(type = "buttons", buttons = [dict(label = "Play",
                                                    method = "animate", args = [None, animation_settings])])],
                                                    scene = dict(
                                                        xaxis = dict(range = [-20, 20], nticks = 4, visible = False),
                                                        yaxis = dict(range = [-20, 20], nticks = 4, visible = False),
                                                        zaxis = dict(range = [-20, 20], nticks = 4, visible = False)
                                                    ))

        return self.fig

    def create_plotly_animation_2d(self):
        # Create a subplot with 3D scatter plot

        labels = [x.name for x in self.masses]
        positions, velocities, full_output = self.simulate_n_body_problem()

        # Add scatter plots for each celestial body
        for i in range(len(self.masses)):
            self.fig.add_trace(
                go.Scatter(
                    x = [positions[0, i, 0]],
                    y = [positions[0, i, 1]],
                    mode = "markers",
                    marker = dict(size = 15),
                    name = labels[i]
                            )
            )

        # Define frames for the animation
        frames = [go.Frame(data = [go.Scatter(x = [positions[k, i, 0]], y = [positions[k, i, 1]],
                                             mode = "markers",
                                             marker = dict(size = 15),
                                             name = labels[i]
                                             ) for i in range(len(self.masses))]) for k in range(len(self.t))]

        self.fig.frames = frames

        # Set up the animation settings
        animation_settings = dict(frame = dict(duration = 25, redraw = True), fromcurrent = True)

        self.fig.update_layout(updatemenus = [dict(type = "buttons", buttons = [dict(label = "Play",
                                                    method = "animate", args = [None, animation_settings])])],
                                                    scene = dict(
                                                        xaxis = dict(range = [-20, 20], nticks = 4, visible = False),
                                                        yaxis = dict(range = [-20, 20], nticks = 4, visible = False)
                                                    ))

        return self.fig

    def create_plotly_animation(self):
        if self.dims == 2:
            fig = self.create_plotly_animation_2d()
        else:
            fig = self.create_plotly_animation_3d()

        fig.show()

# conducts a search on a large parameter space to find bounded trajectories
class BoundedTrajectorySearch:
    def __init__(self, t = np.linspace(0, 365*24*60*60, 10000), num_masses = 2) -> None:
        self.t = t
        self.num_masses = num_masses
        self.successful_positions = []

    def run_simulation(self):
        # inputs: vector of initial conditions
        # returns: initial position if bounded and stable, None otherwise
        simulator = NBodySimulation(self.t, self.num_masses)
        masses = simulator.masses
        initial_positions = np.array([mass.init_position for mass in masses])

        positions, velocities, full_output = simulator.simulate_n_body_problem()

        # check boundedness with ball tree search
        tree = KDTree(initial_positions)
        count = tree.query_radius(positions[-1], 100, count_only = True)
        if np.sum(count)/len(masses) != len(masses):
            # system unbounded
            return None

        elif full_output['message'] == "Excess work done on this call (perhaps wrong Dfun type).":
            # unstable
            return None

        else:
            self.successful_positions.append(initial_positions)
            return True

    def vae_search(self, library):
        # inputs: library of successful initial conditions
        # outputs: guess for next initial condition
        # randomly choose between random guess and this method for next initial condition;
        # weight so this method is favored for large training data

        n = len(library)
        k = self.num_masses

        # Flatten the array to shape (n, k*3)
        flattened_array = library.reshape(n, -1)

        # Define VAE parameters
        input_dim = k * 3
        latent_dim = 32
        intermediate_dim = 64
        batch_size = 128
        epochs = 100

        # Encoder network
        inputs = Input(shape = (input_dim,))
        h = Dense(intermediate_dim, activation = 'relu')(inputs)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch_size = tf.shape(z_mean)[0]
            epsilon = tf.random.normal(shape=(batch_size, latent_dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling)([z_mean, z_log_var])

        # Decoder network
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        output = decoder_mean(h_decoded)

        # VAE model
        vae = Model(inputs, output)

        # Loss function
        reconstruction_loss = MeanSquaredError()(inputs, output)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        # Compile and train the VAE
        vae.add_loss(vae_loss)
        vae.compile(optimizer = Adam(learning_rate = 0.001))
        vae.fit(flattened_array, epochs=epochs, batch_size = batch_size)

        # Generate new matrix
        def generate_new_matrix(decoder_h, decoder_mean, latent_dim, k):
            random_latent = np.random.normal(size=(1, latent_dim))
            h_decoded = decoder_h(random_latent)
            generated_matrix = decoder_mean(h_decoded)
            return generated_matrix.numpy().reshape(k, 3)

        new_matrix = generate_new_matrix(decoder_h, decoder_mean, latent_dim, k)
        return new_matrix
        
    def main(self, sim_length):
        for i in range(sim_length):
            self.run_simulation()

        return self.successful_positions

sun_moon_earth_system = [Mass('Earth', 5.972e24, [-147095000000, 0, 0], [0, -30000, 0]),
                  Mass('Moon', 7.342e22, [-147095000000 + 384400000, 0, 0], [0, -30000 + 1022, 0]),
                  Mass('Sun', 1.989e30, [0, 0, 0], [0, 0, 0])]
generic_system = [Mass('M1', 5e28, [1.5e10, 1.5e9, 1e8], [0, -3000, 0]),
                  Mass('M2', 5e28, [-1.5e10, 0, 2e8], [0, 3000, 0]),
                  Mass('M3', 1e24, [-1.2e11, 0.3e11, 1e8], [500, -300, 0])]
chaotic_system = [Mass('M1', 1e7, [0, 1.6e3, 0], [0, 0, 0]),
                  Mass('M2', 1e7, [0, 0, 0], [0, 0, 0]),
                  Mass('M3', 1e7, [1.2e3, 0, 0], [0, 0, 0])]
parallelogram = [Mass('M1', 1e7, [-0.4e3, 0, 0], [0, 0, 0]),
                  Mass('M2', 1e7, [0, 0, 0], [0, 0, 0]),
                  Mass('M3', 1e7, [-0.2e3, 0.5e3, 0], [0, 0, 0]),
                  Mass('M4', 1e7, [0.4e3, 0.5e3, 0], [0, 0, 0])]
unstable_system = [Mass('M1', 1e6, [0, 1e3, 0], [0, 0, 0]),
                  Mass('M2', 1e6, [0, -1e3, 0], [0, 0, 0])]
two_bodies = [Mass('M1', 1, [0, 0, 0], [0, 0, 0]),
                  Mass('M2', 0.0000001, [1, 0, 0], [0, 6.5e-8, 0])]

# search = BoundedTrajectorySearch().run_simulation()
# search.run_simulation(np.array([[0, 0, 0], [0, 6, 0], [8, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
# print(search.successful_positions)
# successful_positions = BoundedTrajectorySearch(num_masses = 3).main(1000)
# print(successful_positions)

# m = [Mass('M1', 1, [7.74465805, 4.6389292 , 0.08169687], [0, 0, 0]),
#      Mass('M2', 1, [6.37323385, 7.30887032, 3.25009621], [0, 0, 0]),
#      Mass('M3', 1, [8.19265107, 7.82594365, 7.43824675], [0, 0, 0])]
# Display(masses = two_bodies, dims = 2).create_plotly_animation()

n_body = NBodySimulation(np.linspace(0, 100000, 10001), sun_moon_earth_system).simulate_n_body_problem()
Animation(n_body).export_animation()