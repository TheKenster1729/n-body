import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np

# Assuming you have an array of shape (n, k, 3)
n = 1000
k = 10
example_array = np.random.rand(n, k, 3)

# Flatten the array to shape (n, k*3)
flattened_array = example_array.reshape(n, -1)

# Define VAE parameters
input_dim = k * 3
latent_dim = 32
intermediate_dim = 64
batch_size = 128
epochs = 100

# Encoder network
inputs = Input(shape=(input_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
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
print(reconstruction_loss)
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
print(vae_loss)

# Compile and train the VAE
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(learning_rate=0.001))
print(vae.summary())
vae.fit(flattened_array, epochs=epochs, batch_size=batch_size)

# Generate new matrix
def generate_new_matrix(decoder_h, decoder_mean, latent_dim, k):
    random_latent = np.random.normal(size=(1, latent_dim))
    h_decoded = decoder_h(random_latent)
    generated_matrix = decoder_mean(h_decoded)
    return generated_matrix.numpy().reshape(k, 3)

new_matrix = generate_new_matrix(decoder_h, decoder_mean, latent_dim, k)
print(new_matrix)