#
# downstalce pr to precip
import os
import tensorflow as tf
from Network import Generator
from tensorflow.keras.optimizers import Adam
import numpy as np
# from Custom_loss import custom_loss
from Custom_loss_original import MaskedWeightedMAEPrecip
from numpy.random import randint

# Low-res shape: (time, lat, lon, variables)
# High-res shape: (time, lat, lon, 1) - only temperature (pr)

# Input shape for low-resolution data (13x11 with multiple variables)
image_shape_lr = (13, 11, 6)  # Changed from (11, 13, 6) to match actual data shape

# Output shape for high-resolution data 
image_shape_hr = (156, 132, 1)

downscale_factor = 12 

# Function to select a batch of random samples
def generate_batch_samples(data_gcm, data_obs, n_samples):
    # Choose random instances
    ix = randint(0, min(data_gcm.shape[0], data_obs.shape[0]), n_samples)
    # Retrieve selected images
    gcm = data_gcm[ix]
    obs = data_obs[ix]
    return gcm, obs

def save_models(step, model):
    # Save model in './save_model' directory
    directory = os.path.expanduser('./save_model')
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    # Save the model to the directory
    filename = 'generator_%03d.h5' % (step)
    model.save(os.path.join(directory, filename))
    print(f"Model saved as {os.path.join(directory, filename)}")


def train_step(data_gcm, data_obs):
    # data_obs comes in normalized–log1p space, shape [B,H,W,1]
    with tf.GradientTape() as tape:
        hr_fake   = generator(data_gcm, training=True)
        loss_value = loss_fn(data_obs, hr_fake)
    grads = tape.gradient(loss_value, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return loss_value

def train(train_gcm, train_obs, epochs, batch_size):
    n_epochs, n_batch = epochs, batch_size
    bat_per_epo = int(len(train_gcm) / n_batch)
    n_steps = bat_per_epo * n_epochs
    print(f"Starting training for {n_epochs} epochs with {n_batch} batches per epoch ({n_steps} total steps).")
    
    # Save the initial model (Epoch 0)
    print("Saving initial model (Epoch 0)...")
    save_models(0, generator)
    
    # Create loss log file
    with open('losses.txt', 'w') as loss_file:
        loss_file.write("Training Loss Log\n")
    
    for i in range(n_steps):
        print(f"\nStarting iteration {i+1}/{n_steps}")
        batch_gcm, batch_obs = generate_batch_samples(train_gcm, train_obs, batch_size)
        loss_value = train_step(batch_gcm, batch_obs)
        # Log the loss value
        print('Iteration>%d, loss=%.6f' % (i+1, loss_value))
        with open('losses.txt', 'a') as loss_file:
            loss_file.write('Iteration>%d, loss=%.6f\n' % (i+1, loss_value))
        # Save the model at the end of each epoch
        if (i+1) % bat_per_epo == 0:
            epoch_num = (i + 1) // bat_per_epo
            try:
                print(f"Saving model at Epoch {epoch_num}")
                save_models(epoch_num, generator)
            except Exception as e:
                print(f"Error saving model at Epoch {epoch_num}: {e}")
                break  # Break the loop if saving fails
    print("Training completed.")

# Load the preprocessed data
print("Loading preprocessed data...")

# Load mean and std for  (pr)
mean_pr = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/ERA5_mean_train.npy')
std_pr = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/ERA5_std_train.npy')
land_mask = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/land_mask.npy')
if land_mask.ndim == 3:
    # assume every timeslice is identical, so just grab the first
    land_mask = land_mask[0, ...]
print("mean_pr shape =", mean_pr.shape)
print("std_pr shape =", std_pr.shape)
loss_fn   = MaskedWeightedMAEPrecip(mean_pr, std_pr, land_mask, scale=36.6, w_min=0.1, w_max=2.0) # Define the input and output shapes based on preprocessed data

# Load low-resolution data for training (all variables)
predictors = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/predictors_train_mean_std_separate.npy')
print(f"Original predictors shape: {predictors.shape}")

# Load high-resolution data for training (pr only)
obs = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/obs_train_mean_std_single.npy')
print(f"Original obs shape: {obs.shape}")
print("mask:",       land_mask.shape)



# Make sure obs has the correct shape (should be [samples, 132, 169, 1])
if len(obs.shape) == 3:
    # Add channel dimension if missing
    obs = obs[:, :, :, np.newaxis]
elif obs.shape[3] > 1:
    # If multiple channels exist, keep only pr (first channel)
    obs = obs[:, :, :, 0:1]

print(f"Final predictors shape: {predictors.shape}")
print(f"Final obs shape: {obs.shape}")

# Update image_shape_lr based on actual data
image_shape_lr = predictors.shape[1:4]
print(f"Actual input shape: {image_shape_lr}")
# Instantiate the generator model with the adjusted input shape
generator = Generator(image_shape_lr).generator()
generator_optimizer = tf.keras.optimizers.Adam(1e-5)
# Start training
train(predictors, obs, epochs=160, batch_size=64)
