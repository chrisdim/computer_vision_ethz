import numpy as np


def propagate(particles, frame_height, frame_width, params):
    # time step
    dt = 1

    # No Motion Just Noise
    if params["model"] == 0:
        A = np.array([[1, 0],
                      [0, 1]])
    # Constant Velocity Motion Model
    else:
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    sigma_pos = params["sigma_position"]
    sigma_vel = params["sigma_velocity"]

    # Define Gaussian Noise
    w = np.random.normal(loc=0, scale=sigma_pos, size=(2, particles.shape[0]))

    # If Constant Velocity Model, augment state vector by two to include velocities
    if params["model"] == 1:
        w_vel = np.random.normal(loc=0, scale=sigma_vel, size=(2, particles.shape[0]))
        w = np.vstack((w, w_vel))

    # Equation s'(t) = As(t-1) + w(t-1)
    new_particles = np.transpose(np.matmul(A, np.transpose(particles)) + w)

    # Check if center of particle lies inside the frame, otherwise move to borders
    new_particles[:, 0] = np.minimum(new_particles[:, 0], frame_width-1)
    new_particles[:, 1] = np.minimum(new_particles[:, 1], frame_height-1)
    new_particles[:, 0:2] = np.maximum(new_particles[:, 0:2], 0)

    return new_particles