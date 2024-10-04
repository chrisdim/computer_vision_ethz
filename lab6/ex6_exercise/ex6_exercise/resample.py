import numpy as np

def resample(particles, particles_w):
    # Get random samples based on the weights particles_w from Equation 6
    idxs = np.random.choice(np.arange(particles.shape[0]), size=particles.shape[0], p=particles_w)

    # Update chosen particles and their weights
    new_particles = particles[idxs, :]
    new_particles_w = particles_w[idxs]/np.sum(particles_w[idxs])
    return new_particles, new_particles_w