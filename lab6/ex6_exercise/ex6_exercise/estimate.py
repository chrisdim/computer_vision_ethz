import numpy as np

# Equation 8
def estimate(particles, particles_w):
    return np.transpose(np.matmul(np.transpose(particles), particles_w))