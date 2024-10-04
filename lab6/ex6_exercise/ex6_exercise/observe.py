import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma):
    particles_w = np.zeros(particles.shape[0])
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Iterate over each particle
    for i, particle in enumerate(particles):
        
        # Compute the new color histogram of particle
        new_hist = color_histogram(
            min(max(0, round(particles[i, 0]-0.5*bbox_width)), frame_width-1),
            min(max(0, round(particles[i, 1]-0.5*bbox_height)), frame_height-1),
            min(max(0, round(particles[i, 0]+0.5*bbox_width)), frame_width-1),
            min(max(0, round(particles[i, 1]+0.5*bbox_height)), frame_height-1),
            frame, hist_bin)
        
        # Compute the x^2 distance between histograms
        chi2_dist = chi2_cost(new_hist, hist)

        # Weight using Equation 6
        particles_w[i] = (1./(np.sqrt(2.*np.pi)*sigma))*np.exp(-(chi2_dist**2)/(2.*(sigma**2)))

    return particles_w/np.sum(particles_w) # normalize weights