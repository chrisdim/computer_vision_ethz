import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    # Radius is infinite so examine all point pairs
    return np.sqrt(np.sum((X - x)**2, axis=1))

def gaussian(dist, bandwidth):
    # Compute Gaussian weights
    return np.exp(-(dist**2) / (2 * (bandwidth**2)))

def update_point(weight, X):
    """
    This function takes the weights and the array of points X, 
    and it updates the position of the point
    by computing the weighted sum of all points in X. 
    The updated position is then normalized by the total weight.

    """
    # Update point position based on Gaussian weights
    weighted_sum = np.sum(weight[:, np.newaxis] * X, axis=0)
    total_weight = np.sum(weight)
    return weighted_sum / total_weight

def meanshift_step(X, bandwidth=2.5):
    for i, x in enumerate(X):
        # Calculate distances from the current point to all other points
        dist = distance(x, X)

        # Compute Gaussian weights based on distances and bandwidth
        weight = gaussian(dist, bandwidth)

        # Update the point position based on Gaussian weights
        X[i] = update_point(weight, X)

    return X

def meanshift(X):
    for _ in range(20):
        X = meanshift_step(X)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = io.imread('./eth.jpg')
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)


"""

The bandwidth parameter in the mean shift algorithm controls the size of the spatial neighborhood used to compute the mean shift. It influences how far the algorithm looks for points to consider in the update process for each point.

A larger bandwidth means a larger spatial neighborhood, and points from a wider region will contribute to the update of the current point. This can lead to smoother and more gradual convergence, but it might also result in a slower algorithm.

Conversely, a smaller bandwidth restricts the spatial neighborhood, causing the algorithm to focus on a narrower region around each point. This can lead to faster convergence, but it may result in a less smooth output, potentially capturing more detailed structures in the data.

Choosing the appropriate bandwidth is often a crucial aspect of using the mean shift algorithm, and it depends on the characteristics of the data you are working with. It's common to experiment with different bandwidth values to find the one that produces the desired level of smoothing or detail in the output.
"""

"""

If the bandwidth is set too large, it may cause numerical instability or excessive memory usage in the mean shift algorithm. Here are a couple of reasons why this might happen and potential solutions:

Numerical Instability:

If the bandwidth is too large, the Gaussian weights calculated in the algorithm may become very small for some points, leading to numerical instability.
This can result in extremely small or zero weights, causing division by very small numbers or zero in the update step, leading to numerical instability.
Solution:

Try reducing the bandwidth to a smaller value. This can stabilize the computation of Gaussian weights and prevent numerical instability.
Memory Usage:

A large bandwidth increases the number of points considered in the spatial neighborhood for each point, leading to increased memory usage.
This can be problematic, especially for large datasets, as the algorithm needs to store distances and weights for all pairs of points.
Solution:

Use a smaller bandwidth to reduce the size of the spatial neighborhood. Alternatively, you can consider optimizing the algorithm or working with a subset of the data to manage memory constraints.
In general, it's a good practice to start with a small bandwidth and gradually increase it while monitoring the algorithm's behavior. This allows you to find a balance between capturing the desired level of detail and ensuring numerical stability and efficient memory usage.

"""