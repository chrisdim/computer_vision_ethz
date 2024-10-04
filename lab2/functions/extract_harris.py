import numpy as np
import cv2
from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
import pdb

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    # Define Sobel filters for gradient computation
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convolve the image with Sobel filters to compute gradients
    gradient_x = signal.convolve2d(img, sobel_x, mode='same', boundary='symm')
    gradient_y = signal.convolve2d(img, sobel_y, mode='same', boundary='symm')

    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)

    blurred_gradient_x = cv2.GaussianBlur(gradient_x, (0, 0), sigmaX=sigma,sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    blurred_gradient_y = cv2.GaussianBlur(gradient_y, (0, 0), sigmaX=sigma,sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here

    Jxx = cv2.GaussianBlur(blurred_gradient_x*blurred_gradient_x, (0, 0), sigmaX=sigma,sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    Jxy = cv2.GaussianBlur(blurred_gradient_x*blurred_gradient_y, (0, 0), sigmaX=sigma,sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    Jyy = cv2.GaussianBlur(blurred_gradient_y*blurred_gradient_y, (0, 0), sigmaX=sigma,sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)


    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    
    # Compute Eigenvalues
    lambda_plus  = 1/2 * (Jxx + Jyy + np.sqrt( np.square(Jxx-Jyy) + 4*np.square(Jxy) ))
    lambda_minus = 1/2 * (Jxx + Jyy - np.sqrt( np.square(Jxx-Jyy) + 4*np.square(Jxy) ))

    # Compute function R
    determinant = lambda_plus*lambda_minus
    #determinant = Jxx*Jyy-Jxy**2
    trace = lambda_minus + lambda_plus
    #trace = Jxx+Jyy
    R = determinant - k*np.square(trace)

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    
    # Condition 1
    Cond1 = R==ndimage.maximum_filter(R, size=(3,3), mode='constant')
    
    # Condition 2
    Cond2 = np.where(R > thresh, 1, 0)
    
    edge_corners = np.where(Cond1 & Cond2, 1, 0) # both conditions satisfied

    # Convert to the correct output format
    # initialize with dummy first row
    corners = np.array([[-1,-1]])
    
    for iy, ix in np.ndindex(edge_corners.shape):
        if edge_corners[iy][ix]:
            temp = np.array( [[ix, iy]] )
            corners = np.vstack((corners,temp))

    return corners[1:], R # discard first dummy row