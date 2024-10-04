import numpy as np
import cv2

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    
    # Get the region of interest of current frame
    frame = np.asarray(frame)
    roi = frame[ymin:ymax, xmin:xmax, :]

    # Calculate the histogram for each color channel
    hist = cv2.calcHist([roi], [0, 1, 2], None, [hist_bin, hist_bin, hist_bin], [0, 256, 0, 256, 0, 256])

    # Normalize histograms
    hist = hist/(np.sum(hist) + 1e-9) # avoid division by zero
    return hist