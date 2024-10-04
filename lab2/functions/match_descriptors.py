import numpy as np
import pdb

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the second image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    # np.expand_dims is used to add singleton dimensions so that broadcasting can be applied
    # --> Avoid for loops

    diff = desc1[:, np.newaxis, :] - desc2
    distances = np.sum(diff**2, axis=-1)
    return distances


def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        min_idx = np.argmin(distances, axis=1)
        #pdb.set_trace()
        matches = np.vstack((np.arange(0, q1,1), min_idx)).T
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        min_idx1 = np.argmin(distances, axis=1)
        min_idx2 = np.argmin(distances[:,min_idx1], axis=0)
        valid_pairs = min_idx2==np.arange(q1) # use mask
        matches = np.vstack((np.arange(q1)[valid_pairs], min_idx1[valid_pairs])).T

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        # Find the indices of the two nearest neighbors
        sorted_indices = np.argpartition(distances, 2, axis=1)[:, :2]

        # Extract the indices of the nearest and second nearest neighbors
        first_nearest = sorted_indices[:, 0]
        second_nearest = sorted_indices[:, 1]

        # Filter out matches based on the ratio threshold
        valid_matches = []
        for i in range(q1):
            if distances[i][first_nearest[i]]<ratio_thresh*distances[i][second_nearest[i]]:
                valid_matches.append((i, first_nearest[i]))
        matches = np.array(valid_matches)
    else:
        raise NotImplementedError
    return matches

