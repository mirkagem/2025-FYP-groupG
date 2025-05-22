import pandas as pd
import numpy as np
from math import floor, ceil
from skimage.transform import rotate
import cv2

# def find_midpoint_v1(image):
    
#     row_mid = image.shape[0] / 2
#     col_mid = image.shape[1] / 2
#     return row_mid, col_mid

# def asymmetry(mask):
    

#     row_mid, col_mid = find_midpoint_v1(mask)

#     upper_half = mask[:ceil(row_mid), :]
#     lower_half = mask[floor(row_mid):, :]
#     left_half = mask[:, :ceil(col_mid)]
#     right_half = mask[:, floor(col_mid):]

#     flipped_lower = np.flip(lower_half, axis=0)
#     flipped_right = np.flip(right_half, axis=1)

#     hori_xor_area = np.logical_xor(upper_half, flipped_lower)
#     vert_xor_area = np.logical_xor(left_half, flipped_right)

#     total_pxls = np.sum(mask)
#     hori_asymmetry_pxls = np.sum(hori_xor_area)
#     vert_asymmetry_pxls = np.sum(vert_xor_area)

#     asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

#     return round(asymmetry_score, 4)

def find_midpoint_v4(mask):
        summed = np.sum(mask, axis=0) # Sum the mask along the x-axis.
        half_sum = np.sum(summed) / 2 # Calculate the half sum of the mask.
        for i, n in enumerate(np.add.accumulate(summed)): # Iterate through the summed mask.
            if n > half_sum: # If the accumulated sum is greater than the half sum, return the index.
                return i # Return the index of the midpoint.

def crop(mask):
        mid = find_midpoint_v4(mask) # Find the midpoint of the mask.
        y_nonzero, x_nonzero = np.nonzero(mask) # Get the non-zero coordinates of the mask.
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)] # Get the limits of the y-coordinates.
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)]) # Get the limits of the x-coordinates.
        x_dist = max(np.abs(x_lims - mid)) # Calculate the distance from the midpoint to the limits of the x-coordinates.
        x_lims = [mid - x_dist, mid+x_dist] # Get the limits of the x-coordinates.
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]] # Crop the mask to the limits of the y-coordinates and x-coordinates.

def get_asymmetry(mask):
    # mask = color.rgb2gray(mask)
    scores = [] # Initialize an empty list to store scores.
    for _ in range(6):
        segment = crop(mask) # Crop the mask to a smaller region.
        (np.sum(segment)) # Calculate the sum of the cropped mask.
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment))) #Append the asymmetry score to the list. The Asymmetry score is calculated by taking the sum of the logical XOR between the segment and its flipped version, divided by the sum of the segment.
        mask = rotate(mask, 30) # Rotate the mask by 30 degrees.
    return sum(scores) / len(scores) # Calculate the average of the scores.

def feature_A(mask,im_id,df:pd.DataFrame):
    '''Takes mask of an image and dataframe to write results in\\
    Calculates asymmetry of the mask \\
    Outputs a modified dataframe with new column 'A - asymmetry'
    '''
    # df.loc[im_id,'A - asymmetry']=asymmetry(mask)
    df.loc[im_id,'A - asymmetry']=np.where(np.isnan(get_asymmetry(mask)),0,get_asymmetry(mask))
    return df