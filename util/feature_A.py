import pandas as pd
import numpy as np
from math import floor, ceil
from skimage.transform import rotate
from skimage import color

## All used for mean/worst asymmetry V
def find_midpoint_v1(image):
    
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid
            
def asymmetry(mask):
    

    row_mid, col_mid = find_midpoint_v1(mask) # Get the midpoints of the mask along the y-axis and x-axis.

    upper_half = mask[:ceil(row_mid), :] # Get the upper half of the mask.
    lower_half = mask[floor(row_mid):, :] # Get the lower half of the mask.
    left_half = mask[:, :ceil(col_mid)] # Get the left half of the mask.
    right_half = mask[:, floor(col_mid):] # Get the right half of the mask.

    flipped_lower = np.flip(lower_half, axis=0) # Flip the lower half of the mask along the y-axis.
    flipped_right = np.flip(right_half, axis=1) # Flip the right half of the mask along the x-axis.

    hori_xor_area = np.logical_xor(upper_half, flipped_lower) # Calculate the horizontal asymmetry area.
    vert_xor_area = np.logical_xor(left_half, flipped_right) # Calculate the vertical asymmetry area.

    total_pxls = np.sum(mask) # Calculate the total number of pixels in the mask.
    hori_asymmetry_pxls = np.sum(hori_xor_area) # Calculate the number of pixels in the horizontal asymmetry area.
    vert_asymmetry_pxls = np.sum(vert_xor_area) # Calculate the number of pixels in the vertical asymmetry area.

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2) # Calculate the asymmetry score.

    return round(asymmetry_score, 4) # Return the asymmetry score rounded to 4 decimal places.

def cut_mask(mask):
    col_sums = np.sum(mask, axis=0) # Sum the mask along the y-axis.
    row_sums = np.sum(mask, axis=1) # Sum the mask along the x-axis.
    active_cols = [] # Initialize an empty list to store the active columns.
    for index, col_sum in enumerate(col_sums): # Iterate through the summed mask.
        if col_sum != 0: # If the column sum is not 0, append the index to the list.
            active_cols.append(index) # Append the index to the list.

    active_rows = [] # Initialize an empty list to store the active rows.
    for index, row_sum in enumerate(row_sums): # Iterate through the summed mask.
        if row_sum != 0: # If the row sum is not 0, append the index to the list.
            active_rows.append(index) # Append the index to the list.

    col_min = active_cols[0] # Get the minimum index of the active columns.
    col_max = active_cols[-1] # Get the maximum index of the active columns.
    row_min = active_rows[0] # Get the minimum index of the active rows.
    row_max = active_rows[-1] # Get the maximum index of the active rows.

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1] # Crop the mask to the limits of the active rows and columns.

    return cut_mask_ # Return the cropped mask.

def rotation_asymmetry(mask, n: int):

    asymmetry_scores = {} # Initialize an empty dictionary to store the asymmetry scores.

    for i in range(n): # Iterate through the number of rotations.

        degrees = 90 * i / n # Calculate the degrees of rotation.

        rotated_mask = rotate(mask, degrees) # Rotate the mask by the calculated degrees.
        cutted_mask = cut_mask(rotated_mask) # Crop the rotated mask to the limits of the active rows and columns.

        asymmetry_scores[degrees] = asymmetry(cutted_mask) # Calculate the asymmetry score of the cropped rotated mask.

    return asymmetry_scores # Return the dictionary of asymmetry scores.

def mean_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score

def worst_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations) # Calculate the asymmetry scores of the rotated masks.
    worst_score = max(asymmetry_scores.values()) # Calculate the maximum of the asymmetry scores.

    return worst_score # Return the worst asymmetry score. 
## ----------------------------------------------------

## All used for get_asymmetry V
def find_midpoint_v4(mask):
        summed = np.sum(mask, axis=0)
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i
            
def crop(mask):
        mid = find_midpoint_v4(mask)
        y_nonzero, x_nonzero = np.nonzero(mask)
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        x_dist = max(np.abs(x_lims - mid))
        x_lims = [mid - x_dist, mid+x_dist]
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]      

def get_asymmetry(mask):
    # mask = color.rgb2gray(mask)
    scores = []
    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        mask = rotate(mask, 30)
    return sum(scores) / len(scores)

def feature_A(image,mask,im_id,df:pd.DataFrame):
    '''Takes mask of an image and dataframe to write results in\\
    Calculates asymmetry of the mask \\
    Outputs a modified dataframe with new column 'A - asymmetry'
    '''
    ## Choose one of these, comment out others
    # df.loc[im_id,'A - asymmetry']=asymmetry(mask)
    # df.loc[im_id,'A - get_asymmetry']=get_asymmetry(mask)
    df.loc[im_id,'A - worst asymmetry']=worst_asymmetry(mask)
    # df.loc[im_id,'A - mean asymmetry']=mean_asymmetry(mask)
    return df