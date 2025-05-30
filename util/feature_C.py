import cv2
import pandas as pd
import numpy as np
from statistics import variance
from skimage.segmentation import slic
def feature_C(image,mask,im_id,df:pd.DataFrame):
    '''Takes image and dataframe to write results in\\
    Measures the blue veil of an image\\
    Outputs a modified dataframe with a new column 'C - blueveil'
    '''
    
    def blue_veil(image):
        r, g, b = cv2.split(image) ## Split into RGB
        mask = (b > 60) & ((r - 46 < g) & (g < r + 15)) ## Create a mask based on the condition

        return cv2.countNonZero(mask.astype(np.uint8)) ## Count non-zero (True) values in the mask
    
    def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    
        slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
        return slic_segments
    
    def get_rgb_means(image,slic_segments):
        
        max_segment_id = np.unique(slic_segments)[-1]

        rgb_means = []
        for i in range(1, max_segment_id + 1):

            segment = image.copy()
            segment[slic_segments != i] = -1

            rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
            rgb_means.append(rgb_mean) 
        return rgb_means
    
    def rgb_var(image, slic_segments):
    
        if len(np.unique(slic_segments)) == 2:  # If there are only two segments, return 0 for all color variances.
            return 0, 0, 0 # Return 0 for all color variances.

        rgb_means = get_rgb_means(image, slic_segments) # Get the RGB means of the image.
        n = len(rgb_means)  # Get the number of RGB means.

        red = [] # Initialize an empty list to store the red values.
        green = [] # Initialize an empty list to store the green values.
        blue = [] # Initialize an empty list to store the blue values.
        for rgb_mean in rgb_means: # Iterate through the RGB means.
            red.append(rgb_mean[0]) # Append the red value to the list.
            green.append(rgb_mean[1]) # Append the green value to the list.
            blue.append(rgb_mean[2]) # Append the blue value to the list.

        red_var = variance(red, sum(red)/n) # Calculate the variance of the red values.
        green_var = variance(green, sum(green)/n) # Calculate the variance of the green values.
        blue_var = variance(blue, sum(blue)/n) # Calculate the variance of the blue values.

        return red_var, green_var, blue_var # Return the variances of the red, green, and blue values.
    
    red,green,blue=rgb_var(image,slic_segmentation(image, mask, n_segments = 50, compactness = 0.1))
    #df.loc[im_id,'C - blueveil']=blue_veil(image)
    df.loc[im_id,'C - red variance']=red
    df.loc[im_id,'C - green variance']=green
    df.loc[im_id,'C - blue variance']=blue
    return df