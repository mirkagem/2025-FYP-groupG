import cv2
import pandas as pd
import numpy as np
def feature_C(image,im_id,df:pd.DataFrame):
    '''Takes image and dataframe to write results in\\
    Measures the blue veil of an image\\
    Outputs a modified dataframe with a new column 'C - blueveil'
    '''
    
    r, g, b = cv2.split(image) ## Split into RGB

    mask = (b > 60) & ((r - 46 < g) & (g < r + 15)) ## Create a mask based on the condition

    df.loc[im_id,'C - blueveil']=cv2.countNonZero(mask.astype(np.uint8)) ## Count non-zero (True) values in the mask
    return df