import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import cv2
from skimage import morphology
def convexity_score(mask):

    coords = np.transpose(np.nonzero(mask))

    hull = ConvexHull(coords)

    lesion_area = np.count_nonzero(mask)

    convex_hull_area = hull.volume + hull.area

    convexity = lesion_area / convex_hull_area
    
    return convexity

def measure_streaks(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesion_area = cv2.contourArea(contours[0])
    border_perimeter = cv2.arcLength(contours[0], True)
    if lesion_area == 0:
        irregularity = 0
    else:
        irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)

    return irregularity 

def get_compactness(mask):
    # mask = color.rgb2gray(mask)
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask - mask_eroded)

    return perimeter**2 / (4 * np.pi * area)

def feature_B(image,mask,im_id,df:pd.DataFrame):
    '''Takes mask of image, image and dataframe to write results in\\
    Find the convexity score and measures streaks for the mask and image respectively\\
    Outputs a modified dataframe with 2 new columns 'B - convexity','B - streaks'
    '''
    ## Choose one of these, comment out others
    # df.loc[im_id,'B - convexity']=convexity_score(mask)
    # df.loc[im_id,'B - streaks']=measure_streaks(image)
    df.loc[im_id,'B - compactness']=get_compactness(mask)
    return df