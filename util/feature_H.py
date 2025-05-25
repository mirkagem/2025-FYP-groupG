import cv2
import numpy as np
##NOTE: mess around with threshold to get better values, might be too sensitive - adaptive threshold bad!
def removeHair(img_org, img_gray, kernel_size=25, threshold=0.1, radius=3.0):
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8) #chatgpt fix, otherwise it wouldn't work

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out
def feature_H(image,df_id,df):
    '''Takes image, image ID and DataFrame, returns the amount of hair pixels found by the removeHair function'''
    gray_im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _,thresh,_=removeHair(image,gray_im) # uses the removeHair function
    thresh=np.where(thresh==255,1,0)
    df.loc[df_id,'Hair']=thresh.sum()/(thresh.shape[0]*thresh.shape[1])