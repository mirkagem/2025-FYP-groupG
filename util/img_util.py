import random
import os
import cv2
import numpy as np

#Import the function used to create masks
from mask_function import get_mask

#Function for reading the images
def readImageFile(file_path):
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray

#Function for saving images
def saveImageFile(img_rgb, file_path):
    try:
        # convert BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # save the image
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success

    except Exception as e:
        print(f"Error saving the image: {e}")
        return False

#The class that takes care of loading the pictures
class ImageDataLoader:
    #Here we initialize the variables and run the code required at the creation of the instance
    def __init__(self, directory, shuffle=False, transform=None):
        #The required variables, most of them representing paths are commonly used
        self.directory = directory
        self.shuffle = shuffle
        self.transform = transform
        self.img_path = directory + r'\images'
        self.mask_path = directory + r'\masks'

        #Gets a list of all of the paths of the images
        self.img_list= sorted([os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])
        #Gets a list of all of the names of the images
        self.images=os.listdir(self.img_path)
        #Gets a list of all of the names of the masks
        self.masks=os.listdir(self.mask_path)
        #Gets a list of all of the paths of the masks
        self.mask_list= sorted([os.path.join(self.mask_path, f) for f in os.listdir(self.mask_path) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])
        if not self.img_list:
            raise ValueError("No image files found in the directory.")

        # shuffle file list if required
        if self.shuffle:
            random.shuffle(self.img_list)

        # get the total number of batches
        self.num_batches = len(self.img_list)

    def __len__(self):
        return self.num_batches

    #This function iterrates through the images and applies their respective mask to them
    def __iter__(self):
        #A list which will contain the masked image, and its own mask and name
        masked_images = []
        #Iterating through the list of image paths
        for file_path in self.img_list:
            #This value tells if  the image doesnt have a mask
            noMask = False
            #Reads the image
            img_rgb, img_gray = readImageFile(file_path)
            #Searches through the list of names and looks for the name of the image
            for i in self.images:
                if i in file_path:
                    img_name = i
                    break
            #Then searches for the mask of the image, if it has one already
            for j in self.masks:
                if img_name == (j[:-9]+'.png'):
                    mask_name = j
                    noMask = False
                    break       
                else:  
                    noMask = True
            #If there's no mask it will create one, if there is one it will load it
            if noMask:
                mask = get_mask(img_rgb)
            else:
                mask=cv2.imread(f'{self.mask_path}/{mask_name}', cv2.IMREAD_GRAYSCALE)
                #Checks to makes sure that the mask its not a black image, if it is then it will create a new mask
                if np.sum(np.nonzero(mask))==0: ## If mask black image
                    mask=get_mask(img_rgb) ## Make own mask
            #Here it applies the mask to the image
            mask = mask.astype(int)
            image_masked = img_rgb.copy()
            image_masked[mask == 0] = 0

            #And appends everything to the list of masked images
            masked_image = [image_masked, mask, img_name]
            masked_images.append(masked_image)

            #Here it saves the masked images as JPEGs
            dir_path = self.directory + r'\New'
            os.makedirs(dir_path, exist_ok=True)
            saveImageFile(image_masked, os.path.join(dir_path, os.path.basename(file_path)))
        
        return masked_images

