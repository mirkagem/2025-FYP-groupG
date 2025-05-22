import random
import os
import cv2


def readImageFile(file_path):
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray


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


class ImageDataLoader:
    def __init__(self, directory, shuffle=False, transform=None):
        self.directory = directory
        self.shuffle = shuffle
        self.transform = transform
        self.img_path = directory + r'\images'
        self.mask_path = directory + r'\masks'

        # get a sorted list of all files in the directory
        # fill in with your own code below
        self.img_list= sorted([os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])
        print(self.img_list)
        self.images=os.listdir(self.img_path)
        self.masks=os.listdir(self.mask_path)
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

    def __iter__(self):
        # Delete hair if annotation for hair is NOT 0 
        for file_path in self.img_list:
            noMask = False
            img_rgb, img_gray = readImageFile(file_path)
            for i in self.images:
                if i in file_path:
                    img_name = i
                    break
            for j in self.masks:
                if img_name == (j[:-9]+'.png'):
                    mask_name = j
                    break
                else:
                    noMask = True
            
            if noMask:
                #You load the img and create a mask for it
                pass
            else:
                mask=cv2.imread(f'{self.mask_path}/{mask_name}', cv2.IMREAD_GRAYSCALE)
            print(f'{img_name} {img_rgb.shape} {mask_name} {mask.shape}')
            mask = mask.astype(int)
            image_masked = img_rgb.copy()
            image_masked[mask == 0] = 0
            dir_path = self.directory + r'\New'
            os.makedirs(dir_path, exist_ok=True)
            saveImageFile(image_masked, os.path.join(dir_path, os.path.basename(file_path)))

