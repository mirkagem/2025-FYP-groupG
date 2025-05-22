from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
def get_mask(image):
    gray_im = rgb2gray(image)
    blurred_im = gaussian(gray_im, sigma=1.0)
    t=threshold_otsu(blurred_im)
    mask = gray_im<t
    return mask