import cv2
import math
import numpy as np
from PIL import Image, ImageStat

class ImageCorruptor:
    def __init__(self):
        pass
    
    def get_image_blur(self, image):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplace_variance = round(cv2.Laplacian(grayscale_image, cv2.CV_64F).var(), 2)
        
        return laplace_variance
    
    def get_image_resolution(self, image):
        height = image.shape[0]
        width = image.shape[1]
        
        return {
            "height": height,
            "width": width
        }
        
    def get_image_brightness(self, image):
        pil_image = Image.fromarray(image)
        stat = ImageStat.Stat(pil_image)
        r,g,b = stat.mean
        return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

    def blur(self, image, blur_intensity=10):
        return cv2.blur(image, (blur_intensity, blur_intensity))

    def darken(self, image, gamma=0.8):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) *
                         255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

    def handle_resolution(self, image, scale_percent=80):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        dim = (width, height)
        
        return cv2.resize(
            image, dim, interpolation=cv2.INTER_AREA)

    def display_image(self, image):
        cv2.imshow("Filtered image", image)
        cv2.waitKey(0)