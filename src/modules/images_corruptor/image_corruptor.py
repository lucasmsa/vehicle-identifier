from functools import reduce
import operator
import cv2
import math
import numpy as np
from PIL import Image, ImageStat

class ImageCorruptor:
    RGB_IMAGE_LAYERS = 3
    
    def __init__(self):
        pass
    
    def get_image_blur(self, image):
        # blur_map = blur_detector.detectBlur(image, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3)
        # print('blur_map *' + blur_map)

        return "BLUR"
    
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
        return round(math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)), 2)
    

    def equalize(self, image_array):
        image = Image.fromarray(image_array)
        h = image.convert("L").histogram()
        lut = []
        for b in range(0, len(h), 256):
            step = reduce(operator.add, h[b:b+256]) / 255
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + h[i+b]
                
        return image.point(lut*self.RGB_IMAGE_LAYERS)

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