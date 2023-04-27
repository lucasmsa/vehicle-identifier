import cv2
import math
import operator
import numpy as np
from functools import reduce
from PIL import Image, ImageStat

class ImageCorruptor:
    RGB_IMAGE_LAYERS = 3
    
    def __init__(self):
        pass
    
    def get_image_blur(self, image):
        return "BLUR"
    
    def get_image_blur_fft(self, image, size=60):
        # grab the dimensions of the image and use the dimensions to
        # derive the center (x, y) - coordinates
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (h, w) = gray_image.shape
        (c_x, c_y) = (int(w / 2.0), int(h / 2.0))
        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more
        # easy to analyze
        fft = np.fft.fft2(gray_image)
        fft_shift = np.fft.fftshift(fft)
        fft_shift[c_y - size:c_y + size, c_x - size:c_x + size] = 0
        fft_shift = np.fft.ifftshift(fft_shift)
        recon = np.fft.ifft2(fft_shift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        # the image will be considered "blurry" if the mean value of the
        # magnitudes is less than the threshold value
        return round(mean, 2)
        
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
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
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