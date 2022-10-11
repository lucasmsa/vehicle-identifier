import cv2
import numpy as np


class ImageCorruptor:
    def __init__(self):
        pass

    def blur(self, image):
        return cv2.blur(image, (10, 10))

    def darken(self, image, gamma=0.5):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) *
                         255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

    def handle_resolution(self, image, scale_percent=20):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        dim = (width, height)
        print(width, height, dim)
        return cv2.resize(
            image, dim, interpolation=cv2.INTER_AREA)

    def display_image(self, image):
        cv2.imshow("Filtered image", image)
        cv2.waitKey(0)
