import os
import cv2
import random
import numpy as np
from modules.images_corruptor.image_corruptor import ImageCorruptor

class DatasetCorruptor:
    DATASET_PATH = "./src/modules/license_plates_detection/images/"

    def __init__(self):
        self.image_corruptor = ImageCorruptor()
        self.filter_mapping = {
            "BLUR": lambda image, intensity: self.image_corruptor.blur(image, intensity),
            "DARKEN": lambda image, intensity: self.image_corruptor.darken(image, intensity),
            "RESOLUTION": lambda image, intensity: self.image_corruptor.handle_resolution(image, intensity)
        }

    def run(self, quantity, filter_types):
        self.fetch_random_images(quantity)
        self.filter_images(filter_types)

    def fetch_random_images(self, quantity):
        file_list = os.listdir(self.DATASET_PATH)
        images = list(
            filter(lambda filename: filename[-3:] == "JPG", file_list))
        self.random_images = random.choices(
            images,
            k=quantity)

    def filter_all_random_images(self, filter_types):
        if not len(filter_types):
            return

        for image_name in self.random_images:
            self.filter_image(image_name, filter_types)

    def filter_image(self, image_name, filter_tuple, equalize=False):
        open_cv_image = cv2.imread(self.DATASET_PATH + image_name)
        self.image_corruptor.display_image(open_cv_image)
        
        (filter_style, filter_intensity) = filter_tuple
        
        should_equalize_image = filter_style == "DARKEN" and equalize
        
        if should_equalize_image:
            image_points = self.image_corruptor.equalize(open_cv_image)
            open_cv_image = np.array(image_points)
            self.image_corruptor.display_image(open_cv_image)
        
        filtered_image = self.filter_mapping[filter_style](
            open_cv_image, filter_intensity)
        
        return filtered_image
            