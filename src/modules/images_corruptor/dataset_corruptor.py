import os
import cv2
import random
from image_corruptor import ImageCorruptor


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

    def filter_image(self, image_name, filter_type):
        image = cv2.imread(self.DATASET_PATH + image_name)
        
        (filter_style, filter_intensity) = filter_type
        
        filtered_image = self.filter_mapping[filter_style](
            image, filter_intensity)
        
        # self.image_corruptor.display_image(filtered_image)
        
        return filtered_image
            


# dataset_corruptor = DatasetCorruptor()
# dataset_corruptor.run(1, [("BLUR", 10), ("DARKEN", 0.5), ("RESOLUTION", 80)])

# [] Fazer os testes de extração
# [] Salvar resultados em um csv


# img1.jpg, img2.jpg, img3.jpg...
# Para cada uma dessas aplicar os filtros

dataset_corruptor = DatasetCorruptor()
dataset_corruptor.fetch_random_images(1)
initial_image = cv2.imread("./src/modules/license_plates_detection/images/" + dataset_corruptor.random_images[0])
image_corruptor = ImageCorruptor()

def fetch_blur_image_results(image):
    print(image_corruptor.get_image_blur(image))
    image_corruptor.display_image(image)
    blurred_image = image_corruptor.blur(image, 10)
    image_corruptor.display_image(blurred_image)
    print(image_corruptor.get_image_blur(blurred_image))

def fetch_brightness_image_results(image):
    print(image_corruptor.get_image_brightness(image))
    image_corruptor.display_image(image)
    darkened_image = image_corruptor.darken(image, 0.2)
    image_corruptor.display_image(darkened_image)
    print(image_corruptor.get_image_blur(darkened_image))
    
fetch_brightness_image_results(initial_image)