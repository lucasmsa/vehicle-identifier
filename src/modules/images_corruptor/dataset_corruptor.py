import os
import cv2
import random
from image_corruptor import ImageCorruptor


class DatasetCorruptor:
    DATASET_PATH = "./src/modules/license_plates_detection/images/"

    def __init__(self):
        self.image_corruptor = ImageCorruptor()
        self.filter_mapping = {
            "BLUR": lambda image: self.image_corruptor.blur(image),
            "DARKEN": lambda image: self.image_corruptor.darken(image),
            "RESOLUTION": lambda image: self.image_corruptor.handle_resolution(image)
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

    def filter_images(self, filter_types):
        if not len(filter_types):
            return

        for image_name in self.random_images:
            image = cv2.imread(self.DATASET_PATH + image_name)
            for filter_type in filter_types:
                filtered_image = self.filter_mapping[filter_type](image)
                self.image_corruptor.display_image(filtered_image)


dataset_corruptor = DatasetCorruptor()
dataset_corruptor.run(5, ["BLUR", "DARKEN", "RESOLUTION"])


# 1. Pegar imagens aleatórias do dataset
# 2. Para cada uma delas aplicar operação do imageCorruptor
# 3. Faz os testes de extração [amanhã / quarta]
# 4. Salvar resultados em um csv [amanhã / quarta]
