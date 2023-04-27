# dataset_corruptor = DatasetCorruptor()
# dataset_corruptor.run(1, [("BLUR", 10), ("DARKEN", 0.5), ("RESOLUTION", 80)])

# [] Fazer os testes de extração
# [] Salvar resultados em um csv

# img1.jpg, img2.jpg, img3.jpg...
# Para cada uma dessas aplicar os filtros
import cv2
import os
import sys
import numpy as np
sys.path.insert(1, f'{os.getcwd()}/src')
from modules.images_corruptor.image_corruptor import ImageCorruptor
from modules.images_corruptor.dataset_corruptor import DatasetCorruptor


dataset_corruptor = DatasetCorruptor()
dataset_corruptor.fetch_random_images(1)
initial_image = cv2.imread("./vehicle.png")
image_corruptor = ImageCorruptor()

image_blurred = image_corruptor.blur(initial_image, 20)
image_corruptor.display_image(image_blurred)
cv2.imwrite("./image-blurred-20%.png", image_blurred)

image_darkened = image_corruptor.darken(initial_image, 0.4)
image_corruptor.display_image(image_darkened)
cv2.imwrite("./image-darkened-0.4.png", image_darkened)

image_resolution = image_corruptor.handle_resolution(initial_image, 20)
image_corruptor.display_image(image_resolution)
cv2.imwrite("./image-resolution-20%.png", image_resolution)