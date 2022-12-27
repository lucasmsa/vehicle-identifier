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
initial_image = cv2.imread("./assets/brazilian-car-back.jpg")
image_corruptor = ImageCorruptor()

def fetch_blur_image_results(image):
    initial_image_blur = image_corruptor.get_image_blur_fft(image)
    print('initial image blur: ', initial_image_blur)
    image_corruptor.display_image(image)
    blurred_image = image_corruptor.blur(image, int(6))
    image_corruptor.display_image(blurred_image)
    final_image_blur = image_corruptor.get_image_blur_fft(blurred_image)
    print('final image blur: ', final_image_blur)
    
fetch_blur_image_results(initial_image)

def fetch_brightness_image_results(image):
    # initial_image_brightness = image_corruptor.get_image_brightness(image)
    # print("Initial image brightness: ", initial_image_brightness)
    # image_corruptor.display_image(image)
    # normalized_image_brightness = initial_image_brightness / 255
    # darkened_image = image_corruptor.darken(image, 0.2)
    # image_corruptor.display_image(darkened_image)
    # print("New image brightness: ", image_corruptor.get_image_brightness(darkened_image))
    print("Initial image brightness: ", image_corruptor.get_image_brightness(image))
    image_corruptor.display_image(image)
    image_points = image_corruptor.equalize(image)
    open_cv_image = np.array(image_points)
    image_corruptor.display_image(open_cv_image)
    print("Image brightness after equalization: ", image_corruptor.get_image_brightness(open_cv_image))
    
    
# fetch_brightness_image_results(initial_image)

# 255 -> 100
# 200 -> X * 0.6