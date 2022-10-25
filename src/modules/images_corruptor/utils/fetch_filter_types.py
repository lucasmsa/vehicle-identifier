# dataset_corruptor = DatasetCorruptor()
# dataset_corruptor.run(1, [("BLUR", 10), ("DARKEN", 0.5), ("RESOLUTION", 80)])

# [] Fazer os testes de extração
# [] Salvar resultados em um csv

# img1.jpg, img2.jpg, img3.jpg...
# Para cada uma dessas aplicar os filtros
import cv2
from modules.images_corruptor.image_corruptor import ImageCorruptor
from modules.images_corruptor.dataset_corruptor import DatasetCorruptor


dataset_corruptor = DatasetCorruptor()
dataset_corruptor.fetch_random_images(1)
initial_image = cv2.imread("./assets/bright_blue_car.png")# cv2.imread("./src/modules/license_plates_detection/images/" + dataset_corruptor.random_images[0])
image_corruptor = ImageCorruptor()

def fetch_blur_image_results(image):
    print(image_corruptor.get_image_blur(image))
    initial_image_blur = image_corruptor.get_image_blur(image)
    image_corruptor.display_image(image)
    blurred_image = image_corruptor.blur(image, int(initial_image_blur*0.05))
    image_corruptor.display_image(blurred_image)
    print(image_corruptor.get_image_blur(blurred_image))
    
# fetch_blur_image_results(initial_image)

def fetch_brightness_image_results(image):
    print(image_corruptor.get_image_brightness(image))
    image_corruptor.display_image(image)
    darkened_image = image_corruptor.darken(image, 0.2)
    image_corruptor.display_image(darkened_image)
    print(image_corruptor.get_image_blur(darkened_image))
    
fetch_brightness_image_results(initial_image)

# 255 -> 100
# 200 -> X * 0.6