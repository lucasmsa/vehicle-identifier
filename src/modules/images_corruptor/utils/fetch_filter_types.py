import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, f'{os.getcwd()}/src')
from modules.images_corruptor.image_corruptor import ImageCorruptor
from modules.images_corruptor.dataset_corruptor import DatasetCorruptor

def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

dataset_corruptor = DatasetCorruptor()
dataset_corruptor.fetch_random_images(1)
initial_image = cv2.imread("./vehicle.png")
image_corruptor = ImageCorruptor()

def plot_filtered_image(initial_image, filtered_image, filename):
    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(initial_image)
    axes[0].axis("off")

    axes[1].imshow(filtered_image)
    axes[1].axis("off")

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

plot_filtered_image(initial_image, image_corruptor.blur(initial_image, 20), "./image-blurred-20%.png")   
plot_filtered_image(initial_image, image_corruptor.darken(initial_image, 0.4), "./image-darkened-0.4.png")
plot_filtered_image(initial_image, image_corruptor.handle_resolution(initial_image, 20), "./image-resolution-20%.png")