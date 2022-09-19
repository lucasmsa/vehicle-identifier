import os
import cv2
import numpy as np
from tqdm import tqdm


class ColorHistogramFeatureExtraction:
    def __init__(self):
        pass

    def write_feature_data_file(self, feature_data, file_name):
        with open(file_name, "w") as file:
            file.write(feature_data)

    def extract_color_histogram_from_image(self, image_name):
        image = cv2.imread(image_name)
        rgb_colors, features, channels = (
            "b", "g", "r"), [], cv2.split(image)

        for (idx, (channel, _)) in enumerate(zip(channels, rgb_colors)):
            histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
            features.extend(histogram)
            peak_pixel_value = np.argmax(histogram)

            if idx == 0:
                blue = str(peak_pixel_value)
            elif idx == 1:
                green = str(peak_pixel_value)
            elif idx == 2:
                red = str(peak_pixel_value)
                feature_data = red + ',' + green + ',' + blue

        return feature_data

    def fetch_color_images(self, directory_images, color):
        for color_image in directory_images:
            feature_data = self.extract_color_histogram_from_image(
                f"{self.dataset_path}/{color}/{color_image}")
            color_image_feature_data = f"{feature_data},{color}\n"
            self.training_images_feature_data.append(color_image_feature_data)

    def train_color_histogram_data(self):
        self.current_directory_path = f"{os.path.dirname(os.path.realpath(__file__))}"
        self.dataset_path = f"{self.current_directory_path}/dataset"

        directory_colors = os.listdir(self.dataset_path)

        self.training_images_feature_data = []
        for directory_color in tqdm(directory_colors):
            self.fetch_color_images(os.listdir(
                f"{self.dataset_path}/{directory_color}"), directory_color)

        return self.write_feature_data_file("".join(self.training_images_feature_data), f"{self.current_directory_path}/training.data")


color_histogram_feature_extraction = ColorHistogramFeatureExtraction()
color_histogram_feature_extraction.train_color_histogram_data()
