import os
from turtle import color


class ColorHistogramFeatureExtraction:
    def __init__(self):
        pass

    def fetch_color_images(self, directory_images, color):
        for color_image in directory_images:
            print("color image is: ", color_image)

    def train_color_histogram_data(self):
        dataset_path = f"{os.path.dirname(os.path.realpath(__file__))}/dataset"
        directory_colors = os.listdir(dataset_path)
        for directory_color in directory_colors:
            self.fetch_color_images(os.listdir(
                f"{dataset_path}/{directory_color}"), color)


color_histogram_feature_extraction = ColorHistogramFeatureExtraction()
color_histogram_feature_extraction.train_color_histogram_data()
