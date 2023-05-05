import pandas as pd
from tqdm import tqdm
from config.vehicle_detection_constants import RESULTS_PATH
from modules.images_corruptor.image_corruptor import ImageCorruptor
from modules.images_corruptor.dataset_corruptor import DatasetCorruptor
from modules.vehicle_detection.vehicle_detection import VehicleClassifier

image_corruptor = ImageCorruptor()
dataset_corruptor = DatasetCorruptor()
vehicle_detection = VehicleClassifier()
dataset_corruptor.fetch_random_images(300)

def prepend_filter_intensity(filter_type, intensities):
    return list(map(lambda intensity: (filter_type, intensity), intensities))

blur_mapping = prepend_filter_intensity("BLUR", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 75, 85, 95])
darken_mapping = prepend_filter_intensity("DARKEN", [0.05, 0.1, 0.15, 0.22, 0.25, 0.28, 0.3, 0.33, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
darken_equalized_mapping = prepend_filter_intensity("DARKEN", [0.05, 0.1, 0.15, 0.22, 0.25, 0.28, 0.3, 0.33, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
resolution_mapping = prepend_filter_intensity("RESOLUTION", [3, 5, 7, 10, 20, 25, 35, 45, 60, 75, 90, 95])


def generate_corrupted_images_results(random_images, filter_mapping, should_equalize=False):
    filter_type = filter_mapping[0][0]
    image_names, intensities, match_percentages = [], [], []
    for image_name in tqdm(random_images):
        for filter_tuple in filter_mapping:
            filtered_image = dataset_corruptor.filter_image(image_name, filter_tuple, should_equalize)  
            image_names.append(image_name)
            intensities.append(filter_tuple[1])
            try: 
                vehicle_detection.run(filtered_image)
                match_percentages.append(vehicle_detection.license_plate_confidence)
            except Exception as e:
                match_percentages.append(f"ERROR: {str(e)}")
                
    dictionary = {
        "image_names": image_names,
        f"{filter_type.lower()}_intensities": intensities,
        "match_percentages": match_percentages
    }

    dataframe = pd.DataFrame(dictionary)
    csv_filename = f"{filter_type.lower()}_equalized" if should_equalize else f"{filter_type.lower()}"
    
    dataframe.to_csv(f"{RESULTS_PATH}/{csv_filename}.csv", encoding='utf-8')
   
generate_corrupted_images_results(dataset_corruptor.random_images, darken_mapping) 
generate_corrupted_images_results(dataset_corruptor.random_images, resolution_mapping)
generate_corrupted_images_results(dataset_corruptor.random_images, blur_mapping)
generate_corrupted_images_results(dataset_corruptor.random_images, darken_equalized_mapping, True) 