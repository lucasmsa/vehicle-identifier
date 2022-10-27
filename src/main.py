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

print("random images", dataset_corruptor.random_images)

def prepend_filter_intensity(filter_type, intensities):
    return list(map(lambda intensity: (filter_type, intensity), intensities))

blur_mapping = prepend_filter_intensity("BLUR", [5, 8, 11, 14, 17, 20])
darken_mapping = prepend_filter_intensity("DARKEN", [0.45, 0.4, 0.35, 0.3, 0.25, 0.2])
resolution_mapping = prepend_filter_intensity("RESOLUTION", [30, 25, 20, 15, 10, 5])


def generate_corrupted_images_results(random_images, filter_mapping):
    filter_type = filter_mapping[0][0]
    image_names, intensities, match_percentages = [], [], []
    for image in tqdm(random_images):
        for filter_tuple in filter_mapping:
            filtered_image = dataset_corruptor.filter_image(image, filter_tuple)  
            image_names.append(image)
            intensities.append(filter_tuple[1])
            try: 
                vehicle_detection.run(filtered_image)
                match_percentages.append(vehicle_detection.license_plate_confidence)
                print(f"Confidence: {vehicle_detection.license_plate_confidence} | Filter: {filter_tuple}")
            except Exception as e:
                match_percentages.append(f"ERROR: {str(e)}")
                print(str(e))
                
    dictionary = {
        "image_names": image_names,
        f"{filter_type.lower()}_intensities": intensities,
        "match_percentages": match_percentages
    }

    dataframe = pd.DataFrame(dictionary)
    dataframe.to_csv(f"{RESULTS_PATH}/{filter_type.lower()}.csv", encoding='utf-8')
   
generate_corrupted_images_results(dataset_corruptor.random_images, blur_mapping)
generate_corrupted_images_results(dataset_corruptor.random_images, darken_mapping) 
generate_corrupted_images_results(dataset_corruptor.random_images, resolution_mapping)