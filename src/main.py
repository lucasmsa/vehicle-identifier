from modules.images_corruptor.dataset_corruptor import DatasetCorruptor
from modules.vehicle_detection.vehicle_detection import VehicleClassifier

dataset_corruptor = DatasetCorruptor()
vehicle_detection = VehicleClassifier()
dataset_corruptor.fetch_random_images(1)

print("random images", dataset_corruptor.random_images)

def prepend_filter_intensity(filter_type, intensities):
    return list(map(lambda intensity: (filter_type, intensity), intensities))

blur_mapping = prepend_filter_intensity("BLUR", [5, 7, 9, 11, 13, 15, 17, 19, 21])
darken_mapping = prepend_filter_intensity("DARKEN", [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
resolution_mapping = prepend_filter_intensity("RESOLUTION", [60, 50, 40, 30, 20])

for image in dataset_corruptor.random_images:
    for blur in blur_mapping:
        filtered_image = dataset_corruptor.filter_image(image, blur)
        try:
            vehicle_detection.run(filtered_image)
            print(f"Confidence: {vehicle_detection.license_plate_confidence} | Blur: {blur}")
        except Exception as e:
            print(str(e))
        
        
# vehicle_classifier = VehicleClassifier("./assets/brazilian-car-back.jpg")
# vehicle_classifier.run()