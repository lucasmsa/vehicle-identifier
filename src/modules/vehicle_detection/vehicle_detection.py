import os
import cv2
import collections
import numpy as np
from operator import itemgetter
from infra.aws.aws_operations import AwsOperations
from modules.color_detection.cnn_vehicle_color_classifier import CnnColorClassifier
from modules.license_plates_detection.license_plates_detector import LicensePlateDetector
from modules.license_plates_detection.license_plates_character_extractor import LicensePlateCharacterExtractor
from config.vehicle_detection_constants import COCO_CLASS_NAMES, CONFIDENCE_THRESHOLD, FONT_COLOR, FONT_SIZE, FONT_THICKNESS, INPUT_SIZE, \
    LICENSE_PLATE_TEMP_FILE_PATH, MODEL_CONFIGURATION, MODEL_WEIGHTS,\
    NMS_THRESHOLD, REQUIRED_CLASS_INDICES, VEHICLE_TEMP_FILE_PATH


class VehicleClassifier:
    def __init__(self):
        self.colors = np.random.randint(0, 255, size=(
            len(COCO_CLASS_NAMES), 3), dtype='uint8')
        self.network_model = cv2.dnn.readNetFromDarknet(
            MODEL_CONFIGURATION, MODEL_WEIGHTS)
        self.cnn_color_classifier = CnnColorClassifier()
        
    def get_image(self, image):
        if isinstance(image, str):
            if image == "AWS":
                self.image = self.aws_operations.get_random_object()
            else:
                self.image = cv2.imread(image)
        else: 
            self.image = image
        
    def get_initial_values(self):
        np.random.seed(42)
        self.detection = []
        self.detected_classes = []
        self.license_plate_confidence = 0
        self.aws_operations = AwsOperations()
        self.license_plates_detector = LicensePlateDetector()
        self.license_plates_character_extractor = LicensePlateCharacterExtractor()
        

    def pre_process_data(self):
        self.original_image = self.image.copy()
        blob = cv2.dnn.blobFromImage(
            self.image, 1 / 255, (INPUT_SIZE, INPUT_SIZE), [0, 0, 0], 1, crop=False)
        self.network_model.setInput(blob)
        layers_names = self.network_model.getLayerNames()
        output_names = [(layers_names[i - 1])
                        for i in self.network_model.getUnconnectedOutLayers()]
        self.outputs = self.network_model.forward(output_names)
        
    def extract_license_plate_confidence(self, license_plate_confidence):
        print("CURRENT LICENSE PLATE CONFIDENCE", self.license_plate_confidence)
        print("INCOMING LICENSE PLATE CONFIDENCE", license_plate_confidence)
        self.license_plate_confidence = max(self.license_plate_confidence, license_plate_confidence)
        
    def filter_vehicle_box_negative_values(self, center_y: float, center_x: float, box_height: float, box_width: float):
        [center_y, center_x, box_height, box_width] = list(\
            map(lambda x: abs(x), [center_y, center_x, box_height, box_width]))
        
        return (center_y, center_x, box_height, box_width)
        

    def extract_vehicle_informations(self, center_y: float, center_x: float, box_height: float, box_width: float) -> dict:
        vehicle_box_image = self.original_image[center_y:center_y +
                                                box_height, center_x:center_x + box_width]
        try:
            cv2.imwrite(VEHICLE_TEMP_FILE_PATH, vehicle_box_image)
        except Exception as e:
            raise Exception(str(e)) 
            
        color_predictions = self.cnn_color_classifier.run_test(
            VEHICLE_TEMP_FILE_PATH)
        (license_plates_coordinates, license_plate_confidence) = self.license_plates_detector.run(
            VEHICLE_TEMP_FILE_PATH)
        license_plate_text = ""
        self.extract_license_plate_confidence(license_plate_confidence)

        if len(license_plates_coordinates):
            license_plate_image = self.license_plates_detector.crop_plate(vehicle_box_image,
                                                                          license_plates_coordinates[0])
            cv2.imwrite(LICENSE_PLATE_TEMP_FILE_PATH, license_plate_image)
            license_plate_text = self.license_plates_character_extractor.run(
                LICENSE_PLATE_TEMP_FILE_PATH, True)

        os.remove(VEHICLE_TEMP_FILE_PATH)

        return {
            "color": color_predictions[0],
            "license_plates_coordinates": license_plates_coordinates,
            "license_plate_text": license_plate_text
        }

    def post_process_data(self):
        original_image_height, original_image_width = self.image.shape[:2]
        boxes, class_ids, confidence_scores = [], [], []

        for output in self.outputs:
            for forward_results in output:
                scores = forward_results[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id in REQUIRED_CLASS_INDICES:
                    if confidence > CONFIDENCE_THRESHOLD:
                        w, h = int(
                            forward_results[2]*original_image_width), int(forward_results[3]*original_image_height)
                        x, y = int(
                            (forward_results[0]*original_image_width)-w/2), int((forward_results[1]*original_image_height)-h/2)
                        boxes.append([x, y, w, h])
                        class_ids.append(class_id)
                        confidence_scores.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(
            boxes, confidence_scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        if len(indices) > 0:
            for i in indices.flatten():
                center_x, center_y, box_width, box_height = boxes[
                    i][0], boxes[i][1], boxes[i][2], boxes[i][3]

                color = [int(c) for c in self.colors[class_ids[i]]]
                name = COCO_CLASS_NAMES[class_ids[i]]
                self.detected_classes.append(name)
                center_y, center_x, box_height, box_width = self.filter_vehicle_box_negative_values(\
                                                        center_y, center_x, box_height, box_width)
                vehicle_information = self.extract_vehicle_informations(
                    center_y, center_x, box_height, box_width)

                vehicle_color, license_plate_coordinates, license_plate_text = itemgetter(
                    "color", "license_plates_coordinates", "license_plate_text")(vehicle_information)

                detected_vehicle_text = f'{vehicle_color.upper()} {name.upper()} {int(confidence_scores[i]*100)}%'

                cv2.putText(self.image, detected_vehicle_text, (center_x,
                            center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.rectangle(self.image, (center_x, center_y),
                            (center_x + box_width, center_y + box_height), color, 1)

                if license_plate_coordinates:
                    self.paint_license_plate_characters(
                        license_plate_coordinates, center_x, center_y, color, license_plate_text)
                    os.remove(LICENSE_PLATE_TEMP_FILE_PATH)

                self.detection.append([center_x, center_y, box_width, box_height,
                                    REQUIRED_CLASS_INDICES.index(class_ids[i])])
        else:
            raise Exception("No vehicle detected")

    def paint_license_plate_characters(self, license_plate_coordinates, center_x, center_y, color, license_plate_text):
        x, y, width, height = itemgetter(
            "x", "y", "width", "height")(license_plate_coordinates[0])
        x_position = x + center_x
        y_position = y + center_y

        cv2.rectangle(self.image, (x_position, y_position),
                      (x_position + width, y_position + height), color, 1)
        cv2.putText(self.image, license_plate_text.upper(),
                    (x_position, int(y_position - height/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def print_image(self):
        frequency = collections.Counter(self.detected_classes)
        for index, vehicle in enumerate(["Car", "Motorbike", "Bus", "Truck"]):
            cv2.putText(self.image, f"{vehicle}: {frequency[vehicle.lower()]}", (20, 40 + (index*20)),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
        cv2.imshow("image", self.image)
        cv2.waitKey(0)

    def run(self, image_path):
        self.get_initial_values()
        self.get_image(image_path)
        self.pre_process_data()
        self.post_process_data()
        
        return self.license_plate_confidence or 0