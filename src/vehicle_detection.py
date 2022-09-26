import os
import cv2
import numpy as np
import collections
from operator import itemgetter
from infra.aws.aws_operations import AwsOperations
from modules.color_detection.knn_color_classifier import KNNColorClassifier
from modules.license_plates_detection.license_plates_detector import LicensePlateDetector
from modules.color_detection.color_histogram_feature_extraction import ColorHistogramFeatureExtraction
from config.vehicle_detection_constants import COCO_CLASS_NAMES, COLOR_TEST_DATA, COLOR_TRAIN_DATA, \
    CONFIDENCE_THRESHOLD, FONT_COLOR, FONT_SIZE, FONT_THICKNESS, INPUT_SIZE, MODEL_CONFIGURATION, MODEL_WEIGHTS,\
    NMS_THRESHOLD, REQUIRED_CLASS_INDICES, VEHICLE_TEMP_FILE_PATH


class VehicleClassifier:
    def __init__(self, image_path: str = None):
        self.network_model = cv2.dnn.readNetFromDarknet(
            MODEL_CONFIGURATION, MODEL_WEIGHTS)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(
            len(COCO_CLASS_NAMES), 3), dtype='uint8')
        self.detected_classes = []
        self.detection = []
        self.aws_operations = AwsOperations()
        self.color_histogram_feature_extraction = ColorHistogramFeatureExtraction()
        self.knn_color_classifier = KNNColorClassifier(
            COLOR_TRAIN_DATA, COLOR_TEST_DATA)
        self.image_path = image_path
        self.license_plates_detector = LicensePlateDetector()

    def pre_process_data(self):
        if self.image_path:
            self.image = cv2.imread(self.image_path)
        else:
            self.image = self.aws_operations.get_random_object()

        self.original_image = self.image.copy()
        blob = cv2.dnn.blobFromImage(
            self.image, 1 / 255, (INPUT_SIZE, INPUT_SIZE), [0, 0, 0], 1, crop=False)
        self.network_model.setInput(blob)
        layers_names = self.network_model.getLayerNames()
        output_names = [(layers_names[i - 1])
                        for i in self.network_model.getUnconnectedOutLayers()]
        self.outputs = self.network_model.forward(output_names)

    def extract_vehicle_informations(self, center_y: float, center_x: float, box_height: float, box_width: float) -> dict:
        vehicle_box_image = self.original_image[center_y:center_y +
                                                box_height, center_x:center_x + box_width]

        cv2.imwrite(VEHICLE_TEMP_FILE_PATH, vehicle_box_image)
        feature_extraction = self.color_histogram_feature_extraction.extract_color_histogram_from_image(
            VEHICLE_TEMP_FILE_PATH)
        self.color_histogram_feature_extraction.write_feature_data_file(
            feature_extraction, COLOR_TEST_DATA)
        color_predictions = self.knn_color_classifier.run()
        license_plates_coordinates = self.license_plates_detector.run(
            VEHICLE_TEMP_FILE_PATH)
        os.remove(VEHICLE_TEMP_FILE_PATH)

        return {
            "color": color_predictions[0],
            "license_plates_coordinates": license_plates_coordinates
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
        for i in indices.flatten():
            center_x, center_y, box_width, box_height = boxes[
                i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            color = [int(c) for c in self.colors[class_ids[i]]]
            name = COCO_CLASS_NAMES[class_ids[i]]
            self.detected_classes.append(name)

            vehicle_information = self.extract_vehicle_informations(
                center_y, center_x, box_height, box_width)

            vehicle_color, license_plate_coordinates = itemgetter(
                "color", "license_plates_coordinates")(vehicle_information)

            detected_vehicle_text = f'{vehicle_color.upper()} {name.upper()} {int(confidence_scores[i]*100)}%'

            cv2.putText(self.image, detected_vehicle_text, (center_x,
                        center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.rectangle(self.image, (center_x, center_y),
                          (center_x + box_width, center_y + box_height), color, 1)

            for license_plate_coordinate in license_plate_coordinates:
                x, y, width, height = itemgetter(
                    "x", "y", "width", "height")(license_plate_coordinate)

                x_position = x + center_x
                y_position = y + center_y

                cv2.rectangle(self.image, (x_position, y_position),
                              (x_position + width, y_position + height), color, 1)

            self.detection.append([center_x, center_y, box_width, box_height,
                                  REQUIRED_CLASS_INDICES.index(class_ids[i])])

    def print_image(self):
        frequency = collections.Counter(self.detected_classes)

        cv2.putText(self.image, "Car:        "+str(frequency['car']), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(self.image, "Motorbike:  "+str(frequency['motorbike']), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(self.image, "Bus:        "+str(frequency['bus']), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(self.image, "Truck:      "+str(frequency['truck']), (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)

        cv2.imshow("image", self.image)
        cv2.waitKey(0)

    def run(self):
        self.pre_process_data()
        self.post_process_data()
        self.print_image()


vehicle_classifier = VehicleClassifier("./assets/transit_image.png")
vehicle_classifier.run()
