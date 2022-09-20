import os
import cv2
import numpy as np
import collections
from aws_operations import AwsOperations
from color_detection.color_histogram_feature_extraction import ColorHistogramFeatureExtraction
from color_detection.knn_color_classifier import KNNColorClassifier


class VehicleClassifier:
    FONT_SIZE = 0.5
    INPUT_SIZE = 320
    FONT_THICKNESS = 2
    NMS_THRESHOLD = 0.2
    FONT_COLOR = (0, 0, 255)
    CONFIDENCE_THRESHOLD = 0.2
    MODEL_WEIGHTS = 'yolov3-320.weights'
    REQUIRED_CLASS_INDICES = [2, 3, 5, 7]
    MODEL_CONFIGURATION = 'yolov3-320.cfg'
    COLOR_TEST_DATA = "./src/color_detection/test.data"
    VEHICLE_TEMP_FILE_PATH = "./assets/vehicle_box.png"
    COLOR_TRAIN_DATA = "./src/color_detection/training.data"
    COCO_CLASS_NAMES = open('coco.names').read().strip().split('\n')

    def __init__(self, image_path=None):
        self.network_model = cv2.dnn.readNetFromDarknet(
            self.MODEL_CONFIGURATION, self.MODEL_WEIGHTS)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(
            len(self.COCO_CLASS_NAMES), 3), dtype='uint8')
        self.detected_classes = []
        self.detection = []
        self.aws_operations = AwsOperations()
        self.color_histogram_feature_extraction = ColorHistogramFeatureExtraction()
        self.knn_color_classifier = KNNColorClassifier(
            self.COLOR_TRAIN_DATA, self.COLOR_TEST_DATA)
        self.image_path = image_path

    def pre_process_data(self):
        if self.image_path:
            self.image = cv2.imread(self.image_path)
        else:
            self.image = self.aws_operations.get_random_object()

        self.original_image = self.image.copy()
        blob = cv2.dnn.blobFromImage(
            self.image, 1 / 255, (self.INPUT_SIZE, self.INPUT_SIZE), [0, 0, 0], 1, crop=False)
        self.network_model.setInput(blob)
        layers_names = self.network_model.getLayerNames()
        output_names = [(layers_names[i - 1])
                        for i in self.network_model.getUnconnectedOutLayers()]
        self.outputs = self.network_model.forward(output_names)

    def extract_vehicle_color(self, center_y, center_x, box_height, box_width):
        vehicle_box_image = self.original_image[center_y:center_y +
                                                box_height, center_x:center_x + box_width]

        cv2.imwrite(self.VEHICLE_TEMP_FILE_PATH, vehicle_box_image)
        feature_extraction = self.color_histogram_feature_extraction.extract_color_histogram_from_image(
            self.VEHICLE_TEMP_FILE_PATH)
        self.color_histogram_feature_extraction.write_feature_data_file(
            feature_extraction, self.COLOR_TEST_DATA)
        color_predictions = self.knn_color_classifier.run()

        os.remove(self.VEHICLE_TEMP_FILE_PATH)
        cv2.imshow("vehicle box sharpened", vehicle_box_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return color_predictions[0]

    def post_process_data(self):
        height, width = self.image.shape[:2]
        boxes, class_ids, confidence_scores = [], [], []

        for output in self.outputs:
            for forward_results in output:
                scores = forward_results[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id in self.REQUIRED_CLASS_INDICES:
                    if confidence > self.CONFIDENCE_THRESHOLD:
                        w, h = int(
                            forward_results[2]*width), int(forward_results[3]*height)
                        x, y = int(
                            (forward_results[0]*width)-w/2), int((forward_results[1]*height)-h/2)
                        boxes.append([x, y, w, h])
                        class_ids.append(class_id)
                        confidence_scores.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(
            boxes, confidence_scores, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        for i in indices.flatten():
            center_x, center_y, box_width, box_height = boxes[
                i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            color = [int(c) for c in self.colors[class_ids[i]]]
            name = self.COCO_CLASS_NAMES[class_ids[i]]
            self.detected_classes.append(name)

            vehicle_color = self.extract_vehicle_color(
                center_y, center_x, box_height, box_width)

            detected_vehicle_text = f'{vehicle_color.upper()} {name.upper()} {int(confidence_scores[i]*100)}%'

            cv2.putText(self.image, detected_vehicle_text, (center_x,
                        center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.rectangle(self.image, (center_x, center_y),
                          (center_x + box_width, center_y + box_height), color, 1)

            self.detection.append([center_x, center_y, box_width, box_height,
                                  self.REQUIRED_CLASS_INDICES.index(class_ids[i])])

    def print_image(self):
        frequency = collections.Counter(self.detected_classes)

        cv2.putText(self.image, "Car:        "+str(frequency['car']), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.FONT_COLOR, self.FONT_THICKNESS)
        cv2.putText(self.image, "Motorbike:  "+str(frequency['motorbike']), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.FONT_COLOR, self.FONT_THICKNESS)
        cv2.putText(self.image, "Bus:        "+str(frequency['bus']), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.FONT_COLOR, self.FONT_THICKNESS)
        cv2.putText(self.image, "Truck:      "+str(frequency['truck']), (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.FONT_COLOR, self.FONT_THICKNESS)

        cv2.imshow("image", self.image)
        cv2.waitKey(0)

    def run(self):
        self.pre_process_data()
        self.post_process_data()
        self.print_image()


vehicle_classifier = VehicleClassifier()
vehicle_classifier.run()
