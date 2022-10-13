import cv2
import numpy as np
from operator import itemgetter


class LicensePlateDetector:
    CLASS = "license-plate"
    CONFIDENCE_THRESHOLD = 0.2
    BOUNDING_BOX_COLOR = (141, 63, 234)

    def __init__(self):
        self.network = cv2.dnn.readNet(
            "./yolov3-license_plates.weights", "./yolov3-license_plates-test.cfg")

    def run(self, image_path: str):
        self.image = cv2.imread(image_path)
        height, width, _ = self.image.shape
        blob = cv2.dnn.blobFromImage(
            self.image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.network.setInput(blob)
        output_layer_names = self.network.getUnconnectedOutLayersNames()
        layer_outputs = self.network.forward(output_layer_names)
        bounding_boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    bounding_boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(
            bounding_boxes,  confidences, self.CONFIDENCE_THRESHOLD, 0.4)

        coordinates, confidence_score = [], 0
        if len(indexes):
            for i in indexes.flatten():
                x, y, w, h = bounding_boxes[i]
                label = self.CLASS
                confidence = f"{int(confidences[i]*100)}%"
                cv2.rectangle(self.image, (x, y),
                              (x + w, y + h), self.BOUNDING_BOX_COLOR, 5)
                cv2.putText(self.image, label + ' ' + confidence,
                            (x, y - round(h/4)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                confidence_score = int(confidences[i]*100)
                coordinates.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                })
        
        return (coordinates, confidence_score)

    def crop_plate(self, image, coordinates):
        x, y, width, height = itemgetter(
            "x", "y", "width", "height")(coordinates)

        return image[y:y + height, x:x + width]
