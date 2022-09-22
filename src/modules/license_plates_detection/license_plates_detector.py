import cv2
import numpy as np


class LicensePlateDetector:
    CLASS = "license-plate"
    CONFIDENCE_THRESHOLD = 0.2
    BOUNDING_BOX_COLOR = (141, 63, 234)

    def __init__(self):
        self.network = cv2.dnn.readNet(
            "./yolov3-license_plates.weights", "./yolov3-license_plates-test.cfg")

    def run(self, image_path: str) -> list:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
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

        coordinates = []
        if len(indexes):
            for i in indexes.flatten():
                x, y, w, h = bounding_boxes[i]
                label = self.CLASS
                confidence = f"{int(confidences[i]*100)}%"
                cv2.rectangle(image, (x, y),
                              (x + w, y + h), self.BOUNDING_BOX_COLOR, 5)
                cv2.putText(image, label + ' ' + confidence,
                            (x, y - round(h/4)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                coordinates.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                })

        cv2.imshow("image", image)
        cv2.waitKey(0)

        return coordinates

    def crop_plate(self):
        x, y, w, h = self.coordinates
        roi = self.img[y:y + h, x:x + w]
        self.roi_image = roi
