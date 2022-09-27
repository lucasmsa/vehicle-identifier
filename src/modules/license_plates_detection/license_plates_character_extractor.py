import cv2
import easyocr
import pytesseract
import skimage.segmentation


class LicensePlateCharacterExtractor:
    ALPHANUMERIC_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    PAGE_SEGMENTATION_METHOD = 7

    def __init__(self):
        self.options = f"-c tessedit_char_whitelist={self.ALPHANUMERIC_CHARACTERS} --psm {self.PAGE_SEGMENTATION_METHOD}"

    def run(self, image_path: str, clear_border=False, method: str = "easyocr"):
        self.pre_process(image_path, clear_border)
        return self.execute_ocr(method)

    def execute_ocr(self, method: str):
        license_plate_text = ""
        if method == "easyocr":
            reader = easyocr.Reader(['en'])
            license_plate_informations = reader.readtext(self.grayscale_image)
            license_plate_text = license_plate_informations[0][1]
        else:
            license_plate_text = pytesseract.image_to_string(
                self.grayscale_image, config=self.options)[:-1]

        return license_plate_text

    def pre_process(self, image_path: str, clear_border=False):
        self.grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        threshold = cv2.threshold(self.grayscale_image, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        threshold_image = threshold[1]
        if clear_border:
            threshold_image = skimage.segmentation.clear_border(
                threshold[1])

        self.contour_image = cv2.fastNlMeansDenoising(
            threshold_image, None, 20, 7, 21)

        cv2.imshow("image", self.contour_image)
        cv2.waitKey(0)
