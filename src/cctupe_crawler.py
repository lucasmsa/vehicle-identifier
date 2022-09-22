import os
import uuid
import random
from config.url_constants import CCTUPE_URL
from aws_operations import AwsOperations
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from transit_camera_crawler import TransitCameraCrawler
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException


class CCTUPECrawler(TransitCameraCrawler):
    IMAGE_PATH = f"assets/{str(uuid.uuid4())}.png"

    def __init__(self):
        self.setup_webdriver()
        self.aws_operations = AwsOperations()

    def run(self):
        self.driver.get(CCTUPE_URL)
        self.select_camera()
        self.print_image()
        self.upload_image_to_cloud()
        self.finish_process()

    def upload_image_to_cloud(self):
        if os.path.exists(self.IMAGE_PATH):
            self.aws_operations.upload_file(self.IMAGE_PATH)
            os.remove(self.IMAGE_PATH)
            print(f"{self.IMAGE_PATH} Image uploaded successfully!")

    def print_image(self):
        print("Trying to print image")
        try:
            wait_config = WebDriverWait(self.driver, 10)
            self.image = wait_config.until(
                EC.visibility_of_element_located((By.ID, "imagem")))
            self.image.screenshot(self.IMAGE_PATH)
        except:
            self.finish_process()

    def select_camera(self):
        print("Selecting camera element")
        clickable_cameras = self.driver.find_elements(
            By.CSS_SELECTOR, "img[class='leaflet-marker-icon leaflet-zoom-animated leaflet-clickable']")
        camera = random.choice(clickable_cameras)

        self.driver.execute_script("arguments[0].click()", camera)

        try:
            camera_description = self.driver.find_element(By.TAG_NAME, "h2")
            if camera_description.text == "A imagem deste ponto não está disponível.":
                self.finish_process()
        except NoSuchElementException:
            pass


cctupe_crawler = CCTUPECrawler()
cctupe_crawler.run()
