from config.url_constants import SEMOB_URL
from aws_operations import AwsOperations
from selenium.webdriver.common.by import By
from transit_camera_crawler import TransitCameraCrawler


class SemobCrawler(TransitCameraCrawler):
    def __init__(self):
        self.setup_webdriver()
        self.aws_operations = AwsOperations()
        self.aws_operations.upload_file('assets/car.jpg')

    def check_service_availability(self):
        self.driver.get(SEMOB_URL)
        header = self.driver.find_element(By.TAG_NAME, "h1")
        if header.text == "SERVIÇO DESATIVADO":
            return self.finish_process()

    def run(self):
        self.check_service_availability()


semob_crawler = SemobCrawler()
semob_crawler.check_service_availability()
