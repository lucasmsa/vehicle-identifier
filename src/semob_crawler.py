from selenium import webdriver
from config.constants import URL
import chromedriver_autoinstaller
from aws_operations import AwsOperations
from selenium.webdriver.common.by import By


class SemobCrawler:
    def __init__(self):
        self.aws_operations = AwsOperations()
        self.aws_operations.upload_file('assets/car.jpg')

    def setup_webdriver(self):
        chromedriver_autoinstaller.install()
        chrome_options = webdriver.ChromeOptions()
        rules = ["--no-sandbox", "--disable-infobars",
                 "--disable-dev-shm-usage"]

        for rule in rules:
            chrome_options.add_argument(rule)

        self.driver = webdriver.Chrome(options=chrome_options)

    def check_service_availability(self):
        self.driver.get(URL)
        header = self.driver.find_element(By.TAG_NAME, "h1")
        if header.text == "SERVIÇO DESATIVADO":
            return self.finish_process()

    def run(self):
        self.check_service_availability()

    def finish_process(self):
        self.driver.close()
        return "DRIVER CLOSED"


semob_crawler = SemobCrawler()
