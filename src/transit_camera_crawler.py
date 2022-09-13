import chromedriver_autoinstaller
from selenium import webdriver


class TransitCameraCrawler:
    def setup_webdriver(self):
        chromedriver_autoinstaller.install()
        chrome_options = webdriver.ChromeOptions()
        rules = ["--no-sandbox", "--disable-infobars",
                 "--disable-dev-shm-usage", "--headless"]

        for rule in rules:
            chrome_options.add_argument(rule)

        self.driver = webdriver.Chrome(options=chrome_options)

    def finish_process(self):
        self.driver.quit()
        return "DRIVER CLOSED"
