from selenium import webdriver
from selenium.webdriver.firefox.options import Options

options = Options()
options.headless = True
driver = webdriver.Firefox(options=options)

driver.get('google.com')
driver.save_screenshot('screenshot.png')
driver.quit()

