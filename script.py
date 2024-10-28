from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options

options = Options()
options.headless = True

#service = FirefoxService(executable_path='/usr/local/bin/geckodriver')
driver = webdriver.Firefox()#service=service, options=options)

driver.get('https://www.google.com')
driver.save_screenshot('screenshot.png')
driver.quit()

