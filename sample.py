from selenium import webdriver

driver = webdriver.Firefox(executable_path="geckodriver.exe")
driver.get('https://www.sec.gov/edgar/browse/?CIK=1540855')
for a in driver.find_elements_by_xpath('.//span[contains(text(), "Author")]/a'):
    print(a.get_attribute('href'))