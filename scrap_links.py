import os
import logging
import time
import pandas as pd 
import argparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options

from utils import arguements,init_logger


def main()->None:
    init_logger()
    args = arguements()
    url = args.url
    options = Options()
    options.binary_location = args.chrome_path
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path,options=options)
    # options.binary_location = args.firefox_path #r"C:\Program Files\WindowsApps\Mozilla.Firefox_116.0.2.0_x64__n80bbvh6b1yt2\VFS\ProgramFiles\Firefox Package Root\firefox.exe"  # Update this with your Firefox path
    # driver = webdriver.Firefox(executable_path=args.driver_path,firefox_options=options)#"geckodriver.exe")
    driver.get(url)
    h5_tags = driver.find_elements_by_tag_name("h5")

    for h5_tag in h5_tags:
        if h5_tag.text == "[+] 10-K (annual reports) and 10-Q (quarterly reports)":
            # Click on the h5 tag.
            h5_tag.click()
            break
        
    xpath = '//button[text()="View all 10-Ks and 10-Qs"]'
    element = WebDriverWait(driver,3).until(EC.element_to_be_clickable((By.XPATH,xpath)))
    driver.execute_script("arguments[0].click();", element)
    
    
    html_content = driver.page_source
    with open(os.path.join('htmls',url.split("=")[-1]+".html"), "w",encoding='utf-8') as file:
        file.write(html_content)
    dfs = pd.read_html(html_content)
    for i,df in enumerate(dfs):
        df.to_csv(os.path.join('csv',url.split("=")[-1]+f"_link_table_{i}.csv"))

    conditions = '@data-original-title="Open document" and contains(@href, "Archive") and not(contains(@href, "index")) and not(contains(@href, "xml"))'
    table = driver.find_elements_by_css_selector('div.dataTables_scroll')
    links = table[0].find_elements_by_xpath(f'//td//a[{conditions}]')
    logging.debug(f"LINKS - {len([link.get_attribute('innerHTML') for link in links])}")
    df = pd.read_html(table[0].get_attribute('innerHTML'))[-1]
    filing_date = df['Reporting date']
    logging.debug(f"DATES - {len(filing_date)}")
    with open(os.path.join('urls',url.split("=")[-1]+".txt"),'w') as url_out:
        for a,date in zip(links,filing_date):
            url_out.write('\n%s %s' % (date.split("View")[0],a.get_attribute('href')))
            logging.debug('\n%s %s' % (date.split("View")[0],a.get_attribute('href')))

    driver.close()


if __name__ == '__main__':
    main()