import os
import logging
import time
import pandas as pd 
import argparse
import re
import datetime
from collections import Counter
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from scrap_links import init_logger,arguements


def get_table_date(
    page_content:str
)->str:
    datetime_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
    datetimes = re.findall(datetime_pattern,page_content)
    count = Counter(datetimes)
    table_date = count.most_common(1)
    if not table_date:
        return 'no_date_found'
    return  datetime.datetime.strptime(table_date[0][0], '%B %d, %Y').strftime('%Y%m%d')

def parse_link_element(
    driver:webdriver,
    id:str
)->str:
    link_element = driver.find_element_by_id("menu-dropdown-link")
    if link_element is None:
        return
    driver.execute_script("arguments[0].click();", link_element)
    form_element = driver.find_element_by_id('form-information-html')
    if form_element is None:
        return
    driver.execute_script("arguments[0].click();", form_element)
    logging.debug(driver.window_handles[-1])
    time.sleep(1)
    driver.switch_to.window(driver.window_handles[-1])
    return driver.current_url

def main()->None:
    init_logger()
    args = arguements()
    options = Options()
    options.binary_location = args.firefox_path #r"C:\Program Files\WindowsApps\Mozilla.Firefox_116.0.2.0_x64__n80bbvh6b1yt2\VFS\ProgramFiles\Firefox Package Root\firefox.exe"  # Update this with your Firefox path
    table_title = "Schedule of Investments"
    driver = webdriver.Firefox(executable_path=args.driver_path,firefox_options=options) # "geckodriver.exe"
    with open(args.url_txt,'r') as f:
        urls = [url for url in f.read().splitlines()]
        
    table_xpath = '//*[contains(text(), "{0}")]/parent::*/parent::*/following-sibling::table[1]'.format(table_title)
    for url in urls[1:]:
        # url = 'https://www.sec.gov/Archives/edgar/data/1422183/000119312511141640/d10q.htm#tx188138_6'
        # url = 'https://www.sec.gov/ix?doc=/Archives/edgar/data/0001422183/000162828023027800/fsk-20230630.htm'
        url = 'https://www.sec.gov/Archives/edgar/data/1422183/000162828023027800/fsk-20230630.htm'
        logging.info(f"ACCESSING - {url}")
        driver.get(url)
        try:
            url = parse_link_element(driver,"menu-dropdown-link")
        except Exception as e:
            logging.debug(e)
            
        logging.debug(f'PARSING - {url}')
        if url is not None:
            time.sleep(1)
            driver.get(url)
            
        html_content = driver.page_source
        table_date = get_table_date(html_content)
        logging.debug(f"DATETIMES - {table_date}")
        
        out_path = os.path.join('csv',table_date)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
            
        logging.debug(f'SAVE FILE - {url.split("/")[-1].replace(".htm","")+".html"}')
        with open(os.path.join(out_path,url.split('/')[-1].replace(".htm","")+".html"), "w",encoding='utf-8') as file:
            file.write(html_content)
        
        tables = driver.find_elements_by_xpath(table_xpath)
        if not tables:
            tables = driver.find_elements_by_xpath("(//div[span[contains(text(), 'Schedule of Investments')]]/following-sibling::table)[2]")
        logging.debug(tables)
        for i,table in enumerate(tables):
            logging.debug(f"TABLE - {table.get_attribute('outerHTML')}")
            dfs = pd.read_html(table.get_attribute('outerHTML'))
            dfs[0].to_csv(os.path.join('csv',table_date,f"{table_title.replace(' ','_')}_{i}.csv"))
        break
        time.sleep(1)
    driver.close()
    return

if __name__ == "__main__":
    main()