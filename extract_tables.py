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
from utils import arguements,init_logger

def get_xpath_elements(
    driver:webdriver,
    inline:bool,
)->list:
    xpaths = (
        '//*[contains(text(), "Schedule of Investments")]/parent::*/parent::*/following-sibling::table[1]',
        "(//div[span[contains(text(), 'Schedule of Investments')]]/parent::div/following-sibling::div/table)",
        '//font[b[contains(text(), "Schedule of Investments")]]/parent::p/following-sibling::table'
    )
    tables = driver.find_elements_by_xpath(xpaths[0])
    if not tables:
        tables = driver.find_elements_by_xpath(xpaths[1])
    logging.debug(inline)
    if not inline:
        first_table = driver.find_elements_by_xpath(xpaths[-1])
        tables.extend(first_table)
    logging.debug(f"GOT ELEMENTS  - {tables}")
    return tables

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
)->str:
    link_element = driver.find_elements_by_id("menu-dropdown-link")
    if not link_element:
        return None,False
    driver.execute_script("arguments[0].click();", link_element[0])
    form_element = driver.find_elements_by_id('form-information-html')
    if not form_element:
        return None,False
    driver.execute_script("arguments[0].click();", form_element[0])
    logging.debug(f"SWITCHING HANDLES - {driver.window_handles[-1]}")
    time.sleep(1)
    driver.switch_to.window(driver.window_handles[-1])
    return driver.current_url,True

def malformed_table(
    table:str
)->BeautifulSoup:
    soup = BeautifulSoup(table, 'lxml')
    # logging.debug(f"TABLE - {soup}")

    # If there's no <table> tag, wrap everything inside a table
    if not soup.table:
        new_table = soup.new_tag("table")
        for tag in soup.find_all(True):
            new_table.append(tag.extract())
        soup.append(new_table)
    return soup

def main()->None:
    init_logger()
    args = arguements()
    options = Options()
    options.binary_location = args.chrome_path
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path,options=options)
    table_title = "Schedule of Investments"
    with open(args.url_txt,'r') as f:
        urls = [url for url in f.read().splitlines()]

    for url in urls[1:]:
        inline = False
        # url = 'https://www.sec.gov/Archives/edgar/data/1422183/000119312511141640/d10q.htm#tx188138_6'
        # url = 'https://www.sec.gov/ix?doc=/Archives/edgar/data/0001422183/000162828023027800/fsk-20230630.htm'
        # url = 'https://www.sec.gov/Archives/edgar/data/1422183/000162828023027800/fsk-20230630.htm'
        # url = 'https://www.sec.gov/Archives/edgar/data/1422183/000119312519054647/d679678d10k.htm#tx679678_7'
        logging.info(f"ACCESSING - {url}")
        driver.get(url)
        inline_url,inline = parse_link_element(driver)
 
        logging.info(f'FINAL URL - {inline_url}')
        if inline_url is not None:
            time.sleep(1)
            driver.get(inline_url)
            
        html_content = driver.page_source
        table_date = get_table_date(html_content)
        logging.info(f"DATETIMES - {table_date}")
        
        out_path = os.path.join('csv',table_date)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
            
        logging.info(f'SAVE FILE - {url.split("/")[-1].replace(".htm","")+".html"}')
        with open(os.path.join(out_path,url.split('/')[-1].replace(".htm","")+".html"), "w",encoding='utf-8') as file:
            file.write(html_content)
        
        tables = get_xpath_elements(driver,inline)
        if not tables or not tables[0]:
            continue
        for i,table in enumerate(tables):
            if os.path.exists(os.path.join('csv',table_date,f"{table_title.replace(' ','_')}_{i}.csv")):
                continue
            # logging.debug(f"TABLE - {type(table)} {table}")
            table = malformed_table(table.get_attribute("outerHTML"))
            dfs = pd.read_html(table.prettify(),displayed_only=False)
            if not dfs:
                logging.debug(f"NO TABLES - {dfs}")
                continue
            dfs[0].to_csv(os.path.join('csv',table_date,f"{table_title.replace(' ','_')}_{i}.csv"))
        time.sleep(1)
    driver.close()
    return

if __name__ == "__main__":
    main()