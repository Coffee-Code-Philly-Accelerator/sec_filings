import os
import logging
import time
import pandas as pd 
import argparse
from rich.logging import RichHandler
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from scrap_links import init_logger

def arguements()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Get filing links and dates')
    parser.add_argument(
        '--url_txt', type=str, required=False, 
        default='urls/1422183.txt',
        help='Sec url to get links from'
    )

    return parser.parse_args()

def main()->None:
    table_title = "Schedule of Investments"
    init_logger()
    args = arguements()
    driver = webdriver.Firefox(executable_path="geckodriver.exe")
    with open(args.url_txt,'r') as f:
        urls = [url for url in f.read().splitlines()]
        
    table_xpath = '//*[contains(text(), "{0}")]/parent::*/parent::*/following-sibling::table[1]'.format(table_title)
    for url in urls[1:]:
        # url = 'https://www.sec.gov/Archives/edgar/data/1422183/000119312511141640/d10q.htm#tx188138_6'
        logging.info(f"ACCESSING - {url}")
        driver.get(url)
        tables = driver.find_elements_by_xpath(table_xpath)
        logging.debug(tables)
        for i,table in enumerate(tables):
            dfs = pd.read_html(table.get_attribute('outerHTML'))
            dfs[0].to_csv(os.path.join('csv',f"{table_title.replace(' ','_')}_{url.split('/')[-1].replace('.htm','')}_{i}.csv"))
        # break
        time.sleep(1)
    driver.close()
    return

if __name__ == "__main__":
    main()