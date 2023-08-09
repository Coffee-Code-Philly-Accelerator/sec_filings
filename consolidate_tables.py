import os
import logging
import time
import pandas as pd 
import argparse
import re
from collections import Counter
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


def get_table_date(
    page_content:str
)->str:
    datetime_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
    datetimes = re.findall(datetime_pattern,page_content)
    count = Counter(datetimes)
    table_date = count.most_common(1)
    if not table_date:
        return 'no date found'
    return table_date

def main()->None:
    init_logger()
    options = Options()
    options.binary_location = r"C:\Program Files\WindowsApps\Mozilla.Firefox_116.0.2.0_x64__n80bbvh6b1yt2\VFS\ProgramFiles\Firefox Package Root\firefox.exe"  # Update this with your Firefox path
    table_title = "Schedule of Investments"
    args = arguements()
    driver = webdriver.Firefox(executable_path="geckodriver.exe",firefox_options=options)
    with open(args.url_txt,'r') as f:
        urls = [url for url in f.read().splitlines()]
        
    table_xpath = '//*[contains(text(), "{0}")]/parent::*/parent::*/following-sibling::table[1]'.format(table_title)
    for url in urls[1:]:
        # url = 'https://www.sec.gov/Archives/edgar/data/1422183/000119312511141640/d10q.htm#tx188138_6'
        logging.info(f"ACCESSING - {url}")
        driver.get(url)
        html_content = driver.page_source
        with open(os.path.join('htmls',url.split("/")[-1].replace(".htm","")+".html"), "w",encoding='utf-8') as file:
            file.write(html_content)
            
        table_date = get_table_date(html_content)
        logging.debug(f"DATETIMES - {table_date}")
        
        tables = driver.find_elements_by_xpath(table_xpath)
        logging.debug(tables)
        for i,table in enumerate(tables):
            dfs = pd.read_html(table.get_attribute('outerHTML'))
            dfs[0]['date'] = table_date
            dfs[0].to_csv(os.path.join('csv',f"{table_title.replace(' ','_')}_{url.split('/')[-1].replace('.htm','')}_{i}.csv"))
        time.sleep(1)
        break
    driver.close()
    return

if __name__ == "__main__":
    main()