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

def test_xpath_elements(
    url:str,
    xpath:str
)->list:
    args = arguements()
    options = Options()
    options.binary_location = args.chrome_path
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path)
    driver.get(url)
    tables = driver.find_elements_by_xpath(xpath)
    tables = sorted(tables, key=lambda table: table.location['y'])
    
    if not os.path.exists(os.path.join('csv','test_path')):
        os.mkdir(os.path.join('csv','test_path'))
    print(tables)
    for i,table in enumerate(tables):
        table = malformed_table(table.get_attribute("outerHTML"))
        dfs = pd.read_html(table.prettify(),displayed_only=False)
        for df in dfs:
            df.to_csv(os.path.join('csv','test_path',f"test_{i}.csv"))
    driver.close()
    return tables

def get_xpath_elements(
    driver:webdriver,
    inline:bool,
)->list:
    """
    var xpathExpression = "//font[contains(text(), 'Schedule of Investments')]/parent::div/parent::div/following-sibling::div/table";
    var xpathResult = document.evaluate(xpathExpression, document, null, XPathResult.ORDERED_NODE_ITERATOR_TYPE, null);

    var node = xpathResult.iterateNext();
    while(node) {
        console.log(node);
        node = xpathResult.iterateNext();
    }
    """
    xpaths = (
        "//div[span[contains(text(), 'Schedule of Investments')]]/parent::div/following-sibling::div/table",
        "//font[contains(text(), 'Schedule of Investments')]/parent::div/parent::div/following-sibling::div/table",
        "//b[contains(text(), 'Schedule of Investments')]/parent::p/following-sibling::table",
        '//font[b[contains(text(), "Schedule of Investments")]]/parent::p/following-sibling::table',
        '//b[contains(text(), "As of December 31, 2018") ]/parent::font/parent::p/following-sibling::table'
    )
    tables = []
    logging.debug(inline)
    if not inline:
        first_table = driver.find_elements_by_xpath(xpaths[-1])
        tables.extend(first_table)
        
    for path in xpaths[:-1]:
        tables.extend(driver.find_elements_by_xpath(path))
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
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path)#,options=options)
    table_title = "Schedule of Investments"
    with open(args.url_txt,'r') as f:
        urls = [(*url.split(' '),) for url in f.read().splitlines()]

    for table_date,url in urls[1:]:
        # table_date,url = '2011-03-31', 'https://www.sec.gov/Archives/edgar/data/0001422183/000119312511141640/d10q.htm'
        inline = False
        logging.info(f"ACCESSING - {url}")
        driver.get(url)
        inline_url,inline = parse_link_element(driver)
 
        logging.info(f'FINAL URL - {inline_url}')
        if inline_url is not None:
            time.sleep(1)
            driver.get(inline_url)
            
        html_content = driver.page_source
        logging.info(f"DATETIMES - {table_date}")
        
        out_path = os.path.join('csv',table_date)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
            
        logging.info(f'SAVE FILE - {url.split("/")[-1].replace(".htm","")+".html"}')
        html_to_file = os.path.join(out_path,url.split('/')[-1].replace(".htm","")+".html")
        with open(html_to_file, "w",encoding='utf-8') as file:
            file.write(BeautifulSoup(html_content,'html.parser').prettify())
        
        tables = get_xpath_elements(driver,inline)
        tables = sorted(tables, key=lambda table: table.location['y'])
        for i,table in enumerate(tables):
            if os.path.exists(os.path.join('csv',table_date,f"{table_title.replace(' ','_')}_{i}.csv")):
                continue
            table = malformed_table(table.get_attribute("outerHTML"))
            dfs = pd.read_html(table.prettify(),displayed_only=False)
            if not dfs:
                logging.debug(f"NO TABLES - {dfs}")
                continue
            dfs[0].to_csv(os.path.join('csv',table_date,f"{table_title.replace(' ','_')}_{i}.csv"))
        # time.sleep(1)
    driver.close()
    return

if __name__ == "__main__":
    main()
    # test_xpath_elements(
    #     url='https://www.sec.gov/Archives/edgar/data/0001422183/000119312519054647/d679678d10k.htm',
    #     xpath='//b[contains(text(), "As of December 31, 2018") ]/parent::font/parent::p/following-sibling::table'
    # )