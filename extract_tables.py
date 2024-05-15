import os
import warnings
import logging
import time
import pandas as pd 
import re
import platform
import datetime
from collections import Counter
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import arguements,init_logger,ROOT_PATH



"""
document.querySelectorAll("body > document:nth-child(1) > type:nth-child(1) > sequence:nth-child(1) > filename:nth-child(1) > description:nth-child(1) > text:nth-child(1) > div:nth-child(418) ~ div"
"""
def test_xpath_elements(
    url:str,
    xpath:str
)->list:
    args = arguements()
    options = Options()
    options.binary_location = args.chrome_path
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path)\
        if platform.system() == "Linux" else webdriver.Chrome()
    driver.get(url)
    tables = driver.find_elements(By.XPATH,value=xpath)
    tables = sorted(tables, key=lambda table: table.location['y'])
    
    if not os.path.exists(os.path.join(ROOT_PATH,args.cik,'test_path')):
        os.mkdir(os.path.join(ROOT_PATH,args.cik,'test_path'))

    for i,table in enumerate(tables):
        table = malformed_table(table.get_attribute("outerHTML"))
        dfs = pd.read_html(table.prettify(),displayed_only=False)
        for df in dfs:
            df.to_csv(os.path.join(ROOT_PATH,args.cik,'test_path',f"test_{i}.csv"))
    driver.close()
    return tables

def get_xpath_elements(
    driver:webdriver,
    xpaths:list,
)->list:
    tables = []
    for path in xpaths:
        tables.extend(driver.find_elements(By.XPATH,value=path))
    # driver.execute('document.querySelectorAll("body > document:nth-child(1) > type:nth-child(1) > sequence:nth-child(1) > filename:nth-child(1) > description:nth-child(1) > text:nth-child(1) > div:nth-child(418) ~ div"')
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
    link_element = driver.find_elements(By.ID,value="menu-dropdown-link")
    if not link_element:
        return None
    driver.execute_script("arguments[0].click();", link_element[0]) 
    form_element = driver.find_elements(By.ID,value='form-information-html')
    if not form_element:
        return None
    driver.execute_script("arguments[0].click();", form_element[0])
    logging.debug(f"SWITCHING HANDLES - {driver.window_handles[-1]}")
    time.sleep(1)
    driver.switch_to.window(driver.window_handles[-1])
    return driver.current_url

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

def remove_duplicate_element(
    elements:webdriver.remote.webelement.WebElement
)->list:
    ids = set()
    unique_elements = []
    for e in elements:
        if e.id not in ids:
            ids.add(e.id)
            unique_elements.append(e)
    return unique_elements
    

def main()->None:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    init_logger()
    args = arguements()
    options = Options()
    options.binary_location = args.chrome_path
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path) \
        if platform.system() == "Linux" else webdriver.Chrome()
    table_title = "Schedule of Investments"
    with open(os.path.join(ROOT_PATH,args.url_txt),'r') as f:
        urls = [(*url.split(' '),) for url in f.read().splitlines()]
        
    with open(args.x_path) as file:
        gen_paths = [line.rstrip() for line in file.readlines()]
    for table_date,url in urls[1:]:
        # table_date,url = '2017-06-30' ,'https://www.sec.gov/Archives/edgar/data/0001379785/000137978517000060/a2017063010qtcap.htm'

        logging.info(f"ACCESSING - {url}")
        driver.get(url)
        inline_url = parse_link_element(driver)
 
        logging.info(f'FINAL URL - {inline_url}')
        if inline_url is not None:
            time.sleep(1)
            driver.get(inline_url)
            
        html_content = driver.page_source
        logging.info(f"DATETIMES - {table_date}")
        
        out_path,spec_path = os.path.join(ROOT_PATH,args.cik,table_date),os.path.join(ROOT_PATH,args.cik,table_date,'spec_paths.txt')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
            
        logging.info(f'SAVE FILE - {url.split("/")[-1].replace(".htm","")+".html"}')
        html_to_file = os.path.join(ROOT_PATH,out_path,url.split('/')[-1].replace(".htm","")+".html")
        with open(html_to_file, "w",encoding='utf-8') as file:
            file.write(BeautifulSoup(html_content,'html.parser').prettify())
        
        xpaths = gen_paths
        tables = get_xpath_elements(driver,xpaths)
        if os.path.exists(spec_path):
            with open(spec_path) as file:
                xpaths = [line.rstrip() for line in file.readlines()]
                # spec_paths.extend(xpaths)
            tables = get_xpath_elements(driver,xpaths)
        logging.debug(f"USING XPATHS - {xpaths}")

        tables = sorted(tables, key=lambda table: table.location['y'])
        tables = remove_duplicate_element(tables)      
        for i,table in enumerate(tables):
            if os.path.exists(os.path.join(ROOT_PATH,args.cik,table_date,f"{table_title.replace(' ','_')}_{i}.csv")):
                continue
            table = malformed_table(table.get_attribute("outerHTML"))
            dfs = pd.read_html(table.prettify(),displayed_only=False)
            if not dfs:
                logging.debug(f"NO TABLES - {dfs}")
                continue
            dfs[0].to_csv(os.path.join(ROOT_PATH,args.cik,table_date,f"{table_title.replace(' ','_')}_{i}.csv"),encoding='utf-8')
        # break
    driver.close()
    return

if __name__ == "__main__":
    """
    python .\extract_tables.py --cik 1501729 --url-txt urls/1501729.txt --x-path xpaths/1501729.txt   
    python .\extract_tables.py --cik 1396440 --url-txt urls/1396440.txt --x-path xpaths/1396440.txt
    python .\extract_tables.py --cik 1422183 --url-txt urls/1422183.txt --x-path xpaths/1422183.txt
    python .\extract_tables.py --cik 1490349 --url-txt urls/1490349.txt --x-path xpaths/1490349.txt
    python .\extract_tables.py --cik 1379785 --url-txt urls/1379785.txt --x-path xpaths/1379785.txt
    python .\extract_tables.py --cik 1490927 --url-txt urls/1490927.txt --x-path xpaths/1490927.txt

    /html/body/document/type/sequence/filename/description/text/div[11]/div/table
    /html/body/document/type/sequence/filename/description/text/div[48]/div/table
    /html/body/document/type/sequence/filename/description/text/div[16]/div/table
    /html/body/document/type/sequence/filename/description/text/div[15]/div/table
    /html/body/document/type/sequence/filename/description/text/div[19]/div/table
    
    
    1501729
    /html/body/document/type/sequence/filename/description/text/div[48]/div/table
    /html/body/document/type/sequence/filename/description/text/div[15]/div/table
    /html/body/document/type/sequence/filename/description/text/div[19]/div/table
    
    1396440
    //br[contains(text(), "Schedule of Investments")]/parent::b/parent::font/parent::p/following-sibling::div/child::div/child::table
    //b[contains(text(), "Schedule of Investments")]/parent::font/parent::p/following-sibling::div/child::div/child::table
    //b[contains(text(), "Schedule of Investments")]/parent::p/parent::div/following-sibling::div/child::div/child::table
    //b[contains(text(), "Schedule of Investments")]/parent::p/parent::div/following-sibling::div/child::table
    
    1490349
    //b[contains(text(), "Schedule of Investments")]/parent::p/following-sibling::table
    """
    main()
    # test_xpath_elements(
    #     url='https://www.sec.gov/Archives/edgar/data/0001501729/000110465921102950/tm2124358-1_10q.htm',
    #     xpath='//div[font[contains(text(), "Schedule of Investments")]]/parent::div/parent::font/following-sibling::table'
    # )