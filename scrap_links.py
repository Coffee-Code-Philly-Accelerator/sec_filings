import os
import logging
import time
import pandas as pd 
import argparse
from rich.logging import RichHandler
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

from important_tables import key_fields


def arguements()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Get filing links and dates')
    parser.add_argument(
        '--url', type=str, required=False, 
        default='https://www.sec.gov/edgar/browse/?CIK=1422183',
        help='Sec url to get links from'
    )
    parser.add_argument(
        '--url_txt', type=str, required=False, 
        default='urls/1422183.txt',
        help='.txt to get links from'
    )
    parser.add_argument(
        '--driver-path', type=str, required=False, 
        default="geckodriver.exe",
        help='path to your geckodriver.exe'
    )
    parser.add_argument(
        '--firefox-path', type=str, required=False, 
        default=r"C:\Program Files\WindowsApps\Mozilla.Firefox_116.0.3.0_x64__n80bbvh6b1yt2\VFS\ProgramFiles\Firefox Package Root\firefox.exe",
        help='path to your firefox.exe'
    )
    return parser.parse_args()

def init_logger() -> None:
    logger = logging.getLogger("rich")
    logger.setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)

    FORMAT = "%(name)s[%(process)d] " + \
        "%(processName)s(%(threadName)s) " + \
        "%(module)s:%(lineno)d  %(message)s"

    formatter = logging.Formatter(
        FORMAT,
        datefmt="%Y%m%d %H:%M:%S"
    )
    logging.basicConfig(
        level="NOTSET", format=FORMAT, handlers=[RichHandler()]
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    logging.info("Initializing ok.")

def main()->None:
    init_logger()
    args = arguements()
    url = args.url
    options = Options()
    options.binary_location = args.firefox_path #r"C:\Program Files\WindowsApps\Mozilla.Firefox_116.0.2.0_x64__n80bbvh6b1yt2\VFS\ProgramFiles\Firefox Package Root\firefox.exe"  # Update this with your Firefox path
    driver = webdriver.Firefox(executable_path=args.driver_path,firefox_options=options)#"geckodriver.exe")
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

    table = driver.find_element_by_id('filingsTable')
    soup = BeautifulSoup(table.get_attribute('innerHTML'))
    with open(os.path.join('urls',url.split("=")[-1]+".txt"),'w') as url_out:
        for a in soup.find_all('a', href=True):
            url_out.write('\n%s' % url.split('edgar')[0]+a['href'])
            logging.debug('\n%s' % url.split('edgar')[0]+a['href'])

    driver.close()


if __name__ == '__main__':
    main()