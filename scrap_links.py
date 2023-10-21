import os
import logging
import time
import pandas as pd 
import platform
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options

from utils import arguements,init_logger,ROOT_PATH


def main()->None:
    init_logger()
    args = arguements()
    url = args.url
    options = Options()
    options.binary_location = args.chrome_path
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path) \
        if platform.system() == "Linux" else webdriver.Chrome(options=options)
    driver.get(url)
    html_content = driver.page_source
    if not os.path.exists(os.path.join(ROOT_PATH,'htmls')):
        os.mkdir(os.path.join(ROOT_PATH,'htmls'))
    with open(os.path.join(ROOT_PATH,'htmls',url.split("=")[-1]+".html"), "w",encoding='utf-8') as file:
        file.write(html_content)
    dfs = pd.read_html(html_content)
    
    if not os.path.exists(os.path.join(ROOT_PATH,'csv')):
        os.mkdir(os.path.join(ROOT_PATH,'csv'))
    for i,df in enumerate(dfs):
        df.to_csv(os.path.join(ROOT_PATH,'csv',url.split("=")[-1]+f"_link_table_{i}.csv"))
    h5_tags = driver.find_elements(By.TAG_NAME,value='h5')

    for h5_tag in h5_tags:
        if h5_tag.text == "[+] 10-K (annual reports) and 10-Q (quarterly reports)":
            # Click on the h5 tag.
            h5_tag.click()
            break
        
    xpath = '//button[text()="View all 10-Ks and 10-Qs"]'
    element = WebDriverWait(driver,3).until(EC.element_to_be_clickable((By.XPATH,xpath)))
    driver.execute_script("arguments[0].click();", element)

    conditions = '@data-original-title="Open document" and contains(@href, "Archive") and not(contains(@href, "index")) and not(contains(@href, "xml"))'
    table = driver.find_elements(By.CSS_SELECTOR,value='div.dataTables_scroll')
    links = table[0].find_elements(By.XPATH,value=f'//td//a[{conditions}]')
    logging.debug(f"LINKS - {len([link.get_attribute('innerHTML') for link in links])}")
    df = pd.read_html(table[0].get_attribute('innerHTML'))[-1]
    filing_date = df['Reporting date']
    logging.debug(f"DATES - {len(filing_date)}")
    with open(os.path.join(ROOT_PATH,'urls',url.split("=")[-1]+".txt"),'w') as url_out:
        for a,date in zip(links,filing_date):
            url_out.write('\n%s %s' % (date.split("View")[0],a.get_attribute('href')))
            logging.debug('\n%s %s' % (date.split("View")[0],a.get_attribute('href')))

    driver.close()

if __name__ == '__main__':
    """
    docker run -v $(pwd)/sec_filings/run.sh:/script.sh ubuntu:latest /run.sh
    python3 scrap_links --url [url] --chrome_path [chrome_path] --chrome_driver_path [chrome_driver_path]
    """
    main()