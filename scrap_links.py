import os
import pandas as pd
import platform
import csv
import logging
# import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from utils import arguements, init_logger, ROOT_PATH

def search(
    driver:webdriver,
    search_string:str
)->None:
    # Find the form element by ID
    input_element = driver.find_elements(By.ID, value='searchbox')[0]
    input_element.clear()
    # Enter text into the form
    input_element.send_keys(search_string)
    input_element.send_keys(Keys.RETURN)  # This simulates pressing Enter

def read_table(
    driver:webdriver,
    cik:str,
    url:str,
)->None:
    if not cik == '3906' and os.path.exists(os.path.join(ROOT_PATH, 'urls', url.split("=")[-1]+".csv")):
        return
    logger = init_logger(cik)

    conditions = '@data-original-title="Open document" and contains(@href, "Archive") and not(contains(@href, "index")) and not(contains(@href, "xml"))'
    table = driver.find_elements(
        By.CSS_SELECTOR, value='div.dataTables_scroll')

    links = table[0].find_elements(By.XPATH, value=f'//td//a[{conditions}]')
    logger.debug(
        f"LINKS - {len([link.get_attribute('innerHTML') for link in links])}")
    df = pd.read_html(table[0].get_attribute('innerHTML'))[-1]
    reporting_date, filing_date = df['Reporting date'], df['Filing date']
    logger.debug(f"DATES - {len(filing_date)}")

    headers = ['reporting_date', 'date_filed', 'html_link'] if not os.path.exists(os.path.join(ROOT_PATH, 'urls', url.split("=")[-1]+".csv")) else ''
    with open(os.path.join(ROOT_PATH, 'urls', url.split("=")[-1]+".csv"), 'a', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(headers)
        for a, fd, rd in zip(links, filing_date, reporting_date):
            writer.writerow([fd.split("View")[0], rd.split(
                "View")[0], a.get_attribute('href')])
            logger.info('\n%s %s %s' % (rd.split("View")[
                        0], fd.split("View")[0], a.get_attribute('href')))

def clear_form(
    driver:webdriver,
)->None:
    _from = driver.find_elements(By.ID, value='filingDateFrom')
    _to = driver.find_elements(By.ID, value='filingDateTo')
    driver.implicitly_wait(5)
    if _from[0].is_displayed() and _to[0].is_displayed():
        _from[0].clear()
        _to[0].clear()
        _from[0].send_keys("")
        _to[0].send_keys("")
        

def main() -> None:
    args = arguements()

    url = args.url
    options = Options()
    options.binary_location = args.chrome_path
    options.add_argument("--log-level=OFF")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")  # Bypass OS security model, required in some environments
    options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    service = Service(executable_path='/usr/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)
    if not os.path.exists(f'xpaths/{args.cik}.txt'):
        with open(f'xpaths/{args.cik}.txt', 'w') as file:
            file.write("")
    driver.get(url)
    html_content = driver.page_source
    if not os.path.exists(os.path.join(ROOT_PATH, 'htmls')):
        os.mkdir(os.path.join(ROOT_PATH, 'htmls'))
    with open(os.path.join(ROOT_PATH, 'htmls', url.split("=")[-1]+".html"), "w", encoding='utf-8') as file:
        file.write(html_content)
    dfs = pd.read_html(html_content)

    if not os.path.exists(os.path.join(ROOT_PATH, args.cik)):
        os.mkdir(os.path.join(ROOT_PATH, args.cik))
    for i, df in enumerate(dfs):
        df.to_csv(os.path.join(ROOT_PATH, args.cik,url.split("=")[-1]+f"_link_table_{i}.csv"))
    h5_tags = driver.find_elements(By.TAG_NAME, value='h5')

    if args.cik == '3906':
        clear_form(driver)
        search(driver,search_string='10-K')
        read_table(driver, args.cik,url)
        search(driver,search_string='10-Q')
        read_table(driver, args.cik,url)
        driver.close()
        return 
    
    for h5_tag in h5_tags:
        if h5_tag.text == "[+] 10-K (annual reports) and 10-Q (quarterly reports)":
            # Click on the h5 tag.
            h5_tag.click()
            break

    xpath = '//button[text()="View all 10-Ks and 10-Qs"]'
    element = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, xpath)))
    driver.execute_script("arguments[0].click();", element)
    clear_form(driver)
    read_table(driver,args.cik,url)
    

    driver.close()

if __name__ == '__main__':
    """
    docker run -v $(pwd)/sec_filings/run.sh:/script.sh ubuntu:latest /run.sh
    python3 scrap_links --url [url] --chrome_path [chrome_path] --chrome_driver_path [chrome_driver_path]
    python .\scrap_links.py --cik 1396440 --url https://www.sec.gov/edgar/browse/?CIK=1396440
    python .\scrap_links.py --cik 1490927 --url https://www.sec.gov/edgar/browse/?CIK=1490927
    python .\scrap_links.py --cik 1490349 --url https://www.sec.gov/edgar/browse/?CIK=1490349
    python .\scrap_links.py --cik 1379785 --url https://www.sec.gov/edgar/browse/?CIK=1379785
    python .\scrap_links.py --cik 1418076 --url https://www.sec.gov/edgar/browse/?CIK=1418076
    python .\scrap_links.py --cik 1544206 --url https://www.sec.gov/edgar/browse/?CIK=1544206
    python .\scrap_links.py --cik 1370755 --url https://www.sec.gov/edgar/browse/?CIK=1370755
    python3 scrap_links.py --cik 1326003 --url https://www.sec.gov/edgar/browse/?CIK=1326003
    python .\scrap_links.py --cik 1580345 --url https://www.sec.gov/edgar/browse/?CIK=1580345

    python .\scrap_links.py --cik 1535778 --url https://www.sec.gov/edgar/browse/?CIK=1535778
    python .\scrap_links.py --cik 1487918 --url https://www.sec.gov/edgar/browse/?CIK=1487918
    python3 scrap_links.py --cik 1512931 --url https://www.sec.gov/edgar/browse/?CIK=1512931 

    python3 scrap_links.py --cik 1372807 --url https://www.sec.gov/edgar/browse/?CIK=1372807 
    python3 scrap_links.py --cik 1675033 --url https://www.sec.gov/edgar/browse/?CIK=1675033
    
    python3 scrap_links.py --cik 3906 --url https://www.sec.gov/edgar/browse/?CIK=3906

    """
    main()
