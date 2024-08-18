import os
import warnings
import time
import pandas as pd
import re
import platform
import datetime
from collections import Counter
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import arguements, init_logger, ROOT_PATH


"""
document.querySelectorAll("body > document:nth-child(1) > type:nth-child(1) > sequence:nth-child(1) > filename:nth-child(1) > description:nth-child(1) > text:nth-child(1) > div:nth-child(418) ~ div"
"""


def test_xpath_elements(
    url: str,
    xpath: str
) -> list:
    args = arguements()
    options = Options()
    # options.binary_location = args.chrome_path
    options.set_capability('goog:loggerPrefs', {'browser': 'ALL'})
    options.add_argument("--verbose")
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path, options=options) if platform.system(
    ) == "Linux" else webdriver.Chrome(options=options)
    driver.get(url)
    tables = driver.find_elements(By.XPATH, value=xpath)
    tables = sorted(tables, key=lambda table: table.location['y'])

    if not os.path.exists(os.path.join(ROOT_PATH, args.cik, 'test_path')):
        os.mkdir(os.path.join(ROOT_PATH, args.cik, 'test_path'))

    for i, table in enumerate(tables):
        table = malformed_table(table.get_attribute("outerHTML"))
        dfs = pd.read_html(table.prettify(), displayed_only=False)
        for df in dfs:
            df.to_csv(os.path.join(ROOT_PATH, args.cik,
                      'test_path', f"test_{i}.csv"))
    driver.close()
    return tables


def get_xpath_elements(
    driver: webdriver,
    xpaths: list,
) -> list:
    tables = []
    for path in xpaths:
        tables.extend(driver.find_elements(By.XPATH, value=path))
    # driver.execute('document.querySelectorAll("body > document:nth-child(1) > type:nth-child(1) > sequence:nth-child(1) > filename:nth-child(1) > description:nth-child(1) > text:nth-child(1) > div:nth-child(418) ~ div"')
    logger.debug(f"GOT ELEMENTS  - {tables}")
    return tables


def get_table_date(
    page_content: str
) -> str:
    datetime_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
    datetimes = re.findall(datetime_pattern, page_content)
    count = Counter(datetimes)
    table_date = count.most_common(1)
    if not table_date:
        return 'no_date_found'
    return datetime.datetime.strptime(table_date[0][0], '%B %d, %Y').strftime('%Y%m%d')


def parse_link_element(
    driver: webdriver,
) -> str:
    iframe = driver.find_elements(By.CSS_SELECTOR, value='#ixvFrame')
    if iframe:
        time.sleep(1)
        logger.debug(f"IFRAME - {iframe[0]}")
        driver.switch_to.frame(iframe[0])
    link_element = driver.find_elements(By.ID, value="menu-dropdown-link")
    logger.debug(f"LINK ELEMENT - {link_element}")
    if not link_element:
        return None
    driver.execute_script("arguments[0].click();", link_element[0])
    form_element = driver.find_elements(By.ID, value='form-information-html')
    logger.debug(f"FORM ELEMENT - {form_element}")
    if not form_element:
        return None
    driver.execute_script("arguments[0].click();", form_element[0])
    logger.debug(f"SWITCHING HANDLES - {driver.window_handles[-1]}")
    time.sleep(1)
    driver.switch_to.window(driver.window_handles[-1])
    return driver.current_url


def malformed_table(
    table: str
) -> BeautifulSoup:
    soup = BeautifulSoup(table, 'lxml')
    # If there's no <table> tag, wrap everything inside a table
    if not soup.table:
        new_table = soup.new_tag("table")
        for tag in soup.find_all(True):
            new_table.append(tag.extract())
        soup.append(new_table)
    return soup


def remove_duplicate_element(
    elements: webdriver.remote.webelement.WebElement
) -> list:
    ids = set()
    unique_elements = []
    for e in elements:
        if e.id not in ids:
            ids.add(e.id)
            unique_elements.append(e)
    return unique_elements


def main() -> None:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # desired_dpi = 2.0
    options = Options()
    options.add_argument("--no-sandbox")
    # options.add_argument("--headless")
    # options.add_experimental_option('excludeSwitches', ['enable-logging'])
    # options.add_argument(f"--force-device-scale-factor={desired_dpi}")
    # options.add_experimental_option("mobileEmulation", {"deviceMetrics": {"width": 1920, "height": 1080, "pixelRatio": 3.0}})
    driver = webdriver.Chrome(executable_path=args.chrome_driver_path, options=options) if platform.system(
    ) == "Linux" else webdriver.Chrome(options=options)
    # driver.set_window_size(1920, 1080)
    table_title = "Schedule of Investments"

    urls = pd.read_csv(os.path.join(ROOT_PATH, args.url_csv), index_col=False)
    with open(args.x_path) as file:
        gen_paths = [line.rstrip() for line in file.readlines()]

    for i in range(urls.shape[0]):
        _, table_date, url = urls.iloc[i]
        table_date, url = '2018-12-31', 'https://www.sec.gov/Archives/edgar/data/0001512931/000114420419012276/tv514438_10k.htm'

        logger.info(f"DATETIMES - {table_date}")
        logger.info(f"ACCESSING - {url}")
        driver.get(url)
        inline_url = parse_link_element(driver)

        if inline_url is not None:
            logger.info(f'FINAL URL - {inline_url}')
            time.sleep(2)
            driver.get(inline_url)

        html_content = driver.page_source

        out_path, spec_path = os.path.join(ROOT_PATH, args.cik, table_date), os.path.join(
            ROOT_PATH, args.cik, table_date, 'spec_paths.txt')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        logger.info(
            f'SAVE FILE - {url.split("/")[-1].replace(".htm","")+".html"}')
        html_to_file = os.path.join(ROOT_PATH, out_path, url.split(
            '/')[-1].replace(".htm", "")+".html")
        with open(html_to_file, "w", encoding='utf-8') as file:
            file.write(BeautifulSoup(html_content, 'html.parser').prettify())

        xpaths = gen_paths
        tables = get_xpath_elements(driver, xpaths)
        if os.path.exists(spec_path):
            with open(spec_path) as file:
                xpaths = [line.rstrip() for line in file.readlines()]
                # spec_paths.extend(xpaths)
            tables = get_xpath_elements(driver, xpaths)
        logger.debug(f"USING XPATHS - {xpaths}")

        tables = sorted(tables, key=lambda table: table.location['y'])
        tables = remove_duplicate_element(tables)
        if not os.path.exists(args.save_image_path):
            os.mkdir(args.save_image_path)

        for i, table in enumerate(tables):

            if os.path.exists(os.path.join(ROOT_PATH, args.cik, table_date, f"{table_title.replace(' ','_')}_{i}.csv")):
                continue

            if not os.path.exists(os.path.join(args.save_image_path, table_date)):
                os.mkdir(os.path.join(args.save_image_path, table_date))

            try:
                logger.info(
                    f"Taking screenshot {os.path.join(args.save_image_path, table_date, f'soi_table_{i}.png')}")
                ss = table.screenshot(os.path.join(
                    args.save_image_path, table_date, f"soi_table_{i}.png"))
            except Exception as e:
                logger.info(e)

            table = malformed_table(table.get_attribute("outerHTML"))

            dfs = pd.read_html(table.prettify(), displayed_only=False)
            if not dfs:
                logger.debug(f"NO TABLES - {dfs}")
                continue
            dfs[0].to_csv(os.path.join(ROOT_PATH, args.cik, table_date,
                          f"{table_title.replace(' ','_')}_{i}.csv"), encoding='utf-8')
        break
    driver.close()
    return


if __name__ == "__main__":
    """
    python .\extract_tables.py --cik 1501729 --url-csv urls/1501729.csv --x-path xpaths/1501729.txt   
    python .\extract_tables.py --cik 1396440 --url-csv urls/1396440.csv --x-path xpaths/1396440.txt
    python .\extract_tables.py --cik 1422183 --url-csv urls/1422183.csv --x-path xpaths/1422183.txt
    python .\extract_tables.py --cik 1490349 --url-csv urls/1490349.csv --x-path xpaths/1490349.txt
    python .\extract_tables.py --cik 1379785 --url-csv urls/1379785.csv --x-path xpaths/1379785.txt
    python .\extract_tables.py --cik 1490927 --url-csv urls/1490927.csv --x-path xpaths/1490927.txt
    python .\extract_tables.py --cik 1418076 --url-csv urls/1418076.csv --x-path xpaths/1418076.txt
    python .\extract_tables.py --cik 1544206 --url-csv urls/1544206.csv --x-path xpaths/1544206.txt
    python .\extract_tables.py --cik 1370755 --url-csv urls/1370755.csv --x-path xpaths/1370755.txt
    python .\extract_tables.py --cik 1326003 --url-csv urls/1326003.csv --x-path xpaths/1326003.txt
    python .\extract_tables.py --cik 1580345 --url-csv urls/1580345.csv --x-path xpaths/1580345.txt
    python .\extract_tables.py --cik 1535778 --url-csv urls/1535778.csv --x-path xpaths/1535778.txt

    python .\extract_tables.py --cik 1487918 --url-csv urls/1487918.csv --x-path xpaths/1487918.txt
    python extract_tables.py --cik 1512931 --url-csv urls/1512931.csv --x-path xpaths/1512931.txt --chrome-driver-path /usr/bin/chromedriver
    python .\extract_tables.py --cik 1372807 --url-csv urls/1372807.csv --x-path xpaths/1372807.txt --save-image-path table_images/1372807
    """
    args = arguements()
    logger = init_logger(args.cik)
    main()
    # test_xpath_elements(
    #     url='https://www.sec.gov/Archives/edgar/data/0001501729/000110465921102950/tm2124358-1_10q.htm',
    #     xpath='//div[font[contains(text(), "Schedule of Investments")]]/parent::div/parent::font/following-sibling::table'
    # )
