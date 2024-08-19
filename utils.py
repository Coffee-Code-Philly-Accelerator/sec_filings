import os
import sys
import re
import logging
import datetime
import argparse
import pandas as pd
from rich.logging import RichHandler

ROOT_PATH = os.getcwd()


def remove_row_duplicates(row):
    seen = set()
    return pd.Series([x if x not in seen and not seen.add(x) else None for x in row])


def present_substrings(substrings, main_string):
    check = list(filter(lambda sub: sub in main_string, substrings))
    if check:
        return check[0]
    return main_string


def make_unique(original_list):
    seen = {}
    unique_list = []

    for item in original_list:
        if item in seen:
            counter = seen[item] + 1
            seen[item] = counter
            unique_list.append(f"{item}_{counter}")
        else:
            seen[item] = 1
            unique_list.append(item)

    return unique_list


def concat(*dfs) -> list:
    final = []
    for df in dfs:
        final.extend(df.values.tolist())
    return final

# Function to extract date and convert to datetime object


def extract_date(file_path):
    # Extract date from file path (assuming date is always in 'YYYY-MM-DD' format)
    date_str = re.search(r'\d{4}-\d{2}-\d{2}', file_path).group()
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')


def debug_format(
    df: pd.DataFrame,
    out_path: str,
) -> None:
    csv, date, debug, filename = out_path.split('/')
    if not os.path.exists(os.path.join(csv, date, debug)):
        os.mkdir(os.path.join(csv, date, debug))
    df.to_csv(out_path)
    return


def arguements() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Get filing links and dates')
    parser.add_argument(
        '--cik',
        type=str,
        required=True,
        help='BDC CIK number'
    )
    parser.add_argument(
        '--url', type=str, required=False,
        default='https://www.sec.gov/edgar/browse/?CIK=1422183',
        help='Sec url to get links from'
    )
    parser.add_argument(
        '--url-csv', type=str, required=False,
        default='urls/1422183.csv',
        help='.txt to get links from'
    )
    parser.add_argument(
        '--firefox-driver-path', type=str, required=False,
        default="geckodriver.exe",
        help='path to your geckodriver.exe'
    )
    parser.add_argument(
        '--chrome-driver-path', type=str, required=False,
        # default="chromedriver_win32/chromedriver.exe",

        help='path to your geckodriver.exe'
    )
    parser.add_argument(
        '--firefox-path', type=str, required=False,
        default=r"C:\Program Files\WindowsApps\Mozilla.Firefox_116.0.3.0_x64__n80bbvh6b1yt2\VFS\ProgramFiles\Firefox Package Root\firefox.exe",
        help='path to your firefox.exe'
    )
    parser.add_argument(
        '--chrome-path', type=str, required=False,
        default=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        help='path to your chrome.exe'
    )
    parser.add_argument(
        '--x-path', type=str, required=False,
        default=r'C:\Users\pysol\Desktop\projects\sec_filings\xpaths\1422183.txt',
        help='path to your xpaths.txt that contains the xpaths to the soi tables'
    )
    parser.add_argument(
        '--save-image-path', type=str, required=False,
        default=r'table_images\1372807',
        help='path to your xpaths.txt that contains the xpaths to the soi tables'
    )

    return parser.parse_args()


def _init_logger() -> None:
    logger = logging.getLogger("rich")
    logger.setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)

    logging.basicConfig(level=logging.ERROR)  # Ignore warnings and below
    logging.getLogger("pandas").setLevel(logging.ERROR)

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


def init_logger(
    cik: int
) -> logging.Logger:
    # Set up logging
    logger = logging.getLogger(f"CIK=={cik}")
    logger.setLevel(logging.DEBUG)
    dir = os.getcwd()
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(dir, f"{cik}.log"))
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('[%(name)s:%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Initializing ok.")

    return logger
