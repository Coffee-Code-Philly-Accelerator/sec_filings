import os
import logging
import time
import pandas as pd 
import argparse
from rich.logging import RichHandler
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

def arguements()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Get filing links and dates')
    parser.add_argument(
        '--url_txt', type=str, required=False, 
        default='urls/1540855.txt',
        help='Sec url to get links from'
    )

    return parser.parse_args()

def main()->None:
    args = arguements()
    driver = webdriver.Firefox(executable_path="geckodriver.exe")
    with open(args.url_txt,'r') as f:
        urls = [url for url in f.read().splitlines()]
    print(urls)
    # driver.get(url)
    driver.close()
    return

if __name__ == "__main__":
    main()