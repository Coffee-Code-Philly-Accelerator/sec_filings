# SEC filing automated scraper
* Updated 9/30/2023
## Introduction
* The script extracts Schedule of investment data tables from BDCs publically available from sec.gov 
* The current script uses the [selenium](https://selenium-python.readthedocs.io/installation.html#introduction) webdriver to extract and collect web data that is dynamically displayed in the disorganized DOM hierarchy 
* Relevant url links are scraped into urls/1422183.txt file
* Currently, 15149 observations are collected from BDC CIK=1422183.The csv folder contains the;
    * Downloaded html file of each BDC quarter 
    * Raw scraped data from all BDC quarters
    * Cleaned and standardized data tables for each quarter
    * 2nd tier normalized aggregates from all quarters into 1 table
    * "Totals" aggregates for each BDC quarter into 1 table
* For CIK=1379785, obervations are aggregated into an excel document, each quarter in its separate sheet in the excel document.

## Packages & Dependencies
* Most of the code is written in python
* [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) if you want to use conda is found here for both windows and linux operating systems
* [Python installation guide](https://www.python.org/downloads/) from python.org
* please list of packages and dependencies listed in the **requirements.txt** file if you are using pip or the **environment.yml** if you are using conda

## Core caveats & Warnings
* program works only on version selenium==3.141.0
* program works using a chrome driver
* selenium==3.141.0 works for chrome version 114 or below
    * [List of .deb files](http://mirror.cs.uchicago.edu/google-chrome/pool/main/g/google-chrome-stable/) for different chrome versions from U Chicago
#### Containerization of chrome version binaries and progam dependencies via Docker in next update

## Steps to run scraping scripts
Future updates will have the setup installation automated in a bashscript .sh file
#### Conda steps to run
1. to setup dependencies via conda for your operating system, follow the [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html)
2. once conda is setup, cd to "sec_filings" and run
    *  ```conda env create --name [environment_name] --file environment.yml```
    * [environment_name] is what ever you want to name your environment

#### Pip steps to run
1. to setup dependencies via pip, reference the [Python installation guide](https://www.python.org/downloads/) from python.org
2. run ```python3 -m venv [environment_name]```
    * [environment_name] is what ever you want to name your environment
3. cd to sec_filings
    *  run ```pip install -r requirements.txt``` to install all the python dependencies to run the scraping scripts

#### After dependencies have been setup
1. Download the selenium chrome driver for chrome version 114 or below specific to your operating system from [selenium](https://selenium-python.readthedocs.io/installation.html#introduction)
    * if windows, should have .exe
    * if linux or mac, does not need .exe
2.  make sure your chrome browser version is 114 or below
3. run ```python3 scrap_links.py```
4. run ```python3 extract_tables.py```
5. run ```consolidate_tables.py```
#### To scrap data from CIK=1379785,
1. cd to bbdc_extraction/ directory
2. run "python3 BBDC_Extraction.py"

## Output 
After running all the scripts, there should be a ```urls/``` and ```csv/``` directory created with all the BDC quarter dates, raw data tables, cleaned data tables for each quarter, the merged 2nd tier normalized table of all quarters and the totals aggregates for each quarter in one table.

The script should also automatically run unit tests to assert whether the scraped aggregate totals match the totals statistic available in each quarter.



