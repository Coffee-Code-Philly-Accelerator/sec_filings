# SEC filing automated scraper
* Updated 10/8/2023
## Introduction
The below is an overview of the [github](https://github.com/Tony363/sec_filings) repository's deliverables. After running all the programs, there should be a ```urls/``` and ```csv/``` directory created with all the BDC quarter dates, raw data tables, cleaned data tables for each quarter, the merged 2nd tier normalized table of all quarters and the totals aggregates for each quarter in one table.
The script should also automatically run unit tests to assert whether the scraped aggregate totals match the totals statistic available in each quarter.

![](https://hackmd.io/_uploads/ryiURCeWp.png)

* The program extracts Schedule of investment data tables from BDCs publically available from sec.gov 
* The current script uses the [selenium](https://selenium-python.readthedocs.io/installation.html#introduction) webdriver to extract and collect web data that is dynamically displayed in the disorganized DOM hierarchy 
* Relevant url links are scraped into urls/1422183.txt file
* Currently, 15149 observations are collected from BDC CIK=1422183.The csv folder contains the;
    * Downloaded html file of each BDC quarter 
    * Raw scraped data from all BDC quarters
    * Cleaned and standardized data tables for each quarter
    * 2nd tier normalized aggregates from all quarters into 1 table
    * "Totals" aggregates for each BDC quarter into 1 table
* For CIK=1379785, obervations are aggregated into an excel document, each quarter in its separate sheet in the excel document.

![](https://hackmd.io/_uploads/HkTB0Ag-a.png)


## Packages & Dependencies
* Most of the code is written in python
* [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) 
* [Python installation guide](https://www.python.org/downloads/)
* List of packages and dependencies are listed in the **requirements.txt** file if you are using pip or the **environment.yml** if you are using conda

## Warnings & Caveats
* program works only on version selenium==3.141.0
* program works using a chrome driver
* selenium==3.141.0 works for chrome version 114 or below
    * [Download links of .deb files](http://mirror.cs.uchicago.edu/google-chrome/pool/main/g/google-chrome-stable/) for different chrome versions from U Chicago
    * the chrome driver version should start with the same first 3 digits of your chrome version
        * i.e. chrome driver version 114.0.5735.90 supports chrome version 114
* You need to know where your chrome binaries and chrome drivers are located on your computer

#### Containerization of chrome version binaries and progam dependencies via Docker comes in next update
## Program Arguement Variables
* [environment_name] is what ever you want to name your environment
* [url] = the url of where your BDCs investment docuemnts are archived
* [chrome_path] = the path of where your chrome binaries are located
    * i.e. windows -> ```C:\Program Files\Google\Chrome\Application\chrome.exe```
* [chrome_driver_path] = path to where your chrome driver is
    * i.e. windows - > ```C:\Users\pysol\Desktop\projects\sec_filings\chromedriver_win32\chromedriver.exe```

## Getting Started
Future updates will have the setup installation automated in a bashscript .sh file
#### 1a. Conda steps to run
* to setup dependencies via conda for your operating system, follow the [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html)
* once conda is setup, cd to "sec_filings" 
* run```conda env create --name [environment_name] --file environment.yml```

#### 1b. Pip steps to run
* to setup dependencies via pip, reference the [Python installation guide](https://www.python.org/downloads/) from python.org
* run ```python3 -m venv [environment_name]```
* cd to sec_filings
* run ```pip install -r requirements.txt``` to install all the python dependencies to run the scraping scripts

#### 2. After dependencies have been setup
* Download the selenium chrome driver for chrome version 114 or below specific to your operating system from [this website](https://sites.google.com/chromium.org/driver/downloads?authuser=0)
* make sure your chrome browser version is 114 or below
* run ```python3 scrap_links --url [url] --chrome_path [chrome_path] --chrome_driver_path [chrome_driver_path]```
* run ```python3 extract_tables.py --chrome_path [chrome_path] --chrome_driver_path [chrome_driver_path]```
* run ```python3 consolidate_tables.py --chrome_path [chrome_path] --chrome_driver_path [chrome_driver_path]```
#### 3. To scrap data from CIK=1379785,
* cd to ```bbdc_extraction/``` directory
* run ```python3 BBDC_Extraction.py```

## FAQ
There are currently no known issues with the programs. For any future issues or trouble shoot necessary, do not hesitate the open an issue in this github repository. I will reply as soon as I can.




