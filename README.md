# SEC filing automated scraper
## Introduction
The below is an overview of the [github](https://github.com/Tony363/sec_filings) repository's deliverables.

### Purpose & Benefits
Running the program in the [github repository](https://github.com/Tony363/sec_filings) would automate extraction of Business Development Company data from the sec.gov's EDGAR Application Programming Interface. The main data gained are;

#### Table 1. Example output of 2nd stage normalized single source of truth for BDC SOI tables
![Table 1](https://hackmd.io/_uploads/ryiURCeWp.png)



## Prerequisites
The assumptions of this readme guide is that the individual who finds the programs in this repository useful would have a computer and have access to the internet. 

This readme guide is intended for individuals who have the ability to read official documentation from the [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) or  [Python installation guide](https://www.python.org/downloads/) and be able to google potential solutions to the issues the individual face when running the programs in this github repository. Otherwise, if all things work well, there shouldn't be major learning hurdles.

### Packages & Dependencies
* Python needs to be installed. Please reference the official documentation below
    * [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) 
    * [Python installation guide](https://www.python.org/downloads/)
* List of packages and dependencies are listed in the **requirements.txt** file if you are using pip or the **environment.yml** if you are using conda


### Program Arguement Variables
* [--cik] = cik number to scrap data from
* [--url-csv] = csv of where cik 10q, 10k will be saved or acccessed from
* [--x-path] = .txt of xpath used for specific cik, typically xpaths/{cik}.txt
* [container_id] = id of the container that can be viewed via ```docker ps -a```
* [environment name] = arbitrary name you would like to call your virtual environment

## Using Docker Container to run code 

If setting up virtual environments in the inviduals local computer proves too complicated, a containerized virtualization of all the programs required in this reporsitory can be setup using the Docker technology. Please reference the [Docker Official Documentation](https://www.docker.com/) for further information about the technology. Otherwise skip to section *Getting Started*
* [Install](https://docs.docker.com/engine/install/) docker on your local computer and create an account on [dockerhub signup](https://hub.docker.com/signup)
* Then authenticate your docker credentials with ```docker login -u [your dockerhub username]``` and enter your password
* Run command ```docker run -it -u root pysolver33/sec-filings:10.27.2024/home/seluser/sec_filings/run.sh```
* Confirm that docker pulled container image and ran the container id with ```docker ps -a```
* Use container id to cp file to local computer, ```docker cp [container_id]:/home/seluser/sec_filings/csv/{cik}_soi_table.csv .```

* One can also access the docker container environment from the terminal via the command below or alternatively via vscode once the container is running;
    * ```docker run -it -u root pysolver33/sec-filings:10.27.2024 /bin/bash/```
    * [Guide](https://chatgpt.com/share/6716b6d0-be48-800e-b130-904efc43f327) to attach vscode IDE to docker container for debugging or development purposes

## Getting Started

#### 1a. Installing conda software dependency programs to run code
* To setup dependencies via conda for your operating system, follow the [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html)
* Change directory(```cd```) to *sec_filings* directory
* Run```conda env create --name [environment_name] --file environment.yml```

#### 1b. Installing pip software dependency programs to run code
* To setup dependencies via pip, reference the [Python installation guide](https://www.python.org/downloads/) from python.org
* Run ```python3 -m venv [environment_name]```
* Change directory(```cd```) to *sec_filings* directory
* Run ```pip install -r requirements_windows_dev.txt``` to install all the python dependencies to run the scraping scripts

#### 2. Below are steps to run the programs in this repository
* Download the selenium chrome driver for chrome version 114 or below specific to your operating system from [this website](https://sites.google.com/chromium.org/driver/downloads?authuser=0)
* Make sure your chrome browser version is 114 or below
  * or alternatively use [this docker image](https://hub.docker.com/r/selenium/standalone-chrome) with everything preinstalled
* Run ```python3 scrap_links.py --cik {cik} --url https://www.sec.gov/edgar/browse/?CIK={cik}```
* Run ```python3 extract_tables.py --chrome_path [chrome_path] --chrome_driver_path [chrome_driver_path]```
* Run ```python3 extract_tables.py --cik {cik} --url-csv urls/{cik}.csv --x-path xpaths/{cik}.txt```

## FAQ
There are currently no known issues with the programs. Do not hesitate to open an *issues* in this github repository. The *issues* tab is on the top left of the github repository web inferface. I will reply as soon as I can.
![Screenshot 2023-12-06 190510](https://hackmd.io/_uploads/SyD22KAHT.png)





