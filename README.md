# SEC filing automated scraper

Running the program in the [github repository](https://github.com/Tony363/sec_filings) would automate extraction of Business Development Company data from the sec.gov's EDGAR Application Programming Interface. 


## Prerequisites
Refer to installation of python from [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) or  [Python installation guide](https://www.python.org/downloads/).
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

## Getting Started

#### Using conda to setup virtual environment
* To setup dependencies via conda for your operating system, follow the [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html)
* Change directory(```cd```) to *sec_filings* directory
* Run```conda env create --name [environment_name] --file environment.yml```


#### Using pip to setup virtual environment
* To setup dependencies via pip, reference the [Python installation guide](https://www.python.org/downloads/) from python.org
* Run ```python3 -m venv [environment_name]```
* Change directory(```cd```) to *sec_filings* directory
* If linux run `source {env name}/bin/activate` else if windows run `{env name}\Scripts\activate`
* Then run ```pip install -r requirements_windows_dev.txt``` to install all the python dependencies to run the scraping scripts


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

#### Table 1. Example output of 2nd stage normalized single source of truth for BDC SOI tables
![Table 1](https://hackmd.io/_uploads/ryiURCeWp.png)

## FAQ
There are currently no known issues with the programs. Do not hesitate to open an *issues* in this github repository. The *issues* tab is on the top left of the github repository web inferface. I will reply as soon as I can.
![Screenshot 2023-12-06 190510](https://hackmd.io/_uploads/SyD22KAHT.png)





