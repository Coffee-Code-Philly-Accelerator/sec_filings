# SEC filing automated scraper



## Prerequisites
The assumptions of this readme guide is that the individual who finds the programs in this repository useful would have a computer and have access to the internet. For python environment setup, visit [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) or  [Python installation guide](https://www.python.org/downloads/).

### Packages & Dependencies
* Python needs to be installed. Please reference the official documentation below
    * [Anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) 
    * [Python installation guide](https://www.python.org/downloads/)
* List of packages and dependencies are listed in the **requirements_windows.txt** file if you are using pip or the **environment.yml** if you are using conda.


## Getting Started

### If linux
1. chmod +x run.sh
2. Just run `./run.sh`.


`run.sh` runs the below scripts
```
python3 scrap_links.py --cik 1580345 --url "https://www.sec.gov/edgar/browse/?CIK=1580345"
python3 extract_tables.py --cik 1580345 --url-csv urls/1580345.csv --x-path xpaths/1580345.txt
python3 consolidate_tables.py
```
### If windows
Run the below one by one in the cmd terminal.
```
python3 scrap_links.py --cik 1580345 --url "https://www.sec.gov/edgar/browse/?CIK=1580345"
python3 extract_tables.py --cik 1580345 --url-csv urls/1580345.csv --x-path xpaths/1580345.txt
python3 consolidate_tables.py
```






