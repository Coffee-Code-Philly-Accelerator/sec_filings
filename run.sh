#!/bin/bash

# Below are general instructions
# https://docs.anaconda.com/free/anaconda/install/index.html <- installing conda to your OS document here


# To run below run.sh file
# either use WSL via windows, setup docker and run in linux via docker, or use UNIX like OS with bash
# chmod +x run.sh
# ./run.sh


# create conda environment
# conda env create --file environment.yml -n sec
# conda activate sec

# run scripts
python3 scrap_links.py --cik 1580345 --url https://www.sec.gov/edgar/browse/?CIK=1580345
python3 extract_tables.py --cik 1580345 --url-csv urls/1580345.csv --x-path xpaths/1580345.txt
cd 1580345/
python3 1580345.py


