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
python3 scrap_links.py --cik 1512931 --url "https://www.sec.gov/edgar/browse/?CIK=1512931"
python3 extract_tables.py --cik 1512931 --url-csv urls/1512931.csv --x-path xpaths/1512931.txt
python3 consolidate_tables.py


