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
python3 /home/seluser/sec_filings/scrap_links.py
python3 /home/seluser/sec_filings/extract_tables.py
python3 /home/seluser/sec_filings/consolidate_tables.py


