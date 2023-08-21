import os
import logging
import shutil
import glob
import pandas as pd
# from scrap_links import init_logger

def key_fields()->tuple:
    """
    As of date
    Portfolio company name
    Cost
    Fair value
    Investment Type 
    Industry
    Date Acquired
    """
    return (
        'name',
        'portfolio',
        'cost',
        'fair',
        'value'
        # 'investment',
        # 'type',
        # 'industry',
        # 'acquired'
    )

def contains_keyword(cell):
    if isinstance(cell, str):
        return any(keyword in cell.lower() for keyword in key_fields())
    return False

def get_tables()->None:
    infile = 'csv/**/*/*'
    all_csvs = glob.glob(infile,recursive=True)
    if not os.path.exists("csv/to_process"):
        os.mkdir('csv/to_process')
    for file in all_csvs:
        if '.csv' not in file:
            continue
        logging.info(f"CHECKING FIELDS - {file}")
        df = pd.read_csv(file)
        has_keywords = df.applymap(lambda cell: any(keyword in str(cell) for keyword in key_fields())).values.any()        
        if has_keywords:
            shutil.copy2(file,os.path.join('csv','to_process','_'.join(file.split('/')[-2:])))
            logging.debug(f"HAS KEYWORDS - {file}")
        
        

def main()->None:
    init_logger()
    get_tables()
    return

if __name__ == "__main__":
    main()