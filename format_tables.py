import os
import logging
import glob
import pandas as pd
import numpy as np
from collections import Counter
from scrap_links import init_logger

def common_fields()->tuple:
    """
    [('AmortizedCost', 424), ('Industry', 420), ('Portfolio Company(a)', 411), ('Footnotes', 325), ('Maturity', 265), ('Rate(b)', 177), ('FairValue(c)', 171)]
    """
    return (
        'AmortizedCost',
        'Industry',
        'Portfolio Company(a)',
        'Footnotes',
        'Maturity',
        'Rate(b)',
        'FairValue(c)'
    )


def main()->None:
    init_logger()
    infile = 'csv/**/*/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs,columns = {},[]
    
    for file in all_csvs:
        dirs = file.split('/')
        if  len(dirs) < 3 or '.csv' not in dirs[-1]:
            continue
        df_cur = pd.read_csv(file)
        df_cur.dropna(axis=1,thresh=7,inplace=True) # allowable nan threshold
        df_cur = df_cur.iloc[1:,1:]
        if df_cur.shape[1] < 4:
            continue
        logging.info(f"PROCESSING - {file}")
        columns_to_drop = df_cur.columns[df_cur.iloc[0].isna()]
        df_cur = df_cur.drop(columns=columns_to_drop)
        # df_cur.columns = [col if str(col) != 'nan' else i for i,col in enumerate(df_cur.iloc[0].tolist())]
        df_cur.columns = df_cur.iloc[0].tolist()
        # print(df_cur.head())
        df_cur.drop(index=1,inplace=True)

        if df_cur.empty:
            continue
        columns.extend(df_cur.iloc[0].tolist())
        
    
        df_cur['date'] = dirs[1]
        if dfs.get(df_cur.shape[1]) is None:
            dfs[df_cur.shape[1]] = []
        dfs[df_cur.shape[1]].append(df_cur.reset_index(drop=True))
    
    logging.info(f"ALL UNIQUE COLUMNS - {set(columns)}")
    with open("common_columns.txt",'w') as f:
        columns = Counter(columns).most_common(7)
        for col in columns:
            f.write(str(col))
        
    logging.info(f"COMMON COLUMNS top k\n{columns}")
    for t in dfs:
        try:
            result = pd.concat(dfs[t], axis=0,join='outer', ignore_index=True)
            result.to_csv(f"csv/main_table_{t}.csv")
        except Exception as e:
            logging.debug(f"COL {t}\n{e}")
            # for df in dfs[t]:
            #     logging.debug(df.head())
                
    return

if __name__ == "__main__":
    main()