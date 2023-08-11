import os
import json
import logging
import glob
import pandas as pd
import numpy as np
from collections import Counter
from scrap_links import init_logger,arguements

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

def debug_format(
    df:pd.DataFrame,
    out_path:str,
)->None:
    csv,date,debug,filename = out_path.split('/')
    if not os.path.exists(os.path.join(csv,date,debug)):
        os.mkdir(os.path.join(csv,date,debug))
    df.to_csv(out_path)
    return

def main()->None:
    init_logger()
    infile = 'csv/**/*/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs,columns = {},{}
    
    for file in all_csvs:
        dirs = file.split('/')
        if  len(dirs) < 3 or '.csv' not in dirs[-1]:
            continue
        
        df_cur = pd.read_csv(file)
        df_cur.dropna(axis=1,thresh=7,inplace=True)# allowable nan threshold
        df_cur.dropna(how='all',inplace=True)
        df_cur.fillna(-100,inplace=True) 
        df_cur = df_cur.iloc[1:,1:]
        if df_cur.shape[1] < 4:
            continue
        if (df_cur.iloc[0] != -100).all():
            debug_format(
                df=df_cur,
                out_path=f"csv/{dirs[1]}/debug/{file.split('/')[-1]}"
            )
        
        columns_to_drop = df_cur.columns[df_cur.iloc[0].isna()]
        df_cur = df_cur.drop(columns=columns_to_drop) # drops columns with nan columns name

        if df_cur.empty:
            continue
        
        if dfs.get(dirs[1]) is None and columns.get(dirs[1]) is None:
            dfs[dirs[1]] = []
            columns[dirs[1]] = []
            
        col = df_cur.iloc[0].tolist()
        # logging.debug(f"CUR COLUMNS - {col}")
        if col not in columns[dirs[1]]:
            columns[dirs[1]].append(col+[dirs[1],file])
            
        df_cur['date'] = dirs[1]
        dfs[dirs[1]].append(df_cur.reset_index(drop=True))

    with open('columns.json','w') as f:
        logging.debug(type(columns))
        json.dump(columns,f,indent=4)
    for t in dfs:
        result = pd.concat(dfs[t], axis=0,join='outer', ignore_index=True)
        logging.debug(f"COLUMNS - {columns[t]}")
        logging.debug(f"SHAPE - {result.shape}")
        result.columns = columns[t] + list(range(result.shape[1] - len(columns[t]))) 
        result.to_csv(f"csv/main_table_{t}.csv")
                
    return

if __name__ == "__main__":
    main()