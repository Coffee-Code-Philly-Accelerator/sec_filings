import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

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
    infile = 'csv/**/*/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs,columns = {},[]
    
    for file in all_csvs:
        dirs = file.split('/')
        if  len(dirs) < 3 or '.csv' not in dirs[-1]:
            continue
        # print(file)
        df_cur = pd.read_csv(file)
        df_cur.dropna(axis=1,thresh=7,inplace=True) # allowable nan threshold
        df_cur = df_cur.iloc[1:,1:]
        if df_cur.shape[1] < 4 or df_cur.empty:
            continue

        # df_cur.columns = [col if str(col) != 'nan' else i for i,col in enumerate(df_cur.iloc[0].tolist())]
        columns.extend(df_cur.iloc[0].dropna().unique().tolist())

        df_cur['date'] = dirs[1]
        if dfs.get(df_cur.shape[1]) is None:
            dfs[df_cur.shape[1]] = [df_cur.reset_index(drop=True)]
        else:
            print(df_cur.shape)
            dfs[df_cur.shape[1]].append(df_cur.reset_index(drop=True))
        
    columns = Counter(columns).most_common(7)
    print(f"COMMON COLUMNS\n{columns}")
    print(dfs.keys())
    for t in dfs:
        result = pd.concat(dfs[t], axis=0, ignore_index=True)
        print(result.head())  
        result.to_csv(f"main_table_{t}.csv")
    return

if __name__ == "__main__":
    main()