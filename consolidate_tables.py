import os
import logging
import glob
import pandas as pd
import numpy as np
from collections import Counter
from scrap_links import init_logger,arguements
from important_tables import key_fields


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

def make_unique(original_list):
    seen = {}
    unique_list = []
    
    for item in original_list:
        if item in seen:
            counter = seen[item] + 1
            seen[item] = counter
            unique_list.append(f"{item}_{counter}")
        else:
            seen[item] = 1
            unique_list.append(item)
    
    return unique_list

def debug_format(
    df:pd.DataFrame,
    out_path:str,
)->None:
    csv,date,debug,filename = out_path.split('/')
    if not os.path.exists(os.path.join(csv,date,debug)):
        os.mkdir(os.path.join(csv,date,debug))
    df.to_csv(out_path)
    return

def clean(
    file:str,
)->list:
    if 'main' in file:
        return
    dirs = file.split('/')
    if  len(dirs) < 3 or '.csv' not in dirs[-1]:
        return
    df_cur = pd.read_csv(file)
    df_cur = df_cur.T.drop_duplicates().T
    df_cur.dropna(axis=1,thresh=7,inplace=True) # allowable nan threshold
    df_cur = df_cur.iloc[1:,1:]
    if df_cur.shape[1] < 4:
        return
    columns_to_drop = df_cur.columns[df_cur.iloc[0].isna()]
    df_cur = df_cur.drop(columns=columns_to_drop)
    df_cur = df_cur.dropna(how='all')
    df_cur = df_cur.fillna(-100)
    if df_cur.empty:
        return
    return df_cur

def get_key_fields(
    fields:pd.DataFrame
)->list:
    important_fields = key_fields()
    for idx,row in enumerate(fields.iterrows()):
        if any(any(key in str(field).lower() for key in important_fields)for field in row[-1].tolist()):
            logging.debug(f"FOUND FIELDS - {row[-1].tolist()}")
            return row[-1].tolist(),idx
    logging.info("DEFAULT FIELDS")
    return fields.iloc[0].tolist(),0


def process_date(
    date:str,
)->dict:
    dfs,columns = {},{}
    for file in os.listdir(os.path.join('csv',date)):
        df_cur = clean(os.path.join('csv',date,file))
        if df_cur is None:
            continue
        
        df_cur['date'] = date
        df_cur.reset_index(drop=True,inplace=True)
        important_fields,idx = get_key_fields(df_cur)
        df_cur.columns = important_fields
        key = '_'.join(tuple(map(str,df_cur.columns.tolist()))).replace('/','_')
        key = key.replace(' ',"_")
        if dfs.get(key) is None:# and columns.get(df_cur.shape[1]) is None:
            dfs[key] = []
            # columns[df_cur.shape[1]] = make_unique(
            #     list(map(
            #         lambda col:str(col).lower().replace(" ","_"),important_fields[:-1]
            #     ))
            # ) + ['date']
            
        df_cur.drop(index=idx,inplace=True)
        dfs[key].append(df_cur)
        
    for t in dfs:
        if os.path.exists(f"csv/{date}/{t}.csv") or len(t) > 100:
            continue
        result = pd.concat(dfs[t], axis=0,join='outer', ignore_index=True)
        # result.columns = columns[t] + list(range(result.shape[1] - len(columns[t]))) 
        result.to_csv(f"csv/{date}/{t}.csv")
        
        
        
def join_all_possible()->None:
    infile = 'csv/**/*/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs,columns = {},{}
    
    for file in all_csvs:
        dirs = file.split('/')
        df_cur = clean(file)
        if df_cur is None:
            continue
        
        df_cur['date'] = dirs[1]
        df_cur.reset_index(drop=True,inplace=True)
        important_fields,idx = get_key_fields(df_cur)
        df_cur.columns = important_fields
        key = '_'.join(tuple(map(str,df_cur.columns.tolist()))).replace('/','_')
        key = key.replace(' ',"_")
        if dfs.get(key) is None:# and columns.get(df_cur.shape[1]) is None:
            dfs[key] = []
            # columns[df_cur.shape[1]] = make_unique(
            #     list(map(
            #         lambda col:str(col).lower().replace(" ","_"),important_fields[:-1]
            #     ))
            # ) + ['date']
            
        df_cur.drop(index=idx,inplace=True)
        dfs[key].append(df_cur.reset_index(drop=True))
        
    for t in dfs:
        if os.path.exists(f"csv/{t}.csv") or len(t) > 100:
            continue
        result = pd.concat(dfs[t], axis=0,join='outer', ignore_index=True)
        # result.columns = columns[t] + list(range(result.shape[1] - len(columns[t]))) 
        result.to_csv(f"csv/{t}.csv")
                
    return

def main()->None:
    init_logger()
    for date in os.listdir('csv'):
        if '.csv' in date:
            continue
        logging.info(f"DATE - {date}")
        process_date(date)
    join_all_possible()
    return 

if __name__ == "__main__":
    # remove files that don't contain keyword
    # https://unix.stackexchange.com/questions/150624/remove-all-files-without-a-keyword-in-the-filename 
    # https://stackoverflow.com/questions/26616003/shopt-command-not-found-in-bashrc-after-shell-updation
    
    """
    Traceback (most recent call last):
  File "consolidate_tables.py", line 168, in <module>
    main()
  File "consolidate_tables.py", line 161, in main
    join_all_possible()
  File "consolidate_tables.py", line 133, in join_all_possible
    key = '_'.join(df_cur.columns.tolist()).replace('/','_').replace(' ',"_")
TypeError: sequence item 3: expected str instance, numpy.float64 found
    """
    main()