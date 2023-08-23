import os
import re
import logging
import glob
import pandas as pd
import numpy as np
from collections import Counter
from fuzzywuzzy import process

from scrap_links import init_logger,arguements
from important_tables import key_fields


def standard_field_names()->tuple:
    """
    [('AmortizedCost', 424), ('Industry', 420), ('Portfolio Company(a)', 411), ('Footnotes', 325), ('Maturity', 265), ('Rate(b)', 177), ('FairValue(c)', 171)]
    """
    return (
        'portfolio',
        'footnotes',
        'industry',
        'rate',
        'floor',
        'maturity',
        'principal',
        'cost',
        'value',
        'investment',
        'date'
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
    if df_cur.empty:
        return
    return df_cur

def present_substrings(substrings, main_string):
    check = list(filter(lambda sub: sub in main_string, substrings))
    if check:
        return check[0]
    return main_string

def strip_string(
    columns_names:list,
    standardize:bool=False
)->tuple:
    columns = tuple(map(lambda col:re.sub(r'[^a-z]', '', str(col).lower()),columns_names))
    if standardize:
        standard_fields = standard_field_names()
        return tuple(
            # present_substrings(standard_fields,col) for col in columns 
            get_standard_name(col,standard_fields) for col in columns
        )
    return columns

def get_key_fields(
    fields:pd.DataFrame
)->tuple:
    important_fields = key_fields()
    for idx,row in enumerate(fields.iterrows()):
        found = any(any(key in str(field).lower() for key in important_fields)for field in row[-1].tolist())
        if found:
            fields = strip_string(row[-1].tolist(),standardize=found),idx
            # logging.debug(f"FUZZY FIELDS - {fields[0]}")
            return fields
    # logging.info("DEFAULT FIELDS")
    return strip_string(fields.iloc[0].tolist(),standardize=found),0

 
def get_standard_name(col, choices, score_cutoff=60):
    best_match, score = process.extractOne(col, choices)
    if score > score_cutoff:
        return best_match
    return col

def process_date(
    date:str,
)->dict:
    dfs = {}
    for file in os.listdir(os.path.join('csv',date)):
        df_cur = clean(os.path.join('csv',date,file))
        if df_cur is None:
            continue
        
        df_cur.reset_index(drop=True,inplace=True)
        important_fields,idx = get_key_fields(df_cur)
        df_cur.columns = important_fields
        df_cur['date'] = date
        df_cur = merge_duplicate_columns(df_cur)
        
        cur_cols,standard_names = df_cur.columns.tolist(),standard_field_names()
        cols_to_drop = [col for col in cur_cols if col not in standard_names] \
            if any(col in standard_names for col in cur_cols) else []
        df_cur.drop(columns=cols_to_drop, errors='ignore',inplace=True)
        
        key = '_'.join(tuple(map(str,df_cur.columns.tolist()))).replace('/','_')
        key = key.replace(' ','_')
        if dfs.get(key) is None:
            dfs[key] = []

        df_cur.drop(index=idx,inplace=True)
        dfs[key].append(df_cur)
     
    if not os.path.exists(f"csv/{date}/output"):
        os.mkdir(f"csv/{date}/output")   
    for t in dfs:
        if os.path.exists(f"csv/{date}/output/{t}.csv") or len(t) > 100:
            continue
        result = pd.concat(dfs[t], axis=0,join='outer', ignore_index=True)
        result.to_csv(f"csv/{date}/output/{t}.csv")
        
def merge_duplicate_columns(
    df:pd.DataFrame,
)->pd.DataFrame:
    duplicate_cols = df.columns[df.columns.duplicated(keep=False)]
    # Merge columns with the same name
    for col_name in duplicate_cols.unique():
        # Select columns with the same name
        duplicate_data = df.loc[:, df.columns == col_name]
        # Concatenate the values in these columns row-wise and store in a new 
        # logging.debug(f"{duplicate_data[0,:].dropna().astype(str)} {len(duplicate_data[0].dropna().astype(str))}")
        merged_data = duplicate_data.apply(lambda row: ' '.join(set(row.dropna().astype(str))), axis=1)
        # Drop the original duplicate columns
        df = df.loc[:, df.columns != col_name]
        # Add the new merged column
        df[col_name] = merged_data
    return df


def join_all_possible()->None:
    infile = 'csv/**/*/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs = {}
    
    for file in all_csvs:
        dirs = file.split('/')
        df_cur = clean(file)
        if df_cur is None:
            continue
        
        df_cur.reset_index(drop=True,inplace=True)
        important_fields,idx = get_key_fields(df_cur)
        df_cur.columns = important_fields
        df_cur['date'] = dirs[1]
        df_cur.drop(df_cur.columns[-2],axis=1,inplace=True)
        
        logging.debug(f"DUPLICATED COLUMNS - {df_cur.columns[df_cur.columns.duplicated(keep=False)]}")
        df_cur = merge_duplicate_columns(df_cur)
        
        cur_cols,standard_names = df_cur.columns.tolist(),standard_field_names()
        cols_to_drop = [col for col in cur_cols if col not in standard_names] \
            if any(col in standard_names for col in cur_cols) else []
        df_cur.drop(columns=cols_to_drop, errors='ignore',inplace=True)
        
        key = '_'.join(tuple(map(str,df_cur.columns.tolist()))).replace('/','_')
        key = key.replace(' ','_')
        if dfs.get(key) is None:
            dfs[key] = []

        df_cur.drop(index=idx,inplace=True)
        dfs[key].append(df_cur)
        
    for t in dfs:
        if os.path.exists(f"csv/{t}.csv") or len(t) > 100:
            continue
        result = pd.concat(dfs[t], axis=0,join='outer', ignore_index=True)
        result.to_csv(f"csv/{t}.csv")
                
    return

def main()->None:
    import warnings
    warnings.filterwarnings("ignore")
    init_logger()
    # for date in os.listdir('csv'):
    #     if '.csv' in date:
    #         continue
    #     logging.info(f"DATE - {date}")
    #     process_date(date)
    join_all_possible()
    return 

if __name__ == "__main__":
    """
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    df[col_name] = merged_data
    consolidate_tables.py:160: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    """
    # remove files that don't contain keyword
    # https://unix.stackexchange.com/questions/150624/remove-all-files-without-a-keyword-in-the-filename 
    # https://stackoverflow.com/questions/26616003/shopt-command-not-found-in-bashrc-after-shell-updation
    main()