import os
import re
import logging
import glob
import pandas as pd
import numpy as np
from collections import Counter
from functools import reduce
from fuzzywuzzy import process

from utils import arguements,init_logger


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
        'date',
        'subheaders'
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


def extract_subheaders(
    df:pd.DataFrame,
)->pd.DataFrame:
    result = df.apply(lambda row: pd.notna(row).sum() == 1, axis=1)
    idx = result[result].index.tolist()
    df['subheaders'] = 'no_subheader'
    if not idx:
        return df
    
    df.loc[idx[-1]:,'subheaders'] = df.iloc[idx[-1],0]
    for j,i in enumerate(idx[:-1]):
        df.loc[idx[j]:idx[j+1],'subheaders'] = df.iloc[i,0]
    if idx[0] == 0:
        logging.debug(f"\n{df.iloc[:,:5]}")
        logging.debug(df.index.tolist())
        return df
    df.drop(idx,axis=0,inplace=True,errors='ignore') # drop subheader row
    return df


def clean(
    file:str,
)->list:
    dirs = file.split('/')
    if  len(dirs) < 3 or '.csv' not in dirs[-1]:
        return
    df_cur = pd.read_csv(file)
    df_cur = df_cur.T.drop_duplicates().T
    df_cur = df_cur.iloc[1:,1:]
    if df_cur.shape[1] < 4:
        return
    df_cur = df_cur.dropna(how='all') # drop empty rows
    if df_cur.empty:
        return
    
    df_cur.reset_index(drop=True,inplace=True)
    important_fields,idx = get_key_fields(df_cur)
    df_cur.columns = important_fields
    df_cur = extract_subheaders(df_cur)
    df_cur['date'] = dirs[1]
    df_cur = merge_duplicate_columns(df_cur)
    
    cur_cols,standard_names = df_cur.columns.tolist(),standard_field_names()
    cols_to_drop = [col for col in cur_cols if col not in standard_names] 
    df_cur.drop(columns=cols_to_drop, errors='ignore',inplace=True) # drop irrelevant columns
    df_cur.drop(index=idx,inplace=True) # drop the column row
    # df_cur.dropna(inplace=True) # drop totals rows
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
            get_standard_name(col,standard_fields) for col in columns
        )
    return columns

def get_key_fields(
    fields:pd.DataFrame
)->tuple:
    important_fields = standard_field_names()
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

def process_totals(
    df:pd.DataFrame,
    totals_cols:list=['index','portfolio', 'subheaders', 'date', 'cost', 'value']
)->bool:
    return df.drop(totals_cols, axis=1).isna().all(axis=1)
    # df = df.drop(columns=totals_cols)
    # to_drop = []
    # for index, row in df.iterrows():
    #     if row.isna().all():
    #         to_drop.append(index)
    # return to_drop

def process_date(
    date:str,
)->dict:
    dfs = {}
    for file in os.listdir(os.path.join('csv',date)):
        df_cur = clean(os.path.join('csv',date,file))
        if df_cur is None or df_cur.empty:
            continue
        key = '_'.join(tuple(map(str,df_cur.columns.tolist()))).replace('/','_')
        key = key.replace(' ','_')
        if dfs.get(key) is None:
            dfs[key] = []
        dfs[key].append(df_cur)
        index_list = df_cur[df_cur.iloc[:,0].str.contains('total investments', case=False, na=False)].index.tolist()
        if index_list:
            break
     
    if not os.path.exists(f"csv/{date}/output"):
        os.mkdir(f"csv/{date}/output")   
    for t in dfs:
        if os.path.exists(f"csv/{date}/output/{t}.csv") or len(t.split('_')) < 6:
            continue
        result = pd.concat(dfs[t], axis=0,join='outer', ignore_index=True)
        result.to_csv(f"csv/{date}/output/{t}.csv")
        
def merge_duplicate_columns(
    df:pd.DataFrame,
)->pd.DataFrame:
    duplicate_cols = df.columns[df.columns.duplicated(keep=False)]
    for col_name in duplicate_cols.unique():
        duplicate_data = df.loc[:, df.columns == col_name]
        merged_data = duplicate_data.apply(lambda row: ' '.join(set(row.dropna().astype(str))), axis=1)
        df = df.loc[:, df.columns != col_name]
        df[col_name] = merged_data
    return df


def join_all_possible()->None:
    infile = 'csv/*/output/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs = [pd.read_csv(csv) for csv in all_csvs]
    merged_df = pd.concat(dfs)
    merged_df.drop(columns=merged_df.columns[0],inplace=True)
    merged_df.reset_index(inplace=True)
    merged_df.rename({'index':'original_index'})
    mask = process_totals(merged_df)
    logging.debug(f"TOTALS TO KEEP - {mask}")
    extracted_rows = merged_df.loc[mask]
    merged_df.drop(extracted_rows.index,inplace=True)
    
    logging.debug(f"final table shape - {merged_df.shape}")
    extracted_rows.to_csv('csv/totals.csv')
    merged_df.to_csv('csv/soi_table_all_possible_merges.csv')    
    return

def main()->None:
    import warnings
    warnings.filterwarnings("ignore")
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
    main()