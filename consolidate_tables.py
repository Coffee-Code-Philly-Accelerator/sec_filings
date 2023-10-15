import os
import re
import logging
import glob
import pandas as pd
import numpy as np
import itertools
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
        'principal amount', # TODO change stand names for more dynamic fuzzywuzzy matching
        'cost',
        'value',
        'investment',
        'date',
        'subheaders',
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
    result = df.apply(lambda row: pd.notna(row).sum() == 1 or len(set(row.tolist())) == 2 or 'Fair  Value  (d)' in row.tolist(), axis=1) # TODO fix me 
    idx = result[result].index.tolist()
    df['subheaders'] = 'no_subheader'
    logging.debug(f"SUBHEADER INDEXES - {idx}")
    if not idx:
        return df
    df.loc[idx[-1]:,'subheaders'] = df.iloc[idx[-1],0]
    for j,i in enumerate(idx[:-1]):
        df.loc[idx[j]:idx[j+1],'subheaders'] = df.iloc[i,0]
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
    df_cur = merge_duplicate_columns(df_cur)
    cur_cols,standard_names = df_cur.columns.tolist(),standard_field_names()
    cols_to_drop = [col for col in cur_cols if col not in standard_names] 
    df_cur.drop(columns=cols_to_drop, errors='ignore',inplace=True) # drop irrelevant columns
    df_cur.drop(index=idx,inplace=True) # drop the column row
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
        found = any(any(key in str(field).lower() for key in important_fields)for field in row[-1].dropna().tolist())
        if found and len(set(row[-1].dropna().tolist())) >= 5:
            fields = strip_string(row[-1].tolist(),standardize=found),idx
            return fields
    return strip_string(fields.iloc[0].tolist(),standardize=found),0

 
def get_standard_name(col, choices, score_cutoff=60):
    best_match, score = process.extractOne(col, choices)
    if score > score_cutoff:
        return best_match
    return col

def process_date(
    date:str,
)->dict:
    if not os.path.exists(f"csv/{date}/output"):
        os.mkdir(f"csv/{date}/output") 
    files = os.listdir(os.path.join('csv',date))
    files = sorted(
        files, 
        key=lambda file: int(file.split('_')[-1].replace(".csv","")) if file.split('_')[-1].replace(".csv","").isdigit() else 999
    )

    df_cur = clean(os.path.join('csv',date,files[0]))
    for i,file in enumerate(files[1:]):
        if df_cur is None or df_cur.empty:
            df_cur = clean(os.path.join('csv',date,file))
            continue
        df_cur.to_csv(f"csv/{date}/output/cleaned_{i}.csv")
        index_list = df_cur[df_cur.iloc[:,0].str.contains('total investments', case=False, na=False)].index.tolist()
        if index_list:
            break
        df_cur = clean(os.path.join('csv',date,file))

    cleaned = os.listdir(f'csv/{date}/output')
    cleaned = sorted(
        cleaned, 
        key=lambda file: int(file.split('_')[-1].replace(".csv","")) if file.split('_')[-1].replace(".csv","").isdigit() else 999
    )
    dfs = [
        pd.read_csv(os.path.join(f"csv/{date}/output",f"{file}")) 
        for file in cleaned
    ]        
    
    date_final = pd.concat(dfs,axis=0,join='outer', ignore_index=True)
    if not os.path.exists(f"csv/{date}/output_final"):
        os.mkdir(f"csv/{date}/output_final")
        
    date_final.drop(date_final.columns[0],axis=1,inplace=True)
    date_final = extract_subheaders(date_final)
    date_final['date'] = date
    date_final.to_csv(f"csv/{date}/output_final/{'_'.join(date_final.columns.tolist())}.csv")
    
            
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
    infile = 'csv/*/output_final/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs = [pd.read_csv(csv) for csv in all_csvs]
    merged_df = pd.concat(dfs)
    merged_df.drop(columns=merged_df.columns[0],inplace=True)
    merged_df.reset_index(inplace=True)
    merged_df.rename({'Index':'original_index'},inplace=True)
    
    '''
    # Seperates totals from soi tables
    mask = merged_df[merged_df['portfolio'].str.contains('total investments|liabilities|net assets|total equity/other',case=False,na=False)].index.to_numpy()
    extracted_rows = merged_df.loc[mask]
    merged_df.drop(extracted_rows.index,inplace=True)
    extracted_rows.dropna(axis=1,how='all').drop(['subheaders'],axis=1).to_csv('csv/totals.csv')
    '''

    logging.debug(f"final table shape - {merged_df.shape}")
    merged_df.to_csv('csv/soi_table_all_possible_merges.csv')    
    return

def validate_totals(
    soi:pd.DataFrame,
    totals:pd.DataFrame,
)->bool:
    totals = totals[totals['portfolio'].str.contains('total investments', case=False, na=False)][['date','cost','value']].reset_index()
    totals.cost = totals.cost.replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    totals.value = totals['value'].replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    
    soi.cost = soi.cost.str.replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    soi.value = soi.value.str.replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    soi_totals = soi.groupby(['date']).agg({'cost':'sum','value':'sum'}).reset_index()

    for i in range(soi_totals.shape[0]):
        try:
            assert np.allclose(
                soi_totals[['cost','value']].loc[i].to_numpy(), 
                totals[['cost','value']].loc[i].to_numpy(),
                atol=1000
            ),f"Test {totals['date'].loc[i]} - Failed"
            logging.info(f"Test {totals['date'].loc[i]} - Passed")
        except AssertionError as e:
            logging.error(e)
    
    totals.merge(
        soi_totals, 
        on='date', 
        how='inner',
        suffixes=('_published', '_aggregate')
    ).reset_index().drop(['index','level_0'],axis=1).to_csv('csv/totals_validation.csv',index=False)

def main()->None:
    if not os.path.exists('csv'):
        os.mkdir('csv')
    for date in os.listdir('csv'):
        if '.csv' in date:
            continue
        logging.info(f"DATE - {date}")
        process_date(date)
    join_all_possible()
    # validate_totals(pd.read_csv('csv/soi_table_all_possible_merges.csv'),pd.read_csv('csv/totals.csv'))
    return 

if __name__ == "__main__":
    # remove files that don't contain keyword
    # https://unix.stackexchange.com/questions/150624/remove-all-files-without-a-keyword-in-the-filename 
    # https://stackoverflow.com/questions/26616003/shopt-command-not-found-in-bashrc-after-shell-updation
    import warnings
    warnings.filterwarnings("ignore")
    init_logger()
    main()
